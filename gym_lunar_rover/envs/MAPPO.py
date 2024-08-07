import numpy as np
import random
from gym_lunar_rover.envs.Lunar_Rover_env import *
from gym_lunar_rover.envs.Utils import CustomReduceLROnPlateau
import pickle
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.regularizers import l1, l2, l1_l2 # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, Concatenate, BatchNormalization, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

def actor_model(observation_dim, info_dim, output_dim, dropout_rate=0, l1_rate=0, l2_rate=0):
    # Regularización L1, L2 o ambas combinadas
    if l1_rate!=0 and l2_rate!=0:
        reg = l1_l2(l1=l1_rate, l2=l2_rate)
    elif l1_rate!=0:
        reg = l1(l1_rate)
    elif l2_rate!=0:
        reg = l2(l2_rate)
    else:
        reg = None

    # Capa de entrada para la matriz de observación
    obs_input = Input(shape=(observation_dim, observation_dim, 1))  

    # Procesamiento de la observación
    obs_x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=reg)(obs_input)
    obs_x = BatchNormalization()(obs_x)
    obs_x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=reg)(obs_x)
    obs_x = BatchNormalization()(obs_x)
    obs_x = Flatten()(obs_x)
    obs_x = Dropout(dropout_rate)(obs_x) 

    # Capa de entrada para la matriz de observación
    info_input = Input(shape=(info_dim, ))

    # Procesamiento de la posición
    info_x = Dense(32, activation='relu', kernel_regularizer=reg)(info_input)
    info_x = BatchNormalization()(info_x)
    info_x = Dense(64, activation='relu', kernel_regularizer=reg)(info_x)
    info_x = BatchNormalization()(info_x)
    info_x = Dropout(dropout_rate)(info_x)

    # Concatenar la salida de la observación y la posición
    combined = Concatenate()([obs_x, info_x])

    # Capas densas después de la concatenación
    combined_x = Dense(64, activation='relu', kernel_regularizer=reg)(combined)
    combined_x = BatchNormalization()(combined_x)
    combined_x = Dense(128, activation='relu', kernel_regularizer=reg)(combined_x)
    combined_x = BatchNormalization()(combined_x)
    combined_x = Dropout(dropout_rate)(combined_x) 

    # Capa de policy
    policy = Dense(128, activation='relu', kernel_regularizer=reg)(combined_x)
    policy = BatchNormalization()(policy)
    policy = Dense(output_dim, activation='softmax')(policy)

    actor = Model(inputs=[obs_input, info_input], outputs=policy)
    return actor


def critic_model(map_dim, dropout_rate=0, l1_rate=0, l2_rate=0):
    # Regularización L1, L2 o ambas combinadas
    if l1_rate!=0 and l2_rate!=0:
        reg = l1_l2(l1=l1_rate, l2=l2_rate)
    elif l1_rate!=0:
        reg = l1(l1_rate)
    elif l2_rate!=0:
        reg = l2(l2_rate)
    else:
        reg = None

    # Capa de entrada para la matriz del estado
    map_input = Input(shape=(map_dim, map_dim, 1))  

    # Procesamiento del mapa del estado
    map_x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=reg)(map_input)
    map_x = BatchNormalization()(map_x)
    map_x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=reg)(map_x)
    map_x = BatchNormalization()(map_x)
    map_x = Flatten()(map_x)
    map_x = Dropout(dropout_rate)(map_x)

    # Capa de value
    value = Dense(128, activation='relu', kernel_regularizer=reg)(map_x)
    value = BatchNormalization()(value)
    value = Dense(1)(value)

    critic = Model(inputs=map_input, outputs=value)
    return critic

class ExperienceBuffer():
    def __init__(self, n_agents):
        self.buffers = [[] for _ in range(n_agents)]

    def add_exp(self, agent_id, observation, info, action, reward, done, available_action, state, value, old_prob):
        self.buffer[agent_id].append((observation, info, action, reward, done, available_action, state, value, old_prob))
    
    def get_all_exp(self):
        all_experiences = []
        for buffer in self.buffers:
        # Obtiene todas las experiencias en orden para cada agente
            observations, infos, actions, rewards, dones, available_actions, states, values, old_probs = zip(*buffer)
            all_experiences.append((np.array(observations), np.array(infos), np.array(actions), np.array(rewards), 
                                    np.array(dones), np.array(available_actions), np.array(states), np.array(values), np.array(old_probs)))
        return all_experiences

    def clear(self):
        self.buffers = [[] for _ in range(len(self.buffers))]

    def __len__(self):
        return sum(len(buffer) for buffer in self.buffers)

class MAPPOAgent:
    def __init__(self, map_shape, observation_shape, info_shape, action_dim, 
                 clip_rewards, gamma, lamda, clip,
                 lr, min_lr, lr_decay_factor, patience, cooldown, 
                 dropout_rate, l1_rate, l2_rate, 
                 actor_path=None, critic_path=None, parameters_path=None):
        
        self.action_dim = action_dim
        self.clip_rewards = clip_rewards

        # Parámetro de clip para 
        self.clip = clip

        # Parámetro de gamma para la importancia de las rewards futuras
        self.gamma = gamma

        # Parámetro de lamda para el cálculo del GAE
        self.lamda = lamda

        # Parámetros de lr para los ajustes del critic y el actor
        self.act_lr = lr
        self.cri_lr = lr

        # Contador de actualizaciones totales del entrenamiento
        self.update_counter = 0

        # Cargamos los parámetros si existe un entrenamiento previo
        if parameters_path:
            self.load_parameters(parameters_path)

        self.actor = actor_model(observation_shape, info_shape, action_dim, dropout_rate, l1_rate, l2_rate)
        self.critic = critic_model(map_shape, dropout_rate, l1_rate, l2_rate)

        # Si hay modelos guardados se cargan
        if actor_path and critic_path:
            self.load_models(actor_path, critic_path) 

        self.act_optimizer = Adam(learning_rate=self.act_lr)
        self.cri_optimizer = Adam(learning_rate=self.cri_lr)
        
        self.actor.compile(loss='mse', optimizer=self.act_optimizer)
        self.critic.compile(loss='mse', optimizer=self.cri_optimizer)

        # Callback para la reducción del lr si el loss no mejora
        self.reduce_lr_plateau_actor = CustomReduceLROnPlateau(self.act_optimizer, patience=patience, cooldown=cooldown, factor=lr_decay_factor, initial_lr=self.act_lr, min_lr=min_lr)
        self.reduce_lr_plateau_critic = CustomReduceLROnPlateau(self.cri_optimizer, patience=patience, cooldown=cooldown, factor=lr_decay_factor, initial_lr=self.cri_lr, min_lr=min_lr)

        # Iniciamos el Buffer
        self.buffer = ExperienceBuffer()

    def act(self, observation, state, info, available_actions):
        # Ajustamos las dimensiones de la observación y la info
        observation = np.expand_dims(observation, axis=0)
        observation = np.expand_dims(observation, axis=-1)
        info = np.expand_dims(info, axis=0)

        # Predecimos las probabilidades y con una máscara restringimos las acciones inválidas
        prob = self.actor([observation, info], training=False)
        
        mask = np.full(prob.shape, -np.inf)
        mask[0, available_actions] = 0
        masked_probs = prob + mask

        # Muestreamos una acción según la distribución categórica de las probs
        dist = tfp.distributions.Categorical(probs=masked_probs)

        # Muestreamos una acción según la distribución categórica
        action = dist.sample()

        # Obtenemos la probabilidad de la acción seleccionada
        action_prob = tf.reduce_sum(dist.prob(action)).numpy()

        # Obtenemos el valor del estado actual según el critic
        value = self.critic(state)

        return action, action_prob, value

    
    def add_experience(self, observation, info, action, reward, done, available_action, state, value, prob):
        if self.clip_rewards:
            reward = np.clip(reward, -1.0, 1.0)

        self.buffer.add_exp(observation, info, action, reward, done, available_action, state, value, prob)

    def update_lr(self, loss, act_cri):
        # Disminuimos el lr del critic o actor con la estrategia plateau
        match act_cri:
            case "Actor":
                self.act_lr = self.reduce_lr_plateau_actor.on_epoch_end(loss)

            case "Critic":
                self.cri_lr = self.reduce_lr_plateau_critic.on_epoch_end(loss)

    def save_models(self, file_path_act, file_path_cri):
        self.actor.save_weights(file_path_act)
        self.critic.save_weights(file_path_cri)

    def load_models(self, file_path_act, file_path_cri):
        self.actor.load_weights(file_path_act)
        self.critic.load_weights(file_path_cri)
        
    def save_parameters(self, file_path):
        state = {
            'act_lr': self.act_lr,
            'cri_lr': self.cri_lr,
            'update_counter': self.update_counter,
            'warm_up_steps': self.warm_up_steps
        }
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)
    
    def load_parameters(self, file_path):
        with open(file_path, 'rb') as f:
            state = pickle.load(f)
        self.act_lr = state['act_lr']
        self.cri_lr = state['cri_lr']
        self.update_counter = state['update_counter']
        self.warm_up_steps = state['warm_up_steps']

    def compute_discounted_rewards(self, rewards, dones, gamma):
        discounted_rewards = np.zeros_like(rewards)
        cumulative_reward = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                cumulative_reward = 0
            cumulative_reward = rewards[t] + gamma * cumulative_reward
            discounted_rewards[t] = cumulative_reward
        return discounted_rewards

    # Repasar este algoritmo
    def compute_advantages(self, rewards, values, gamma, lamda):
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0
        # Aqui las rewards se samplean de manera aleatoria por lo que no se 
        # si tiene sentido recorrerlas hacia atras, además hay rewards de todos
        # los agentes
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = last_gae_lam = delta + gamma * lamda * last_gae_lam
        return advantages
    
    #  Cálcular las recompensas acumuladas para el actor loss teniendo en cuenta solo las acciones válidas y el clip
    def compute_actor_loss(self, observations, infos, actions, advantages, availables_actions, old_probs):
        probs = self.actor([observations, infos], training=True)

        # Crear un tensor de índices para las acciones válidas
        indices = [[i, action] for i, actions in enumerate(availables_actions) for action in actions]
        indices = tf.constant(indices, dtype=tf.int32)

        # Crear y aplicar la máscara utilizando tf.scatter_nd
        mask = tf.scatter_nd(indices, tf.ones(len(indices)), probs.shape)
        masked_probs = probs * mask + (1 - mask) * -tf.float32.max  # Aplicar máscara con -inf para acciones no válidas

        # mask = np.full(probs.shape, -np.inf)
        # mask[0, availables_actions] = 0
        # masked_probs = probs + mask

        dist = tfp.distributions.Categorical(probs=masked_probs)
        log_probs = dist.log_prob(actions)
        probs_current = tf.exp(log_probs) 

        # Cálculamos el ratio entre las nuevas y las antiguas probs y realizamos el clip
        ratio = probs_current / old_probs
        clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip, 1 + self.clip)
        actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

        return actor_loss

    # Cálcular las ventajas con GAE para el critic loss
    def compute_critic_loss(self, states, discounted_rewards):
        values = self.critic(states, training=True)
        critic_loss = tf.reduce_mean(tf.square(discounted_rewards - values))
        return critic_loss

    def train(self):

        # Obtenemos las experiencias recolectadas con la política actual y limpiamos el buffer
        all_agents_experiences = self.buffer.get_all_exp()
        self.buffer.clear()

        actor_losses = []
        critic_losses = []

        for agent_exp in all_agents_experiences:
            observations, infos, actions, rewards, dones, available_actions, states, values, old_probs = agent_exp

            # Calcular las recompensas acumuladas y ventajas y normalizar estas últimas
            discounted_rewards = self.compute_discounted_rewards(rewards, dones, self.gamma)
            advantages = self.compute_advantages(rewards, values, self.gamma, self.lamda)
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)

            # Actualizar el actor y el crítico
            with tf.GradientTape() as tape_actor, tf.GradientTape() as tape_critic:
                actor_loss = self.compute_actor_loss(observations, infos, actions, advantages, available_actions, old_probs)
                critic_loss = self.compute_critic_loss(states, discounted_rewards)

            # Aplicar gradientes
            actor_grads = tape_actor.gradient(actor_loss, self.actor.trainable_variables)
            critic_grads = tape_critic.gradient(critic_loss, self.critic.trainable_variables)
        
            self.act_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            self.cri_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

            actor_losses.append(actor_loss.numpy())
            critic_losses.append(critic_loss.numpy())
    

        # Promediar las pérdidas de las actualizaciones
        # de los actores y críticos para cada agente
        avg_actor_loss = np.mean(actor_losses)
        avg_critic_loss = np.mean(critic_losses)

        # Actualizamos el valor de epsilon y lr
        self.update_counter += 1
        self.update_lr(avg_actor_loss, "Actor")
        self.update_lr(avg_critic_loss, "Critic") 

        return round(avg_actor_loss, 4), round(avg_critic_loss, 4)
        
    def save_train(self, actor_path, critic_path, state_path):
        self.save_models(actor_path, critic_path)
        self.save_parameters(state_path)

# Agente para la inferencia tras finalizar el entrenamiento
class InferenceMAPPOAgent:
    def __init__(self, observation_shape, info_shape, action_dim, model_path):
        self.action_dim = action_dim
        self.actor = actor_model(observation_shape, info_shape, action_dim)
        self.actor.compile(loss='mse', optimizer='adam')
        self.actor.load_weights(model_path)

    def act(self, observation, info, available_actions):
        # Ajustamos las dimensiones de la observación y la info
        observation = np.expand_dims(observation, axis=0)
        observation = np.expand_dims(observation, axis=-1)
        info = np.expand_dims(info, axis=0)
        
        # Predecimos las probabilidades y usamos una máscara para evitar las inválidas
        prob = self.actor([observation, info], training=False)

        mask = np.full(prob.shape, 0)
        mask[0, available_actions] = 0
        masked_probs = prob + mask

        # Cogemos la mejor acción dentro de las válidas
        return np.argmax(masked_probs)