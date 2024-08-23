import numpy as np
from gym_lunar_rover.envs.Lunar_Rover_env import *
from gym_lunar_rover.envs.Utils import CustomReduceLROnPlateau, normalize_valid_probs
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
    obs_input = Input(shape=(observation_dim, observation_dim, 2))  

    # Procesamiento de la observación
    obs_x = Conv2D(16, (3, 3), activation='relu', kernel_regularizer=reg)(obs_input)
    obs_x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=reg)(obs_x)
    obs_x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=reg)(obs_x)
    obs_x = Flatten()(obs_x)
    obs_x = Dropout(dropout_rate)(obs_x)   

    # Capa de entrada para la matriz de observación
    info_input = Input(shape=(info_dim, ))

    # Procesamiento de la info
    info_x = Dense(16, activation='relu', kernel_regularizer=reg)(info_input)
    info_x = Dense(32, activation='relu', kernel_regularizer=reg)(info_x)
    info_x = Dense(64, activation='relu', kernel_regularizer=reg)(info_x)
    info_x = Dropout(dropout_rate)(info_x)  

    # Concatenar la salida de la observación y la posición
    combined = Concatenate()([obs_x, info_x])

    # Capas densas después de la concatenación
    combined_x = Dense(64, activation='relu', kernel_regularizer=reg)(combined)
    combined_x = Dense(128, activation='relu', kernel_regularizer=reg)(combined_x)
    combined_x = Dropout(dropout_rate)(combined_x) 

    # Capa de policy
    policy = Dense(128, activation='relu', kernel_regularizer=reg)(combined_x)
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

    # Capa de entrada con dos canales, uno con el estado actual y 
    # otro con el estado inicial sin los agentes
    combined_input = Input(shape=(map_dim, map_dim, 2))
    
    # Procesamiento de los mapas del estado
    combined_x = Conv2D(16, (3, 3), activation='relu', kernel_regularizer=reg)(combined_input)
    combined_x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=reg)(combined_x)
    combined_x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=reg)(combined_x)
    combined_x = Flatten()(combined_x)
    combined_x = Dropout(dropout_rate)(combined_x)

    # Capa de valor
    value = Dense(128, activation='relu', kernel_regularizer=reg)(combined_x)
    value = Dense(1)(value)

    critic = Model(inputs=combined_input, outputs=value)
    return critic

class ExperienceBuffer():
    def __init__(self, n_agents):
        self.buffers = [[] for _ in range(n_agents)]

    def add_exp(self, agent_id, observation, visits, info, action, reward, done, available_action, norm_rovers_state, norm_no_rovers_state, value, old_prob):
        self.buffers[agent_id].append((observation, visits, info, action, reward, done, available_action, norm_rovers_state, norm_no_rovers_state, value, old_prob))

    def get_agent_exps(self, agent_id):
        return zip(*self.buffers[agent_id])
    
    def clear(self):
        self.buffers = [[] for _ in range(len(self.buffers))]

    def __len__(self):
        return sum(len(buffer) for buffer in self.buffers)

class MAPPOAgent:
    def __init__(self, map_shape, observation_shape, info_shape, action_dim, n_agents,
                 clip_rewards, gamma, lamda, clip, entropy_coef,
                 lr, min_lr, lr_decay_factor, patience, cooldown, 
                 dropout_rate, l1_rate, l2_rate, 
                 actor_path=None, critic_path=None, parameters_path=None):
        
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.clip_rewards = clip_rewards

        # Parámetro de clip para 
        self.clip = clip

        # Parámetro de gamma para la importancia de las rewards futuras
        self.gamma = gamma

        # Parámetro de lamda para el cálculo del GAE
        self.lamda = lamda

        # Parámetro de entropía para añadir exploración
        self.entropy_coef = entropy_coef

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

        self.actor.compile(optimizer=self.act_optimizer)
        self.critic.compile(loss='mse', optimizer=self.cri_optimizer)

        # self.actor.summary()
        # self.critic.summary()

        # Callback para la reducción del lr si el loss no mejora
        self.reduce_lr_plateau_actor = CustomReduceLROnPlateau(self.act_optimizer, patience=patience, cooldown=cooldown, factor=lr_decay_factor, initial_lr=self.act_lr, min_lr=min_lr)
        self.reduce_lr_plateau_critic = CustomReduceLROnPlateau(self.cri_optimizer, patience=patience, cooldown=cooldown, factor=lr_decay_factor, initial_lr=self.cri_lr, min_lr=min_lr)

        # Iniciamos el Buffer
        self.buffer = ExperienceBuffer(self.n_agents)

    def act(self, observation, visits, norm_rovers_state, norm_no_rovers_state, info, available_actions):
        # Ajustamos las dimensiones de la observación y la info
        observation = np.expand_dims(observation, axis=0)
        observation = np.expand_dims(observation, axis=-1)
        visits = np.expand_dims(visits, axis=0)
        visits = np.expand_dims(visits, axis=-1)
        obs_visits = np.concatenate([observation, visits], axis=-1)
        info = np.expand_dims(info, axis=0)
        
        # Predecimos las probabilidades y con una máscara restringimos las acciones inválidas
        prob = self.actor([obs_visits, info], training=False)        
        
        # Aplicamos una máscara para contar solo con las acciones válidad
        mask = tf.reduce_any(tf.one_hot(available_actions, self.action_dim, on_value=True, off_value=False, dtype=tf.bool), axis=0)
        masked_prob = tf.where(mask, prob, 0)

        # Normalizamos las probabilidades de las acciones válidas para que sumen 1
        normalized_prob = normalize_valid_probs(masked_prob, mask)

        # Muestreamos una acción según la distribución categórica de las probs
        dist = tfp.distributions.Categorical(probs=normalized_prob)
        action = int(dist.sample().numpy())

        # Obtenemos la probabilidad de la acción seleccionada
        action_prob = tf.reduce_sum(dist.prob(action)).numpy()

        # Combinamos los estados para obtener el value del critic
        norm_rovers_state = np.expand_dims(norm_rovers_state, axis=0)
        norm_rovers_state = np.expand_dims(norm_rovers_state, axis=-1)
        norm_no_rovers_state = np.expand_dims(norm_no_rovers_state, axis=0)
        norm_no_rovers_state = np.expand_dims(norm_no_rovers_state, axis=-1)
        complete_state = np.concatenate([norm_rovers_state, norm_no_rovers_state], axis=-1)

        value = self.critic(complete_state, training=False)[0][0].numpy()

        return action, action_prob, value

    def add_experience(self, agent_id, observation, visits, info, action, reward, done, available_action, norm_rovers_state, norm_no_rovers_state, value, prob):
        if self.clip_rewards:
            reward = np.clip(reward, -1.0, 1.0)

        self.buffer.add_exp(agent_id, observation, visits, info, action, reward, done, available_action, norm_rovers_state, norm_no_rovers_state, value, prob)

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
            'update_counter': self.update_counter
        }
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)
    
    def load_parameters(self, file_path):
        with open(file_path, 'rb') as f:
            state = pickle.load(f)
        self.act_lr = state['act_lr']
        self.cri_lr = state['cri_lr']
        self.update_counter = state['update_counter']

    #  Cálcular las recompensas acumuladas para el critic loss
    def compute_discounted_rewards(self, rewards, dones, gamma):
        discounted_rewards = np.zeros_like(rewards)
        cumulative_reward = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                cumulative_reward = 0
            cumulative_reward = rewards[t] + gamma * cumulative_reward
            discounted_rewards[t] = cumulative_reward
        return discounted_rewards
    
    # Cálcular las ventajas con GAE para el actor loss
    def compute_advantages(self, rewards, values, dones, gamma, lamda):
        next_values = tf.concat([values[1:], tf.zeros_like(values[:1])], axis=0)
        deltas = rewards + (1 - dones) * gamma * next_values - values
        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + (1 - dones[t]) * gamma * lamda * gae
            advantages[t] = gae
        return advantages

    def compute_actor_loss(self, obs_visits, infos, actions, advantages, available_actions, old_probs):
        current_probs  = self.actor([obs_visits, infos], training=True)

        # Creamos y aplicamos la máscara utilizando 0 para acciones no válidas
        ragged_actions = tf.ragged.constant(available_actions)
        mask = tf.reduce_any(tf.one_hot(ragged_actions, self.action_dim, on_value=True, off_value=False, dtype=tf.bool), axis=1)
        masked_probs = tf.where(mask, current_probs, 0)

        # Normalizamos las probabilidades de las acciones válidas para que sumen 1 y
        # nos quedamos con las probs de las acciones elegidas
        normalized_probs = normalize_valid_probs(masked_probs, mask)
        action_probs = tf.gather(normalized_probs, actions, batch_dims=1)

        # Cálculamos el ratio entre las nuevas y las antiguas probs y realizamos el clip
        ratio = action_probs / old_probs

        clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip, 1 + self.clip)
        actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

        # Calculamos la entropía de la distribución de probabilidades actual
        dist = tfp.distributions.Categorical(probs=normalized_probs)
        entropy = dist.entropy()

        entropy_loss = -self.entropy_coef * tf.reduce_mean(entropy)

        # Pérdida total del actor incluyendo el término de entropía
        total_loss = actor_loss + entropy_loss

        return total_loss

    def compute_critic_loss(self, states, discounted_rewards):
        # Añadimos a los estados el estado inicial para obtener su value con el critic
        values = self.critic(states, training=True)
        critic_loss = tf.reduce_mean(tf.square(discounted_rewards - values))
        return critic_loss

    def train(self):

        actor_losses = []
        critic_losses = []

        # Inicializamos los acumuladores de gradientes
        total_actor_grads = [tf.zeros_like(var) for var in self.actor.trainable_variables]
        total_critic_grads = [tf.zeros_like(var) for var in self.critic.trainable_variables]

        # Recorremos las experiencias de los agentes de uno en uno
        for i in range(self.n_agents):
            # Comprobamos que el agente tenga experiencias
            if len(self.buffer.buffers[i])<=0:      
                continue

            observations, visits, infos, actions, rewards, dones, available_actions, norm_rovers_state, norm_no_rovers_state, values, old_probs = self.buffer.get_agent_exps(i)

            # Convertir arrays a tensores
            observations = tf.convert_to_tensor(observations, dtype=tf.float32)
            observations = tf.expand_dims(observations, axis=-1)
            visits = tf.convert_to_tensor(visits, dtype=tf.float32)
            visits = tf.expand_dims(visits, axis=-1)
            obs_visits = tf.concat([observations, visits], axis=-1)

            infos = tf.convert_to_tensor(infos, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.int32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)

            norm_rovers_state = tf.convert_to_tensor(norm_rovers_state, dtype=tf.float32)
            norm_rovers_state = tf.expand_dims(norm_rovers_state, axis=-1)
            norm_no_rovers_state = tf.convert_to_tensor(norm_no_rovers_state, dtype=tf.float32)
            norm_no_rovers_state = tf.expand_dims(norm_no_rovers_state, axis=-1)
            complete_states = tf.concat([norm_rovers_state, norm_no_rovers_state], axis=-1)

            values = tf.convert_to_tensor(values, dtype=tf.float32)
            old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)

            # Actualizar el actor y el crítico
            with tf.GradientTape() as tape_actor, tf.GradientTape() as tape_critic:
                # Calcular las recompensas acumuladas y ventajas y normalizar estas últimas
                discounted_rewards = self.compute_discounted_rewards(rewards, dones, self.gamma)
                advantages = self.compute_advantages(rewards, values, dones, self.gamma, self.lamda)
                advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)
                
                actor_loss = self.compute_actor_loss(obs_visits, infos, actions, advantages, available_actions, old_probs)
                critic_loss = self.compute_critic_loss(complete_states, discounted_rewards)

            # Calcular y acumular los gradientes          
            actor_grads = tape_actor.gradient(actor_loss, self.actor.trainable_variables)
            critic_grads = tape_critic.gradient(critic_loss, self.critic.trainable_variables)
        
            # Acumular los gradientes
            total_actor_grads = [grad_acc + grad for grad_acc, grad in zip(total_actor_grads, actor_grads)]
            total_critic_grads = [grad_acc + grad for grad_acc, grad in zip(total_critic_grads, critic_grads)]

            actor_losses.append(actor_loss.numpy())
            critic_losses.append(critic_loss.numpy())
        
        # Promediar los gradientes
        total_actor_grads = [grad / self.n_agents for grad in total_actor_grads]
        total_critic_grads = [grad / self.n_agents for grad in total_critic_grads]

        # Aplicar los gradientes promediados
        self.act_optimizer.apply_gradients(zip(total_actor_grads, self.actor.trainable_variables))
        self.cri_optimizer.apply_gradients(zip(total_critic_grads, self.critic.trainable_variables))

        # Promediamos las pérdidas de las actualizaciones
        # de los actores y críticos para cada agente
        avg_actor_loss = np.mean(actor_losses)
        avg_critic_loss = np.mean(critic_losses)

        # Limpiamos el buffer para recolectar nuevas experiencias con
        # la nueva política resultante tras el entrenamiento
        self.buffer.clear()

        # Actualizamos el valor de epsilon y lr
        self.update_counter += 1
        self.update_lr(avg_actor_loss, "Actor")
        self.update_lr(avg_critic_loss, "Critic") 

        return avg_actor_loss, round(avg_critic_loss, 4)
        
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
        probs = self.actor([observation, info], training=False)

        mask = np.full(probs.shape, -np.inf)
        mask[0, available_actions] = 0
        masked_probs = probs + mask

        # Cogemos la mejor acción dentro de las válidas
        return np.argmax(masked_probs)