import numpy as np
import random
from gym_lunar_rover.envs.Lunar_Rover_env import *
from gym_lunar_rover.envs.Utils import CustomReduceLROnPlateau
from collections import deque
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

class ExperienceReplayBuffer():
    def __init__(self, buffer_size, batch_size,):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)

    def add_exp(self, observation, info, action, reward, next_observation, next_info, done, state, next_state, value, prob):
        self.buffer.append((observation, info, action, reward, next_observation, next_info, done, state, next_state, value, prob))
    
    def sample_exp(self):
        # Ajusta batch_size si es mayor que el número de experiencias disponibles
        current_size = len(self.buffer)
        batch_size = min(self.batch_size, current_size)

        batch = random.sample(self.buffer, batch_size)
        observations, infos, actions, rewards, next_observations, next_infos, dones, states, next_states, values, probs = zip(*batch)
        return (np.array(observations), np.array(infos), np.array(actions), np.array(rewards), 
                np.array(next_observations), np.array(next_infos), np.array(dones),
                np.array(states), np.array(next_states), np.array(values), np.array(probs))
    
    def __len__(self):
        return len(self.buffer)

class MAPPOAgent:
    def __init__(self, map_shape, observation_shape, info_shape, action_dim, 
                 buffer_size, batch_size, warm_up_steps, clip_rewards, gamma, lamda,
                 lr, min_lr, lr_decay_factor, patience, cooldown, 
                 dropout_rate, l1_rate, l2_rate, 
                 actor_path=None, critic_path=None, buffer_path=None, parameters_path=None):
        
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.warm_up_steps = warm_up_steps
        self.clip_rewards = clip_rewards

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

        # Iniciamos el Experience Replay Buffer o cargamos el del entrenamiento previo
        self.replay_buffer = ExperienceReplayBuffer(buffer_size, batch_size)

        if buffer_path:
            self.load_buffer(buffer_path)

    def act(self, observation, info):
        # Ajustamos las dimensiones de la observación y la info
        observation = np.expand_dims(observation, axis=0)
        observation = np.expand_dims(observation, axis=-1)
        info = np.expand_dims(info, axis=0)

        # Predecimos las probabilidades y muestreamos una según la distribución categórica
        prob = self.actor([observation, info], training=False)
        dist = tfp.distributions.Categorical(probs=prob.numpy())
        action = dist.sample()
        return action

    
    def add_experience(self, observation, info, action, reward, next_observation, next_info, done, state, next_state, value, prob):
        if self.clip_rewards:
            reward = np.clip(reward, -1.0, 1.0)

        self.replay_buffer.add_exp(observation, info, action, reward, next_observation, next_info, done, state, next_state, value, prob)

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
        
    def save_buffer(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.replay_buffer.buffer, f)

    def load_buffer(self, file_path):
        with open(file_path, 'rb') as f:
            self.replay_buffer.buffer = pickle.load(f)

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

    def compute_discounted_rewards(self):
        pass

    #  Cálcular las recompensas acumuladas para el actor loss
    def compute_actor_loss(self, observations, infos, actions, advantages):
        probs = self.actor([observations, infos], training=True)
        dist = tfp.distributions.Categorical(probs=probs) # No se si hace falta .numpy() o no
        log_probs = dist.log_prob(actions)
        actor_loss = -tf.reduce_mean(log_probs * advantages)
        return actor_loss

    def compute_advantages(self):
        pass

    # Cálcular las ventajas con GAE para el critic loss
    def compute_critic_loss(self, states, discounted_rewards):
        values = self.critic(states, training=True)
        critic_loss = tf.reduce_mean(tf.square(discounted_rewards - values))
        return critic_loss

    def train(self):
        # Si aún no hemos terminado los steps de calentamiento no actualizamos la red
        if self.warm_up_steps > 0:
            self.warm_up_steps -= 1
            return 
        
        # Muestreo de experiencias
        observations, infos, actions, rewards, next_observations, next_infos, dones, states, next_states, values, probs = self.replay_buffer.sample_exp()
        
        # Calcular las recompensas acumuladas y ventajas
        discounted_rewards = self.compute_discounted_rewards(rewards, dones, self.gamma)
        advantages = self.compute_advantages(rewards, values, self.gamma, self.lamda)

        # Normalizar ventajas
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)

        # Actualizar el actor y el crítico
        with tf.GradientTape() as tape_actor, tf.GradientTape() as tape_critic:
            actor_loss = self.compute_actor_loss(observations, infos, actions, advantages)
            critic_loss = self.compute_critic_loss(states, discounted_rewards)

        # Aplicar gradientes
        actor_grads = tape_actor.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = tape_critic.gradient(critic_loss, self.critic.trainable_variables)
        
        self.act_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.cri_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
    

        # Actualizamos el valor de epsilon y lr
        self.update_counter += 1
        self.update_lr(actor_loss, "Actor")
        self.update_lr(critic_loss, "Critic") 

        return round(actor_loss.numpy(), 4), round(critic_loss.numpy(), 4)
        
    def save_train(self, actor_path, critic_path, buffer_path, state_path):
        self.save_models(actor_path, critic_path)
        self.save_buffer(buffer_path)
        self.save_parameters(state_path)

# Agente para la inferencia tras finalizar el entrenamiento
class InferenceMAPPOAgent:
    def __init__(self, observation_shape, info_shape, action_dim, model_path):
        self.action_dim = action_dim
        self.actor = actor_model(observation_shape, info_shape, action_dim)
        self.actor.compile(loss='mse', optimizer='adam')
        self.actor.load_weights(model_path)

    def act(self, observation, info):
        # Ajustamos las dimensiones de la observación y la info
        observation = np.expand_dims(observation, axis=0)
        observation = np.expand_dims(observation, axis=-1)
        info = np.expand_dims(info, axis=0)
        
        # Predecimos las probabilidades y cogemos la mejor
        prob = self.actor([observation, info], training=False)
        best_action = np.argmax(prob.numpy())
        return best_action