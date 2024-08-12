import numpy as np
import random
from gym_lunar_rover.envs.Lunar_Rover_env import *
from gym_lunar_rover.envs.Utils import CustomReduceLROnPlateau
from collections import deque
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.regularizers import l1, l2, l1_l2 # type: ignore
from tensorflow.keras.layers import Layer, Dense, Flatten, Conv2D, Input, BatchNormalization, Dropout, Concatenate, ReLU # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

class ReduceMeanLayer(Layer):
    def call(self, inputs): 
        return tf.reduce_mean(inputs, axis=-1, keepdims=True)

def dddqn_model(observation_dim, info_dim, output_dim, dropout_rate=0, l1_rate=0, l2_rate=0):
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

    # Capa de valor
    value = Dense(128, activation='relu', kernel_regularizer=reg)(combined_x)
    value = Dense(1)(value)

    # Capa de ventaja
    advantage = Dense(128, activation='relu', kernel_regularizer=reg)(combined_x)
    advantage = Dense(output_dim)(advantage)

    # Capa ReduceMeanLayer para el cálculo de Q
    mean_advantage = ReduceMeanLayer()(advantage)

    # Cálculo de Q
    q = value + advantage - mean_advantage

    model = Model(inputs=[obs_input, info_input], outputs=q)
    return model

class ExperienceReplayBuffer():
    def __init__(self, buffer_size, batch_size,):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)

    def add_exp(self, observation, visits, info, action, reward, next_observation, next_visits, next_info, done, next_availables_actions):
        self.buffer.append((observation, visits, info, action, reward, next_observation, next_visits, next_info, done, next_availables_actions))
    
    def sample_exp(self):
        # Ajusta batch_size si es mayor que el número de experiencias disponibles
        current_size = len(self.buffer)
        batch_size = min(self.batch_size, current_size)

        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)
    
    def __len__(self):
        return len(self.buffer)

# Agente para el entrenamiento  
class DoubleDuelingDQNAgent:
    def __init__(self, observation_shape, info_shape, action_dim, buffer_size, batch_size,
                warm_up_steps, clip_rewards, epsilon, min_epsilon, epsilon_decay, gamma, 
                lr, min_lr, lr_decay_factor, patience, cooldown, 
                dropout_rate, l1_rate, l2_rate, update_target_freq, 
                model_path=None, buffer_path=None, parameters_path=None):
        
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.warm_up_steps = warm_up_steps
        self.clip_rewards = clip_rewards
        
        # Parámetro de gamma para la importancia de las rewards futuras
        self.gamma = gamma

        # Parámetro de lr para los ajustes del modelo
        self.lr = lr

        # Parámetros de epsilon para el balance entre exploración/explotación
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        # Contador de actualizaciones totales del entrenamiento
        self.update_counter = 0
        self.update_target_freq = update_target_freq

        # Cargamos los parámetros si existe un entrenamiento previo
        if parameters_path:
            self.load_parameters(parameters_path)
            if self.lr < min_lr:
                self.lr = min_lr
            if self.epsilon < min_epsilon:
                self.epsilon = min_epsilon

        self.primary_network = dddqn_model(observation_shape, info_shape, action_dim, dropout_rate, l1_rate, l2_rate)
        self.target_network = dddqn_model(observation_shape, info_shape, action_dim, dropout_rate, l1_rate, l2_rate)

        # Si hay modelos guardados se cargan
        if model_path:
            self.load_model(model_path) 

        # Si no hay pesos guardados se crean los modelos
        else:
            self.new_model()

        # self.primary_network.summary()
        # self.target_network.summary()
        
        # Callback para la reducción del lr si el loss no mejora
        self.reduce_lr_plateau = CustomReduceLROnPlateau(optimizer=self.optimizer, patience=patience, cooldown=cooldown, factor=lr_decay_factor, initial_lr=self.lr, min_lr=min_lr)
            
        # Iniciamos el Experience Replay Buffer o cargamos el del entrenamiento previo
        self.replay_buffer = ExperienceReplayBuffer(buffer_size, batch_size)

        if buffer_path:
            self.load_buffer(buffer_path)
    
    def act(self, observation, visits, info, available_actions):
        if np.random.rand() < self.epsilon:
            # Devolver una acción posible aleatoria
            return np.random.choice(available_actions)
        else:
            # Ajustamos las dimensiones de la observación y la info
            observation = np.expand_dims(observation, axis=0)
            observation = np.expand_dims(observation, axis=-1)
            visits = np.expand_dims(visits, axis=0)
            visits = np.expand_dims(visits, axis=-1)
            obs_visits = np.concatenate([observation, visits], axis=-1)
            info = np.expand_dims(info, axis=0)

            # Predecimos los Q-values en modo inferencia
            q_values = self.primary_network([obs_visits, info], training=False).numpy()
            
            # Implementamos una máscara para restringir las acciones inválidas
            mask = np.full(q_values.shape, -np.inf)
            mask[0, available_actions] = 0
            masked_q_values = q_values + mask

            # Cogemos la mejor acción dentro de las acciones válidas
            return np.argmax(masked_q_values)
    
    def add_experience(self, observation, visits, info, action, reward, next_observation, next_visits, next_info, done, available_actions_next):
        if self.clip_rewards:
            reward = np.clip(reward, -1.0, 1.0)

        self.replay_buffer.add_exp(observation, visits, info, action, reward, next_observation, next_visits, next_info, done, available_actions_next)

    def update_target_network(self):
        # Hard update a la target network con los pesos de la primary network
        self.target_network.set_weights(self.primary_network.get_weights())

    def update_epsilon(self):
        # Disminuimos el epsilon con una estrategia exponencial
        self.epsilon = self.min_epsilon + (self.initial_epsilon  - self.min_epsilon) * np.exp(-self.epsilon_decay * self.update_counter)
        #self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)

    def update_lr(self, loss):
        # Disminuimos el lr con la estrategia plateau
        new_lr = self.reduce_lr_plateau.on_epoch_end(loss)
        if new_lr!= self.lr:
            self.lr = new_lr

    def save_model(self, file_path):
        self.primary_network.save_weights(file_path)

    def load_model(self, file_path):
        self.primary_network.load_weights(file_path)
        self.update_target_network()
        self.optimizer = Adam(learning_rate=self.lr)
        self.primary_network.compile(loss='mse', optimizer=self.optimizer)
        self.target_network.compile(loss='mse', optimizer=self.optimizer)

    def new_model(self):
        self.optimizer = Adam(learning_rate=self.lr)
        self.primary_network.compile(loss='mse', optimizer=self.optimizer)
        self.target_network.compile(loss='mse', optimizer=self.optimizer)
        self.update_target_network()

    def save_buffer(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.replay_buffer.buffer, f)
            
    def load_buffer(self, file_path):
        with open(file_path, 'rb') as f:
            loaded_buffer = pickle.load(f)
            
            # Limita el buffer si se ha reducido su tamaño tras el entreno previo
            if len(loaded_buffer) > self.buffer_size:
                loaded_buffer = deque(list(loaded_buffer)[-self.buffer_size:], maxlen=self.buffer_size)
            
            self.replay_buffer.buffer = loaded_buffer

    def save_parameters(self, file_path):
        state = {
            'epsilon': self.epsilon,
            'lr': self.lr,
            'update_counter': self.update_counter,
            'warm_up_steps': self.warm_up_steps
        }
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)
    
    def load_parameters(self, file_path):
        with open(file_path, 'rb') as f:
            state = pickle.load(f)
        self.epsilon = state['epsilon']
        self.update_counter = state['update_counter']
        self.lr = state['lr']
        self.warm_up_steps = state['warm_up_steps']

    def train(self):

        # Si aún no hemos terminado los steps de calentamiento no actualizamos la red
        if self.warm_up_steps > 0:
            self.warm_up_steps -= 1
            return 
        
        observations, visits, infos, actions, rewards, next_observations, next_visits, next_infos, dones, next_availables_actions = self.replay_buffer.sample_exp()
        
        # Convertir arrays a tensores
        observations = tf.convert_to_tensor(observations, dtype=tf.float32)
        observations = tf.expand_dims(observations, axis=-1)
        visits = tf.convert_to_tensor(visits, dtype=tf.float32)
        visits = tf.expand_dims(visits, axis=-1)
        obs_visits = tf.concat([observations, visits], axis=-1)

        infos = tf.convert_to_tensor(infos, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)

        next_observations = tf.convert_to_tensor(next_observations, dtype=tf.float32)
        next_observations = tf.expand_dims(next_observations, axis=-1)
        next_visits = tf.convert_to_tensor(next_visits, dtype=tf.float32)
        next_visits = tf.expand_dims(next_visits, axis=-1)
        next_obs_visits = tf.concat([next_observations, next_visits], axis=-1)

        next_infos = tf.convert_to_tensor(next_infos, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        with tf.GradientTape() as tape:
            # Obtenemos los q-values del modelo principal para el estado actual
            q_values_primary = self.primary_network([obs_visits, infos], training=True)

            # Obtenemos los q-values del modelo y primary para el siguiente estado
            # Desactivamos regularización para contar con la red completa sin añadir ruido
            next_q_values_primary = self.primary_network([next_obs_visits, next_infos], training=False)
            next_q_values_target = self.target_network([next_obs_visits, next_infos], training=False)

            # Creamos un tensor de índices para saber las posiciones de las acciones válidas
            indices = [[i, action] for i, actions in enumerate(next_availables_actions) for action in actions]
            indices = tf.constant(indices, dtype=tf.int32)

            # Creamos y aplicamos una máscara para contar solo con las acciones válidas del siguiente estado
            mask = tf.scatter_nd(indices, tf.ones(len(indices)), [self.batch_size, self.action_dim])
            masked_next_q_values_primary = next_q_values_primary * mask - (1 - mask) * tf.float32.max
            masked_next_q_values_primary = tf.where(mask == 1, next_q_values_primary, -tf.float32.max)

            # Obtenemos la mejor acción en el siguiente estado desde la primary y su correspondiente valor de target
            next_best_actions_primary = tf.argmax(masked_next_q_values_primary, axis=1)
            target_next_q_values = tf.reduce_sum(next_q_values_target * tf.one_hot(next_best_actions_primary, self.action_dim), axis=1)

            # Calculamos el valor objetivo usando Double DQN
            target_q_values = rewards + (1 - dones) * self.gamma * target_next_q_values

            # Calculamos la pérdida
            action_masks = tf.one_hot(actions, self.action_dim)
            q_values_for_actions = tf.reduce_sum(q_values_primary * action_masks, axis=1)
            loss = tf.reduce_mean(tf.square(target_q_values - q_values_for_actions))
        
        # Calculamos y aplicamps el gradiente
        gradients = tape.gradient(loss, self.primary_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.primary_network.trainable_variables))

        # Si hemos llegado a la frecuencia de actualización de la target network
        # se copian los pesos de la network principal en la target
        self.update_counter += 1
        if self.update_counter % self.update_target_freq == 0:
            self.update_target_network()

        # Actualizamos el valor de epsilon y lr
        self.update_epsilon()
        self.update_lr(loss.numpy())
    
        return round(loss.numpy(), 4)
    
    def save_train(self, model_path, buffer_path, state_path):
        self.save_model(model_path)
        self.save_buffer(buffer_path)
        self.save_parameters(state_path)

# Agente para la inferencia tras finalizar el entrenamiento
class InferenceDDDQNAgent:
    def __init__(self, observation_shape, info_shape, action_dim, model_path):
        self.action_dim = action_dim
        self.primary_network = dddqn_model(observation_shape, info_shape, action_dim)
        self.primary_network.compile(loss='mse', optimizer='adam')
        self.primary_network.load_weights(model_path)

    def act(self, observation, visits, info, available_actions):
        # Ajustamos las dimensiones de la observación
        observation = np.expand_dims(observation, axis=0)
        observation = np.expand_dims(observation, axis=-1)
        visits = np.expand_dims(visits, axis=0)
        visits = np.expand_dims(visits, axis=-1)
        obs_visits = np.concatenate([observation, visits], axis=-1)
        info = np.expand_dims(info, axis=0)

        # Predecimos los Q-values en modo inferencia
        q_values = self.primary_network([obs_visits, info], training=False).numpy()
        
        # Implementamos una máscara para restringir las acciones inválidas
        mask = np.full(q_values.shape, -np.inf)
        mask[0, available_actions] = 0
        masked_q_values = q_values + mask

        # Cogemos la mejor acción dentro de las acciones válidas
        best_act = np.argmax(masked_q_values)
        return best_act