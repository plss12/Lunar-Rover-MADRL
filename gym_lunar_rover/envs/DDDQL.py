import numpy as np
import random
from gym_lunar_rover.envs.Lunar_Rover_env import *
from collections import deque
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, Concatenate # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

def dddqn_model(observation_dim, info_dim, output_dim):
    # Capa de entrada para la matriz de observación
    obs_input = Input(shape=(observation_dim, observation_dim, 1))  

    # Procesamiento de la observación
    obs_x = Conv2D(32, (3, 3), activation='relu')(obs_input)
    obs_x = Conv2D(64, (3, 3), activation='relu')(obs_x)
    obs_x = Flatten()(obs_x)
    obs_x = Dense(128, activation='relu')(obs_x)

    # Capa de entrada para la matriz de observación
    info_input = Input(shape=(info_dim, ))

    # Procesamiento de la posición
    info_x = Dense(32, activation='relu')(info_input)
    info_x = Dense(64, activation='relu')(info_x)

    # Concatenar la salida de la observación y la posición
    combined = Concatenate()([obs_x, info_x])

    # Capas densas después de la concatenación
    combined_x = Dense(128, activation='relu')(combined)
    combined_x = Dense(128, activation='relu')(combined_x)

    # Capa de valor
    value = Dense(128, activation='relu')(combined_x)
    value = Dense(1)(value)

    # Capa de ventaja
    advantage = Dense(128, activation='relu')(combined_x)
    advantage = Dense(output_dim)(advantage)

    # Cálculo de Q
    q = value + advantage - tf.reduce_mean(advantage, axis=-1, keepdims=True)

    model = Model(inputs=[obs_input, info_input], outputs=q)
    return model

class ExperienceReplayBuffer():
    def __init__(self, buffer_size, batch_size,):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)

    def add_exp(self, observation, info, action, reward, next_observation, next_info, done):
        self.buffer.append((observation, info, action, reward, next_observation, next_info, done))
    
    def sample_exp(self):
        # Ajusta batch_size si es mayor que el número de experiencias disponibles
        current_size = len(self.buffer)
        batch_size = min(self.batch_size, current_size)

        batch = random.sample(self.buffer, batch_size)
        observations, infos, actions, rewards, next_observations, next_infos, dones = zip(*batch)
        return (np.array(observations), np.array(infos), np.array(actions), np.array(rewards), 
                np.array(next_observations), np.array(next_infos), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

# Agente para el entrenamiento  
class DoubleDuelingDQNAgent:
    def __init__(self, observation_shape, info_shape, action_dim, buffer_size, batch_size, gamma, lr, update_target_freq, model_path=None, buffer_path=None, parameters_path=None):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        # Parámetros de epsilon para el balance entre exploración/explotación
        self.epsilon = 1
        self.min_epsilon = 0.01
        self.epsilon_decay = 1e-4
        # Contador de actualizaciones
        self.update_counter = 0
        self.update_target_freq = update_target_freq

        # Cargamos los parámetros si existe un entrenamiento previo
        if parameters_path:
            self.load_parameters(parameters_path)

        self.primary_network = dddqn_model(observation_shape, info_shape, action_dim)
        self.target_network = dddqn_model(observation_shape, info_shape, action_dim)

        self.optimizer = Adam(learning_rate=lr)
        self.primary_network.compile(loss='mse', optimizer=self.optimizer)
        self.target_network.compile(loss='mse', optimizer=self.optimizer)

        # Si hay pesos guardados se cargan
        if model_path:
            self.load_model(model_path)

        # Si no hay pesos para cargar, se igualan los pesos iniciales de las redes
        else:
            self.update_target_network()
        
        # Iniciamos el Experience Replay Buffer o cargamos el del entrenamiento previo
        self.replay_buffer = ExperienceReplayBuffer(buffer_size, batch_size)

        if buffer_path:
            self.load_buffer(buffer_path)
        
    
    def act(self, observation, info, available_actions):
        if np.random.rand() < self.epsilon:
            # Devolver una acción posible aleatoria
            return np.int64(np.random.choice(available_actions))
        else:
            # Ajustamos las dimensiones de la observación y la info
            observation = np.expand_dims(observation, axis=0)
            observation = np.expand_dims(observation, axis=-1)
            info = np.expand_dims(info, axis=0)

            # Predecimos los Q-values y cogemos la acción con el mayor
            q_values = self.primary_network([observation, info])
            best_act = np.argmax(q_values[0].numpy())
            return best_act
    
    def add_experience(self, observation, info, action, reward, next_observation, next_info, done):
        self.replay_buffer.add_exp(observation, info, action, reward, next_observation, next_info, done)

    def update_target_network(self):
        # Hard update a la target network con los pesos de la primary network
        self.target_network.set_weights(self.primary_network.get_weights())

    def update_epsilon(self):
        # Disminuimos el epsilon según el epsilon decay y viendo que no baje del mínimo
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.min_epsilon else self.min_epsilon
        return self.epsilon
    
    def save_model(self, file_path):
        self.primary_network.save_weights(file_path)

    def load_model(self, file_path):
        self.primary_network.load_weights(file_path)
        self.target_network.set_weights(self.primary_network.get_weights())

    def save_buffer(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.replay_buffer.buffer, f)

    def load_buffer(self, file_path):
        with open(file_path, 'rb') as f:
            self.replay_buffer.buffer = pickle.load(f)

    def save_parameters(self, file_path):
        state = {
            'epsilon': self.epsilon,
            'update_counter': self.update_counter
        }
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)
    
    def load_parameters(self, file_path):
        with open(file_path, 'rb') as f:
            state = pickle.load(f)
        self.epsilon = state['epsilon']
        self.update_counter = state['update_counter']

    def train(self):

        observations, infos, actions, rewards, next_observations, next_infos, dones = self.replay_buffer.sample_exp()
        
        # Convertir arrays a tensores
        observations = tf.convert_to_tensor(observations, dtype=tf.float32)
        observations = tf.expand_dims(observations, axis=-1)
        infos = tf.convert_to_tensor(infos, dtype=tf.int32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_observations = tf.convert_to_tensor(next_observations, dtype=tf.float32)
        next_observations = tf.expand_dims(next_observations, axis=-1)
        next_infos = tf.convert_to_tensor(next_infos, dtype=tf.int32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Obtener Q-values del modelo principal para el estado actual y el siguiente estado
        with tf.GradientTape() as tape:
            q_values = self.primary_network([observations, infos])
            next_q_values = self.target_network([next_observations, next_infos])

            # Calcular el valor objetivo usando el Dueling Double DQN
            max_next_q_values = tf.reduce_max(next_q_values, axis=1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

            # Calcular la pérdida
            action_masks = tf.one_hot(actions, self.action_dim)
            q_values_for_actions = tf.reduce_sum(q_values * action_masks, axis=1)
            loss = tf.reduce_mean(tf.square(target_q_values - q_values_for_actions))
        
        # Calcular y aplicar el gradiente
        gradients = tape.gradient(loss, self.primary_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.primary_network.trainable_variables))

        # Si hemos llegado a la frecuencia de actualización de la target network
        # se copian los pesos de la network principal en la target
        self.update_counter += 1
        if self.update_counter % self.update_target_freq == 0:
            self.update_target_network()

        # Actualizamos el valor de epsilon
        self.update_epsilon()
    
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

    def act(self, observation, info):
        # Ajustamos las dimensiones de la observación y la info
        observation = np.expand_dims(observation, axis=0)
        observation = np.expand_dims(observation, axis=-1)
        info = np.expand_dims(info, axis=0)
        
        # Predecimos los Q-values y cogemos la acción con el mayor
        q_values = self.primary_network([observation, info])
        best_action = np.argmax(q_values[0].numpy())
        return best_action