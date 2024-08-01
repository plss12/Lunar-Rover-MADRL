import numpy as np
import random
import gymnasium as gym
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from gym_lunar_rover.envs.Lunar_Rover_env import *
from collections import deque

class DDDQN(Model):
    def __init__(self, out_dim):
        super(DDDQN, self).__init__()

        # Set common feature layer
        self.feature_layer = tf.keras.Sequential([
            Dense(128, activation='relu'),
            Dense(128, activation='relu')
        ])

        # Set advantage layer
        self.advantage_layer = tf.keras.Sequential([
            Dense(128, activation='relu'),
            Dense(out_dim)
        ])

        # Set value layer
        self.value_layer = tf.keras.Sequential([
            Dense(128, activation='relu'),
            Dense(1)
        ])

    def call(self, x):
        feature = self.feature_layer(x)
            
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)

        q = value + advantage - tf.reduce_mean(advantage, axis=-1, keepdims=True)
        
        return q
    
class ExperienceReplayBuffer():
    def __init__(self, buffer_size, batch_size,):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)

    def add_exp(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample_exp(self):
        # Ajusta batch_size si es mayor que el número de experiencias disponibles
        current_size = len(self.buffer)
        batch_size = min(self.batch_size, current_size)

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class DoubleDuelingDQNAgent:
    def __init__(self, action_dim, buffer_size, batch_size, gamma, lr, update_target_freq):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.update_target_freq = update_target_freq

        self.epsilon = 0
        self.min_epsilon = 0.01
        self.epsilon_decay = 1e-3
        
        # Primary and Target Networks
        self.primary_network = DDDQN(action_dim)
        self.target_network = DDDQN(action_dim)
        # self.target_network.set_weights(self.primary_network.get_weights())

        # Optimizer
        self.optimizer = Adam(learning_rate=lr)

        self.primary_network.compile(loss='mse', optimizer=self.optimizer)
        self.target_network.compile(loss='mse', optimizer=self.optimizer)

        
        # Experience Replay Buffer
        self.replay_buffer = ExperienceReplayBuffer(buffer_size, batch_size)
        
        # Update counter
        self.update_counter = 0
    
    def act(self, state, available_actions):
        if np.random.rand() < self.epsilon:
            # Devolver una acción posible aleatoria
            return np.random.choice(available_actions, size=1)
        else:
            # Calculamos los valores de las acciones y devolvemos la mejor
            q_values = self.primary_network(np.expand_dims(state, axis=0))
            best_act = np.argmax(q_values[0].numpy())
            return best_act
    
    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add_exp(state, action, reward, next_state, done)

    def update_target_network(self):
        # Hard update of the target network
        self.target_network.set_weights(self.primary_network.get_weights())

    def update_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.min_epsilon else self.min_epsilon
        return self.epsilon
    
    def save_model(self, file_path):
        self.primary_network.save_weights(file_path)

    def load_model(self, file_path):
        self.primary_network.load_weights(file_path)
        self.target_network.set_weights(self.model.get_weights())

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return 
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample_exp()

        # Convertir arrays a tensores
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.int32)

        # Obtener Q-values del modelo principal para el estado actual y el siguiente estado
        with tf.GradientTape() as tape:
            q_values = self.primary_network(states)
            next_q_values = self.target_network(next_states)

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

        self.update_epsilon()

if __name__ == '__main__':

    n_agents = 5
    grid_size = 15
    vision_range = 4

    env = gym.make('lunar-rover-v0', render_mode='train', n_agents=n_agents, grid_size=grid_size, vision_range=vision_range)
    action_dim = env.action_space.nvec[0]

    # Hyperparameters
    buffer_size = 10000
    batch_size = 64
    gamma = 0.99
    lr = 0.001
    update_target_freq = 500

    # Initialize agent
    agent = DoubleDuelingDQNAgent(action_dim, buffer_size, batch_size, gamma, lr, update_target_freq)

    # Example training loop
    num_episodes = 500

    for episode in range(num_episodes):
        states = env.reset()[0]
        dones = [False]*n_agents

        while not all(dones):
            for i, rover in enumerate(env.unwrapped.rovers):
                available_actions = rover.get_movements()
                state = states[i]
                action = agent.act(state, available_actions)
                # Si la recompensa calculada es invalida se devuelve una recompensa 
                # negativa y se mantiene al rover en la misma posición
                if action not in available_actions:
                    action = 0
                    step_act = rover.step(action)
                    next_state, reward, done = step_act[0:3]
                    reward += -10

                else:
                    step_act = rover.step(action)
                    next_state, reward, done = step_act[0:3]

                dones[i]=done
                agent.add_experience(state, action, reward, next_state, done)
                agent.train()

        print(f'Episode {episode + 1}/{num_episodes} finished with a total reward of {env.unwrapped.total_reward}')