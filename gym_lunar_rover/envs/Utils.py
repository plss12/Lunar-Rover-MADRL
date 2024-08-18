import os
import csv
import numpy as np
import tensorflow as tf
from enum import Enum

# Acciones posibles para el Lunar Rover, centradas en el movimiento
class RoverAction(Enum):
    WAIT = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

# Recompensas según situación
class RoverRewards(Enum):
    INVALID = -10
    CRASH = -20
    BIG_OBSTACLE = -10
    SMALL_OBSTACLE = -3
    WAIT = -2
    MOVE = -1
    NEAR_LOCATION = +10
    NEW_LOCATION = +50
    MINE = +500
    BLENDER = +1000

# Representación de los objetos en el entorno lunar
class LunarObjects(Enum):
    FLOOR = 0
    SMALL_OBSTACLE = 1
    BIG_OBSTACLE = 2
    BLENDER = 3

# Representación de los Rovers y sus localizaciones asignadas en el entorno lunar
class RoversObjects:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.objects = {}

        self._add_rovers_mines()
    
    # Añadimos los rovers y sus minas dinamicamente según el número de agentes
    def _add_rovers_mines(self):
        max_value = max(item.value for item in LunarObjects)
        for i in range(1, self.num_agents + 1):

            self.objects[f'ROVER_{i}'] = max_value + i * 2 - 1
            self.objects[f'MINE_{i}'] = max_value + i * 2

    # Método para obtener el número que representa a los agentes y sus minas
    def get_agents_mines(self):
        agents_mines = {}
        for i in range(1, self.num_agents + 1):
            rover_name = f'ROVER_{i}'
            mine_name = f'MINE_{i}'
            
            agents_mines[self.objects[rover_name]] = self.objects[mine_name]
        
        return agents_mines

# Estos son los valores que cada Rover verá en sus observaciones
class RoverObsObjects(Enum):
    OUT = -1
    FLOOR = 0
    SMALL_OBSTACLE = 1
    BIG_OBSTACLE = 2
    MINE_NOT_MINED = 3
    MINE_MINED = 4
    BLENDER_NOT_MINED = 5
    BLENDER_MINED = 6
    OTHER_ROVER = 7
    OTHER_MINE = 8

# Función para generar un nombre de archivo único
def generate_filename(algorithm, base_name, steps, extension):
    return f"saved_trains/{algorithm}/{base_name}_steps_{steps}.{extension}"

# Función para comprobar si existe un archivo
def check_file_exists(filename):
    return os.path.exists(filename)

# Función para escribir en un csv la evolución de las métricas del entrenamiento de dddql
def csv_save_train_dddql(algorithm, initial_steps, final_steps, total_reward, average_reward, average_loss, num_episodes, max_steps):
    file_path = 'training_metrics_dddql.csv'
    fieldnames = ['algorithm', 'initial_steps', 'final_steps', 'total_reward', 'average_reward', 'average_loss', 'num_episodes', 'max_steps']
    
    if not os.path.isfile(file_path):
        # Inicializar archivo CSV si no existe para guardar métricas
        with open('training_metrics_dddql.csv', mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

    # Si ya existe el archivo solo se escribe una nueva fila para no sobrescribir nada
    with open(file_path, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow({'algorithm': algorithm, 'initial_steps': initial_steps, 'final_steps': final_steps, 'total_reward': total_reward, 
                         'average_reward': average_reward, 'average_loss': average_loss, 'num_episodes': num_episodes, 'max_steps': max_steps})

# Función para escribir en un csv la evolución de las métricas del entrenamiento de mappo
def csv_save_train_mappo(algorithm, initial_steps, final_steps, total_reward, average_reward, average_actor_loss, average_critic_loss, num_episodes, max_steps):
    file_path = 'training_metrics_mappo.csv'
    fieldnames = ['algorithm', 'initial_steps', 'final_steps', 'total_reward', 'average_reward', 'average_actor_loss', 'average_critic_loss', 'num_episodes', 'max_steps']
    
    if not os.path.isfile(file_path):
        # Inicializar archivo CSV si no existe para guardar métricas
        with open('training_metrics_mappo.csv', mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

    # Si ya existe el archivo solo se escribe una nueva fila para no sobrescribir nada
    with open(file_path, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow({'algorithm': algorithm, 'initial_steps': initial_steps, 'final_steps': final_steps, 'total_reward': total_reward, 
                         'average_reward': average_reward,'average_actor_loss': average_actor_loss, 'average_critic_loss': average_critic_loss, 'num_episodes': num_episodes, 'max_steps': max_steps})

# Función para oscurecer un color
def darken_color(color, factor=0.7):
    return tuple(int(c * factor) for c in color)

def normalize_pos(positions, grid_size):
    positions = np.array(positions)

    # La posición min es -1 cuando no se conoce
    min = -1
    # La posición max es un el tamaño del mapa -1
    max = grid_size-1
    
    positions = (positions - min) / (max - min)
    return positions

# Normalizamos la observación del Rover en el rango 0 a 1 
# para la entrada de la red DDDQN
def normalize_obs(obs):
    min = RoverObsObjects.OUT.value
    max = RoverObsObjects.OTHER_MINE.value

    obs = (obs - min) / (max - min)

    return obs

# Normalizamos el mapa de visitas del Rover
# según el máximo y el mínimo que contenga
def normalize_visits(visits):
    min = np.min(visits)
    max = np.max(visits)

    if max == min:
        return np.zeros_like(visits)

    else :
        return (visits - min) / (max - min)

# Normalizamos el mapa en el rango 0 a 1 para la entrada del 
# Critic en el algoritmo MAPPO
def normalize_map(map, objs):
    min = LunarObjects.FLOOR.value
    max_n = max(list(objs.keys()) + list(objs.values()))

    map = (map - min) / (max_n - min)

    return map

def normalize_reward(reward): 
    return np.sign(reward) * np.log(1 + abs(reward))

# Dados dos maps se combinan ambos, como si fueran dos canales de una imagen,
# para la entrada del critic del MAPPO
# def combine_maps(agent_map, init_map):
#     return tf.concat([agent_map, init_map], axis=-1)

    # return np.stack([agent_map, init_map], axis=-1)

# Estrategia de reducción de lr si no mejora el loss durante el train
class CustomReduceLROnPlateau:
    def __init__(self, optimizer, patience, cooldown, factor, initial_lr, min_lr):
        self.optimizer = optimizer
        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.factor = factor
        self.lr = initial_lr
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.wait = 0
    
    def on_epoch_end(self, loss):
        # Si estamos en cooldown esperamos
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.wait = 0
        
        # Si el nuevo loss es menor que el mejor se guarda
        if loss < self.best_loss:
            self.best_loss = loss
            self.wait = 0
        
        # Si ya no estamos cooldown y el nuevo loss es mayor que
        # el mejor loss comprobamos si hemos superado el wait para
        # reducir el lr según el factor
        elif self.cooldown_counter <= 0 :
            self.wait += 1
            if self.wait >= self.patience:
                self._reduce_lr()
                self.wait = 0
                self.cooldown_counter = self.cooldown
                # Reiniciamos el best loss para que la nueva lr
                # tenga tiempo de demostrar su efectividad
                self.best_loss = float('inf')
        
        return self.lr
    
    def _reduce_lr(self):
        new_lr = max(self.lr * self.factor, self.min_lr)
        self.optimizer.learning_rate.assign(new_lr)
        self.lr = new_lr

def normalize_valid_probs(probs, mask):
    valid_probs_sum = tf.reduce_sum(probs * mask, axis=-1, keepdims=True)
    normalized_probs = (probs * mask) / valid_probs_sum
    return normalized_probs

# Calcula la distancia euclidiana entre dos puntos (x, y)
def calculate_distance(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)