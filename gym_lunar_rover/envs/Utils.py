import os
import csv
import numpy as np
from gym_lunar_rover.envs.Lunar_Rover_env import RoverObsObjects, LunarObjects, RoversObjects

# Función para generar un nombre de archivo único
def generate_filename(algorithm, base_name, steps, extension):
    return f"saved_trains/{algorithm}/{base_name}_steps_{steps}.{extension}"

# Función para comprobar si existe un archivo
def check_file_exists(filename):
    return os.path.exists(filename)

# Función para escribir en un csv la evolución de las métricas del entrenamiento
def csv_save_train(algorithm, initial_steps, count_steps, total_reward, average_reward, average_loss):
    file_path = 'training_metrics.csv'
    fieldnames = ['algorithm', 'initial_steps', 'count_steps', 'total_reward', 'average_reward', 'average_loss']
    
    if not os.path.isfile(file_path):
        # Inicializar archivo CSV si no existe para guardar métricas
        with open('training_metrics.csv', mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

    # Si ya existe el archivo solo se escribe una nueva fila para no sobrescribir nada
    with open(file_path, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow({'algorithm': algorithm, 'initial_steps': initial_steps, 'count_steps': count_steps, 'total_reward': total_reward, 'average_reward': average_reward,'average_loss': average_loss})

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

# Normalizamos el mapa en el rango 0 a 1 para la entrada del 
# Critic en el algoritmo MAPPO
def normalize_map(map):
    min = LunarObjects.FLOOR.value
    objs = RoversObjects.get_agents_mines()
    max = max(list(objs.keys()) + list(objs.values()))

    map = (map - min) / (max - min)

    return map

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
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.wait = 0
        
        if loss < self.best_loss:
            self.best_loss = loss
            self.wait = 0

        elif self.cooldown_counter <= 0 :
            self.wait += 1
            if self.wait >= self.patience:
                self._reduce_lr()
                self.wait = 0
                self.cooldown_counter = self.cooldown
        
        return self.lr
    
    def _reduce_lr(self):
        new_lr = max(self.lr * self.factor, self.min_lr)
        self.optimizer.learning_rate.assign(new_lr)
        self.lr = new_lr