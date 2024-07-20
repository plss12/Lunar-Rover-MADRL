import gymnasium as gym
import numpy as np
import pygame
from enum import Enum
import random

# Acciones posibles para el Lunar Rover, centradas en el movimiento
class RoverAction(Enum):
    WAIT = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

# Representación de los objetos en el mapa lunar
class LunarObjects(Enum):
    FLOOR = 0
    SMALL_OBSTACLE = 1
    BIG_OBSTACLE = 2

# Representación de los Rovers y sus localizaciones asignadas
class RoversObjects:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.objects = {}

        self._add_rovers_mines_goals()
    
    # Añadimos los rovers, minas y puntos de recogida dinamicamente según el número de agentes
    def _add_rovers_mines_goals(self):
        max_value = max(item.value for item in LunarObjects)
        for i in range(1, self.num_agents + 1):

            self.objects[f'ROVER_{i}'] = max_value + i * 3 - 2
            self.objects[f'MINE_{i}'] = max_value + i * 3 - 1
            self.objects[f'GOAL_{i}'] = max_value + i * 3

    # Método para obtener el número que representa a los agentes y sus minas y puntos de recogida
    def get_agents_mines_goals(self):
        agents_mines_goals = {}
        for i in range(1, self.num_agents + 1):
            rover_name = f'ROVER_{i}'
            mine_name = f'MINE_{i}'
            goal_name = f'GOAL_{i}'
            
            agents_mines_goals[self.objects[rover_name]] = (self.objects[mine_name], self.objects[goal_name])
        
        return agents_mines_goals

class Rover:
    def __init__(self, env, agent_id, mine_id, goal_id, start_pos, mine_pos, goal_pos, vision_range):
        self.env = env
        self.agent_id = agent_id
        self.mine_id = mine_id
        self.goal_id = goal_id
        self.position = start_pos
        self.mine_pos = mine_pos
        self.goal_pos = goal_pos
        self.mined = False
        self.done = False
        self.reward = 0
        self.vision_range = vision_range

    # Método independiente de cada Rover en el que puede
    # realizar movimientos por su cuenta
    def step(self, action):
        if self.done:
            obs = self.get_observation()
            info = {}
            return obs, self.reward, self.done, info

        x, y = self.position
        if action == 1:  # Arriba
            new_x, new_y = x - 1, y
        elif action == 2:  # Abajo
            new_x, new_y = x + 1, y
        elif action == 3:  # Izquierda
            new_x, new_y = x, y - 1
        elif action == 4:  # Derecha
            new_x, new_y = x, y + 1
        elif action == 0:  # Descanso
            new_x, new_y = x, y
            self.reward -= 0.5  # Penalización por descanso
            obs = self.get_observation()
            info = {}
            return obs, self.reward, self.done, info

        new_pos = self.env.grid[new_x, new_y]

        # Comprobamos si la nueva posición está o no vacia
        if new_pos !=0:
            # Recompensa negativa por obstáculo pequeño
            if new_pos == 1:
                self.reward -= 3
            # Recompensa negativa por obstáculo grande
            elif new_pos == 2:
                self.reward -= 8
            # Recompensa negativa por chocar con otro agente
            elif new_pos in self.env.rovers_mines_goals.keys():
                self.reward -=10
            # Recompensa positiva si ha llegado por primera vez a la mina
            elif new_pos == self.mine_id and self.mined == False: 
                self.reward += 100
                self.mined = True
            # Recompensa positiva si ha llegado al punto de recogida tras minar
            elif new_pos == self.goal_id and self.mined == True:
                self.reward += 500
                self.done = True

        # Movemos al agente a la nueva posición y en la posición que estaba 
        # colocamos lo que había antes de que llegase el agente
        self.env.grid[x, y] = self.env.initial_grid[x, y]
        self.env.grid[new_x, new_y] = self.agent_id
        self.position = (new_x, new_y)
        self.reward -= 1  # Penalización por movimiento

        obs = self.get_observation()
        info = {}
        return obs, self.reward, self.done, info

    def get_observation(self):
        x, y = self.position
        min_x = max(x - self.vision_range, 0)
        max_x = min(x + self.vision_range + 1, self.env.grid.shape[0])
        min_y = max(y - self.vision_range, 0)
        max_y = min(y + self.vision_range + 1, self.env.grid.shape[1])

        return self.env.grid[min_x:max_x, min_y:max_y]

    def get_movements(self):
        
        # Se comprueba si el Rover ha terminado
        if self.done:
            return []

        # Si no ha terminado se debe comprobar que acciones puede
        # hacer sin salirse del mapa
        else:
            valid_movements = []

            x, y = self.position
            grid_size_x, grid_size_y = self.env.grid.shape

            movements = {
                RoverAction.WAIT.value: (x, y),
                RoverAction.UP.value: (x - 1, y),
                RoverAction.DOWN.value: (x + 1, y),
                RoverAction.LEFT.value: (x, y - 1),
                RoverAction.RIGHT.value: (x, y + 1)
            }

            for action, (new_x, new_y) in movements.items():
                # Miramos si la nueva posición está dentro de los límites
                if 0 <= new_x < grid_size_x and 0 <= new_y < grid_size_y:
                    valid_movements.append(action)

            return valid_movements


gym.register(
    id='lunar-rover-v0',
    entry_point='Rover:LunarEnv',
)

class LunarEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, n_agents, grid_size, vision_range,  render_mode=None):
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.vision_range = vision_range
        self.render_mode = render_mode

        self.rovers_mines_goals = None
        self.rovers = None
        self.grid = None
        self.initial_grid = None
        self.total_reward = 0

        self.action_space = gym.spaces.MultiDiscrete([len(RoverAction)]*n_agents)
        self.observation_space = gym.spaces.Box(low=0, high=max([o.value for o in LunarObjects]) + n_agents * 3,
                                                shape=(grid_size,grid_size), dtype=np.int32)
        # self.observation_space = gym.spaces.Tuple(
        #                         [gym.spaces.Box(low=0, high=max([o.value for o in LunarObjects]) + n_agents * 3, 
        #                                         shape=(vision_range * 2 + 1, vision_range * 2 + 1), dtype=np.int32) 
                                # for _ in range(n_agents)])

        self.reset()

    def reset(self, seed=None, options=None):

        # Inicializamos el mapa y creamos los agentes y sus objetivos
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.rovers_mines_goals = RoversObjects(self.n_agents).get_agents_mines_goals()
    
        # Colocamos obstáculos, agentes y objetivos en el mapa
        self._place_obstacles()
        self._place_rovers_mines_goals()

        self.initial_grid = np.copy(self.grid)

        self.rovers = []
        for (rover_id, (mine_id, goal_id)) in self.rovers_mines_goals.items():
            start_pos = self._get_empty_position()
            mine_pos = self._get_empty_position()
            goal_pos = self._get_empty_position()
            rover = Rover(self, rover_id, mine_id, goal_id, start_pos, mine_pos, goal_pos, self.vision_range)
            self.rovers.append(rover)

        if self.render_mode=='human':
            self.render()

        obs = self.grid
        info = {}

        return obs, info 
    
    def _place_obstacles(self):
        num_small_obstacles = np.random.randint(1, self.grid_size)
        num_big_obstacles = np.random.randint(1, self.grid_size)
        for _ in range(num_small_obstacles):
            pos = self._get_empty_position()
            self.grid[pos] = LunarObjects.SMALL_OBSTACLE.value
        for _ in range(num_big_obstacles):
            pos = self._get_empty_position()
            self.grid[pos] = LunarObjects.BIG_OBSTACLE.value
    
    def _place_rovers_mines_goals(self):
        for rover_id, (mine_id, goal_id) in self.rovers_mines_goals.items():
            rover_pos = self._get_empty_position()
            self.grid[rover_pos] = rover_id

            mine_pos = self._get_empty_position()
            self.grid[mine_pos] = mine_id

            goal_pos = self._get_empty_position()
            self.grid[goal_pos] = goal_id
    
    def _get_empty_position(self):
        while True:
            pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            if self.grid[pos] == 0:
                return pos
    
    def _get_reward(self):
        return sum([rover.reward for rover in self.rovers])
        
    def _get_done(self):
        return all(rover.done for rover in self.rovers)    
    
    # Método para centralizar movimientos de los Rovers
    def step(self, actions):

        for rover, action in zip(self.rovers, actions):
            if rover.done:
                print(f"Rover {rover.agent_id} ya terminado con recompensa {rover.reward}")
            else:
                if action in rover.get_movements():
                    rover.step(action)
                    print(f"Rover {rover.agent_id} realiza movimiento {action} con recompensa {rover.reward}")
                else:
                    print(f"Rover {rover.agent_id} no realiza el movimiento por ser invalido")
        obs = self.grid
        reward = self._get_reward()
        self.total_reward = reward
        done = self._get_done()
        info = {}

        return obs, reward, done, False, info

    def render(self, mode='human'):
        # Código para renderizar el entorno con pygame
        pass

    def close(self):
        # Clean up resources (optional)
        pass

def prueba_env(n_agents):
    env = gym.make('lunar-rover-v0', render_mode='human', n_agents=n_agents, grid_size=10, vision_range=2)

    # Iniciamos el entorno
    env.reset()

    # Acciones realizadas independientemente por cada Rover
    dones = [False]*n_agents
    while False in dones:
        for i, rover in enumerate(env.unwrapped.rovers):
            if rover.done == False:
                action = random.choice(rover.get_movements())
                obs, reward, done, info = rover.step(action)
                print(f"Rover {i} realiza movimiento {action} con recompensa {reward}")
            else:
                dones[i] = True
                print(f"Rover {i} ya terminado con recompensa {rover.reward}")
    
    # Iniciamos el entorno
    env.reset()

    # Acciones realizadas conjuntamente por el env
    done = False
    while done is False:
        actions = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(actions)
        print(f"Finalizada fase de ejecución con {reward} de recompensa")

prueba_env(5)