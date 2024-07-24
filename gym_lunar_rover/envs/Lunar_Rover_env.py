import gymnasium as gym
import numpy as np
import pygame
from enum import Enum

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

        # Hay que aclarar si esta información es sabida por el Rover desde el inicio
        # o será descubierta con la observación del mismo
        self.mine_pos = mine_pos
        self.goal_pos = goal_pos

        self.mined = False
        self.done = False
        self.reward = 0
        self.vision_range = vision_range

    # Método independiente de cada Rover en el que puede
    # realizar movimientos por su cuenta
    def step(self, action):
        # Comprobamos si el rover está terminado para no 
        # continuar con el proceso del movimiento
        if self.done:
            obs = self.get_observation()
            info = {}
            return obs, self.reward, self.done, info

        x, y = self.position
        # Arriba
        if action == 1:
            new_x, new_y = x - 1, y
        # Abajo
        elif action == 2: 
            new_x, new_y = x + 1, y
        # Izquierda
        elif action == 3: 
            new_x, new_y = x, y - 1
        # Derecha
        elif action == 4:
            new_x, new_y = x, y + 1
        # Descanso, y terminamos el proceso al no necesitar
        # comprobaciones extras por no haber movimiento
        elif action == 0:
            new_x, new_y = x, y
            # Penalización menor por gasto de energía en descanso
            self.reward -= 0.5 
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
            # Además no movemos al rover ya que no puede haber
            # dos agentes en una misma posición
            elif new_pos in self.env.rovers_mines_goals.keys():
                self.reward -=10
                obs = self.get_observation()
                info = {}
                return obs, self.reward, self.done, info
            # Recompensa positiva si ha llegado por primera vez a la mina
            elif new_pos == self.mine_id and self.mined == False: 
                self.reward += 100
                self.mined = True
            # Recompensa positiva si ha llegado al punto de recogida tras minar
            elif new_pos == self.goal_id and self.mined == True:
                self.reward += 500
                self.done = True

        # Movemos al agente a la nueva posición y en la posición que estaba 
        # colocamos lo que había en la copia inicial del mapa

        # Si inicialmente en esa posición había un agente
        # en este caso si se coloca un espacio vacio en el mapa
        if (self.env.initial_grid[x, y] in self.env.rovers_mines_goals.keys()):
            self.env.grid[x, y] = LunarObjects.FLOOR.value        

        else:
            self.env.grid[x, y] = self.env.initial_grid[x, y]

        self.env.grid[new_x, new_y] = self.agent_id
        self.position = (new_x, new_y)
        # Penalización por gasto de energia en movimiento
        self.reward -= 1 

        obs = self.get_observation()
        info = {}

        # Falta aclarar si el conocimiento de la posición de la mina y la goal
        # se conocerá inicialmente o se informará con las observaciones

        self.env.render()

        return obs, self.reward, self.done, info

    # Función para obtener el trozo del mapa que es capaz de ver el Rover 
    # según su campo de visión
    def get_observation(self):
        x, y = self.position
        min_x = max(x - self.vision_range, 0)
        max_x = min(x + self.vision_range + 1, self.env.grid.shape[0])
        min_y = max(y - self.vision_range, 0)
        max_y = min(y + self.vision_range + 1, self.env.grid.shape[1])

        return self.env.grid[min_x:max_x, min_y:max_y]

    # Función para obtener los movimientos posibles para un Rover, teniendo
    # en cuenta condiciones como no poder salir del mapa
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
    
class LunarEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 100}

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

        if self.render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((self.grid_size * 50, self.grid_size * 50))
            pygame.display.set_caption("Lunar Rover Simulation")
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):

        # Inicializamos el mapa y creamos los agentes y sus objetivos
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.rovers_mines_goals = RoversObjects(self.n_agents).get_agents_mines_goals()

        # Colocamos obstáculos y creamos y colocamos agentes y objetivos en el mapa
        self._place_obstacles()
        self._place_rovers_mines_goals()

        self.initial_grid = np.copy(self.grid)

        # Definir colores para los rovers, minas y metas
        self.agent_colors = {}
        self.mine_colors = {}
        self.goal_colors = {}
        self._assign_colors()

        obs = self.grid
        info = {}

        if self.render_mode=='human':
            self.render()

        return obs, info 
    
    def _place_obstacles(self):
        num_small_obstacles = np.random.randint(self.grid_size, 2*self.grid_size)
        num_big_obstacles = np.random.randint(1, self.grid_size)
        for _ in range(num_small_obstacles):
            pos = self._get_empty_position()
            self.grid[pos] = LunarObjects.SMALL_OBSTACLE.value
        for _ in range(num_big_obstacles):
            pos = self._get_empty_position()
            self.grid[pos] = LunarObjects.BIG_OBSTACLE.value
    
    def _place_rovers_mines_goals(self):
        self.rovers = []
        for rover_id, (mine_id, goal_id) in self.rovers_mines_goals.items():
            rover_pos = self._get_empty_position()
            self.grid[rover_pos] = rover_id

            # Define las mitades del mapa para separar las minas de las metas
            half_height = self.grid_size // 2

            # Decide aleatoriamente en qué mitad estará la mina y en cuál la meta
            if np.random.choice([True, False]):
                # Mina en la primera mitad, meta en la segunda mitad
                mine_pos = self._get_empty_position(0, half_height, 0, self.grid_size)
                goal_pos = self._get_empty_position(half_height, self.grid_size, 0, self.grid_size)
            else:
                # Mina en la segunda mitad, meta en la primera mitad
                mine_pos = self._get_empty_position(half_height, self.grid_size, 0, self.grid_size)
                goal_pos = self._get_empty_position(0, half_height, 0, self.grid_size)

            self.grid[mine_pos] = mine_id
            self.grid[goal_pos] = goal_id

            rover = Rover(self, rover_id, mine_id, goal_id, rover_pos, mine_pos, goal_pos, self.vision_range)
            self.rovers.append(rover)

    # Obtener una posición vacia, con la posibilidad de elegir unos límites en donde buscar
    def _get_empty_position(self, min_row=0, max_row=None, min_col=0, max_col=None):
        max_row = max_row if max_row is not None else self.grid_size
        max_col = max_col if max_col is not None else self.grid_size

        while True:
            pos = (np.random.randint(min_row, max_row), np.random.randint(min_col, max_col))
            if self.grid[pos] == 0:
                return pos
    
    # Obtiene la suma de las recompensas de todos los rovers
    def _get_reward(self):
        return sum([rover.reward for rover in self.rovers])
    
    # Obtiene un bool según si han terminado o no todos los rovers
    def _get_done(self):
        return all(rover.done for rover in self.rovers)   

    def _assign_colors(self):
        # Generar colores dinámicamente para los rovers, minas y metas
        for agent_id, (mine_id, goal_id) in self.rovers_mines_goals.items():
            color = (agent_id * 50 % 255, agent_id * 80 % 255, agent_id * 110 % 255)
            self.agent_colors[agent_id] = color
            self.mine_colors[mine_id] = color
            self.goal_colors[goal_id] = color
    
    # Método para centralizar movimientos de los Rovers
    def step(self, actions):

        for rover, action in zip(self.rovers, actions):
            if rover.done:
                pass
                # print(f"Rover {rover.agent_id} ya terminado con recompensa {rover.reward}")
            else:
                if action in rover.get_movements():
                    rover.step(action)
                    # print(f"Rover {rover.agent_id} realiza movimiento {action} con recompensa {rover.reward}")
                else:
                    pass
                    # print(f"Rover {rover.agent_id} no realiza el movimiento por ser invalido")
        obs = self.grid
        reward = self._get_reward()
        self.total_reward = reward
        done = self._get_done()
        info = {}

        if self.render_mode == 'human':
            self.render()

        return obs, reward, done, False, info

    def render(self, mode='human'):
        if self.render_mode is None:
            return
        
        # Tamaño del área adicional para mostrar la información
        info_width = 200
        screen_width = self.grid_size * 50 + info_width
        screen_height = self.grid_size * 50

        # Ampliar la ventana de Pygame
        self.screen = pygame.display.set_mode((screen_width, screen_height))

        # Manejo de eventos de Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Definir colores
        floor = (200, 200, 200)
        small_object = (0, 255, 0)
        big_object = (255, 0, 0)

        # Limpiar la pantalla
        self.screen.fill((0, 0, 0))

        font = pygame.font.Font(None, 36)

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                value = self.grid[x, y]
                text = ""

                if value == LunarObjects.FLOOR.value:
                    color = floor
                elif value == LunarObjects.SMALL_OBSTACLE.value:
                    color = small_object
                elif value == LunarObjects.BIG_OBSTACLE.value:
                    color = big_object
                
                elif value in self.agent_colors.keys():
                    color = self.agent_colors[value]
                    text = "A"
                elif value in self.mine_colors.keys():
                    color = self.mine_colors[value]
                    text = "M"
                elif value in self.goal_colors.keys():
                    color = self.goal_colors[value]
                    text = "G"

                pygame.draw.rect(self.screen, color, pygame.Rect(y * 50, x * 50, 50, 50))

                if text:
                    text_surface = font.render(text, True, (0, 0, 0))  # Texto en color negro
                    text_rect = text_surface.get_rect(center=(y * 50 + 25, x * 50 + 25))  # Centrar el texto en el rectángulo
                    self.screen.blit(text_surface, text_rect)

        # Ajustar el tamaño de fuente dinámicamente según el número de agentes
        info_font_size = 24
        info_font = pygame.font.Font(None, info_font_size)
        
        info_x = self.grid_size * 50 + 10
        info_y = 10

        for i, rover in enumerate(self.rovers):
            info_texts = [
                f"Rover {i+1}:",
                f"  Reward: {rover.reward}",
                f"  Position: {rover.position}"
            ]
            for text in info_texts:
                info_surface = info_font.render(text, True, (255, 255, 255))
                self.screen.blit(info_surface, (info_x, info_y))

                info_y += 30  # Espacio entre líneas
        # Actualizar la pantalla
        pygame.display.flip()

        # Control de la tasa de actualización
        self.clock.tick(self.metadata['render_fps'])

    def close(self):
        if self.render_mode == 'human':
            pygame.display.quit()
            pygame.quit()