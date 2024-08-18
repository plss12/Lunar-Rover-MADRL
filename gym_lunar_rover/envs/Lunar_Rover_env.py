import gymnasium as gym
import numpy as np
import pygame
from gym_lunar_rover.envs.Utils import *

class Rover:
    def __init__(self, env, agent_id, mine_id, start_pos, vision_range, mine_pos=(-1,-1), blender_pos=(-1,-1)):
        self.env = env
        self.agent_id = agent_id
        self.mine_id = mine_id
        self.position = start_pos

        self.mine_pos= mine_pos
        self.blender_pos = blender_pos

        self.visits_maps = np.zeros((self.env.unwrapped.grid_size, self.env.unwrapped.grid_size), dtype=int)
        self.visits_maps[self.position] = 1

        self.near_distance = np.inf

        self.mined = False
        self.done = False
        self.last_reward = 0
        self.total_reward = 0
        self.vision_range = vision_range

    # Método independiente de cada Rover en el que puede
    # realizar movimientos por su cuenta
    def step(self, action):
        self.last_reward = 0
        reward = 0
        # Como este rover ya ha realizado el turno 
        # pasamos el turno al siguiente rover no terminado
        self.env.next_rover()
        # Comprobamos si el rover está terminado para no 
        # continuar con el proceso del movimiento
        if self.done:
            obs, visits, obs_rew = self.get_observation()
            info = {}
            return obs, visits, reward, self.done, info
        
        # Si la acción no es valida no se realiza y 
        # se devuelve una recompensa negativa
        elif action not in self.get_movements():
            self.visits_maps[x, y] += 1
            obs, visits, obs_rew = self.get_observation()
            info = {}
            reward += RoverRewards.INVALID.value
            self.last_reward += reward
            self.total_reward += reward
            self.env.total_reward += reward
            return obs, visits, reward, self.done, info

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
            # Penalización por gasto de energía en descanso y no explorar
            reward += RoverRewards.WAIT.value * (self.visits_maps[x, y] + 1)
            self.last_reward += reward
            self.total_reward += reward
            self.env.total_reward += reward
            self.visits_maps[x, y] += 1
            obs, visits, obs_rew = self.get_observation()
            info = {}

            if self.env.unwrapped.render_mode == 'human':
                self.env.render()

            return obs, visits, reward, self.done, info

        new_pos = self.env.grid[new_x, new_y]

        # Comprobamos si la nueva posición está o no vacia para aplicar las recompensas
        if new_pos != 0:
            # Recompensa negativa por obstáculo pequeño
            if new_pos == LunarObjects.SMALL_OBSTACLE.value:
                reward += RoverRewards.SMALL_OBSTACLE.value * (self.visits_maps[new_x, new_y] + 1)

            # Recompensa negativa por obstáculo grande
            elif new_pos == LunarObjects.BIG_OBSTACLE.value:
                reward += RoverRewards.BIG_OBSTACLE.value * (self.visits_maps[new_x, new_y] + 1)

            # Recompensa negativa por chocar con otro agente
            # Además no movemos al rover ya que no puede haber
            # dos agentes en una misma posición
            elif new_pos in self.env.rovers_mines_ids.keys():
                reward += RoverRewards.CRASH.value
                self.last_reward += reward
                self.total_reward += reward
                self.env.total_reward += reward
                self.visits_maps[x, y] += 1
                obs, visits, obs_rew = self.get_observation()
                info = {}
                
                if self.env.unwrapped.render_mode == 'human':
                    self.env.render()

                return obs, visits, reward, self.done, info
            
            # Recompensa positiva si ha llegado por primera vez a la mina
            elif new_pos == self.mine_id and self.mined == False: 
                reward += RoverRewards.MINE.value
                self.mined = True
                self.near_distance = np.inf
            # Recompensa positiva si ha llegado al punto de recogida tras minar
            elif new_pos == LunarObjects.BLENDER.value and self.mined == True:
                reward += RoverRewards.BLENDER.value
                self.done = True

                # Al terminar el Rover se debe borrar del mapa para que los demás
                # Rovers puedan entrar en el Blender sin que esté ocupada la posición
                self.env.grid[x, y] = self.env.initial_grid[x, y]

                self.position = (new_x, new_y)
                self.visits_maps[new_x, new_y] += 1
                obs, visits, obs_rew = self.get_observation()
                reward += obs_rew
                self.last_reward += reward
                self.total_reward += reward
                self.env.total_reward += reward
                info = {}

                if self.env.unwrapped.render_mode == 'human':
                    self.env.render()

                return obs, visits, reward, self.done, info

            # Recompensa negativa por moverse sobre cualquier otra posición 
            # con objeto sin recompensa especial
            else:
                reward += RoverRewards.MOVE.value * (self.visits_maps[new_x, new_y] + 1)

        # Recompensa negativa por el gasto de energía en el movimiento a un espacio vacio
        else:
            reward += RoverRewards.MOVE.value * (self.visits_maps[new_x, new_y] + 1)

        # Movemos al agente a la nueva posición y en la posición que estaba 
        # colocamos lo que había en la copia inicial del mapa
        self.env.grid[x, y] = self.env.initial_grid[x, y]

        self.env.grid[new_x, new_y] = self.agent_id
        self.position = (new_x, new_y)
        self.visits_maps[new_x, new_y] += 1

        # Si la posición de la mina y el blender se descubre con la observación
        # será añadida al Rover tras realizar el movimiento y conllevará recompensa
        obs, visits, obs_rew = self.get_observation()
        reward += obs_rew
        self.last_reward += reward
        self.total_reward += reward
        self.env.total_reward += reward
        info = {}

        if self.env.unwrapped.render_mode == 'human':
            self.env.render()

        return obs, visits, reward, self.done, info

    # Función para obtener el trozo del mapa que es capaz de ver el Rover 
    # según su campo de visión. Para que devuelva siempre una matriz del 
    # mismo tamaño, los espacios fuera del mapa serán representados por -1
    def get_observation(self):
        reward = 0
        x, y = self.position

        # Se obtiene una matriz de -1 según el rango de visión
        obs_size = 2 * self.vision_range + 1
        observation = RoverObsObjects.OUT.value * np.ones((obs_size, obs_size), dtype=int)
        visit_observation = RoverObsObjects.OUT.value * np.ones((obs_size, obs_size), dtype=int)

        min_x = max(x - self.vision_range, 0)
        max_x = min(x + self.vision_range + 1, self.env.grid.shape[0])
        min_y = max(y - self.vision_range, 0)
        max_y = min(y + self.vision_range + 1, self.env.grid.shape[1])

        # Definir índices en la matriz de observación donde copiar los datos del mapa
        start_x = self.vision_range - (x - min_x)
        end_x = start_x + (max_x - min_x)
        start_y = self.vision_range - (y - min_y)
        end_y = start_y + (max_y - min_y)

        observation[start_x:end_x, start_y:end_y] = self.env.grid[min_x:max_x, min_y:max_y]
        visit_observation[start_x:end_x, start_y:end_y] = self.visits_maps[min_x:max_x, min_y:max_y]

        if not self.done:
            # Si la información de la mina y la meta se obtienen con observación se añade aquí
            # y se suma una recompensa por encontrarse cerca de la mina si no hay minado o
            # cerca de la meta si ya ha minado anteriormente
            mine_position = np.argwhere(observation == self.mine_id)
            if len(mine_position) > 0:
                if self.mine_pos == (-1,-1):
                    self.mine_pos = tuple(np.argwhere(self.env.grid == self.mine_id)[0])
                    reward += RoverRewards.NEW_LOCATION.value
                # Si no ha minado es importante estar cerca de la mina
                if not self.mined:
                    distance_to_mine = calculate_distance(self.position, self.mine_pos)
                    # Damos solo recompesa por acercarse no por alejarse
                    if distance_to_mine < self.near_distance:
                        self.near_distance = distance_to_mine
                        reward += round(RoverRewards.NEAR_LOCATION.value / (distance_to_mine))

            blender_position = np.argwhere(observation == LunarObjects.BLENDER.value)
            if len(blender_position) > 0:
                if self.blender_pos == (-1,-1):
                    self.blender_pos = tuple(np.argwhere(self.env.grid == LunarObjects.BLENDER.value)[0])
                    reward += RoverRewards.NEW_LOCATION.value
                # Si ya ha minado es importante estar cerca de la mezcladora
                if self.mined:
                    distance_to_blender = calculate_distance(self.position, self.blender_pos)
                    # Damos solo recompesa por acercarse no por alejarse
                    if distance_to_blender < self.near_distance:
                        self.near_distance = distance_to_blender
                        reward += round(RoverRewards.NEAR_LOCATION.value / (distance_to_blender))

        # Se realiza una normalización a la observación para que las observaciones de todos los Rovers
        # tengan el mismo formato. Para un Rover todos los demás Rovers y minas son iguales sin distinción,
        # solo debe distinguir entre el mismo y su mina asociada

        # Antes de normalizar guardamos la posición del propio Rover, su mina y el blender en la observación
        rover_pos = np.argwhere(observation == self.agent_id)
        mine_position = np.argwhere(observation == self.mine_id)
        blender_position = np.argwhere(observation == LunarObjects.BLENDER.value)

        # Añadimos el objeto que realmente hay en la posición que ocupa el rover para no ocultarlo
        observation[tuple(rover_pos.T)] = self.env.initial_grid[x, y]

        # Normalizar otros rovers visibles
        all_rover_ids = list(self.env.rovers_mines_ids.keys())
        all_rover_ids.remove(self.agent_id)
        for rover_id in all_rover_ids:
            other_rover_pos = np.argwhere(observation == rover_id)
            if other_rover_pos.size > 0:
                observation[tuple(other_rover_pos.T)] = RoverObsObjects.OTHER_ROVER.value

        # Normalizar otras minas visibles
        all_mine_ids = list(self.env.rovers_mines_ids.values())
        all_mine_ids.remove(self.mine_id)
        for mine_id in all_mine_ids:
            other_mines_pos = np.argwhere(observation == mine_id)
            if other_mines_pos.size > 0:
                observation[tuple(other_mines_pos.T)] = RoverObsObjects.OTHER_MINE.value
        
        # Normalizar la mina y el blender asociado al propio Rover, si es visible, según si se ha
        # realizado el objetivo en esa posición o no, para orientar al Rover
        if mine_position.size > 0:
            if self.mined:
                observation[tuple(mine_position.T)] = RoverObsObjects.MINE_MINED.value
            else:
                observation[tuple(mine_position.T)] = RoverObsObjects.MINE_NOT_MINED.value

        if blender_position.size > 0:
            if self.mined:
                observation[tuple(blender_position.T)] = RoverObsObjects.BLENDER_MINED.value
            else:
                observation[tuple(blender_position.T)] = RoverObsObjects.BLENDER_NOT_MINED.value

        return observation, visit_observation, reward

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

    metadata = {"render_modes": ["human","train"], "render_fps": 120}

    def __init__(self, n_agents, grid_size, vision_range, know_pos=False, render_mode=None):
        super(LunarEnv, self).__init__()
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.vision_range = vision_range
        self.know_pos = know_pos
        self.render_mode = render_mode

        self.actual_rover = None
        self.rovers_mines_ids = None
        self.rovers = None
        self.blender_pos = None
        self.grid = None
        self.initial_grid = None
        self.total_reward = 0

        self.action_space = gym.spaces.MultiDiscrete([len(RoverAction)]*n_agents)

        self.observation_space = gym.spaces.Tuple(
                                [gym.spaces.Box(low=RoverObsObjects.OUT.value, high=RoverObsObjects.OTHER_MINE.value,
                                                shape=(vision_range * 2 + 1, vision_range * 2 + 1), dtype=np.int32) 
                                for _ in range(n_agents)])
        
        if self.render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((self.grid_size * 50, self.grid_size * 50))
            pygame.display.set_caption("Lunar Rover Simulation")
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        
        self.total_reward = 0

        # Inicializamos el mapa y creamos los agentes y sus objetivos
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.rovers_mines_ids = RoversObjects(self.n_agents).get_agents_mines()

        # Colocamos obstáculos y creamos y colocamos agentes y objetivos en el mapa
        self._place_obstacles_goal()
        self._place_rovers_mines()

        self.actual_rover = self.rovers[0]

        # Hacemos una copia del estado inicial del mapa sin los agentes
        self.initial_grid = np.copy(self.grid)
        self.initial_grid[np.isin(self.initial_grid, list(self.rovers_mines_ids.keys()))] = 0

        # Definir colores para los rovers, minas y metas
        self.agent_colors = {}
        self.mine_colors = {}
        self._assign_colors()

        # Al ser el inicio del env no queremos que actualice la info
        # de los Rover dada su observación
        obs = self._get_obs(False)
        info = {}

        if self.render_mode=='human':
            self.render()

        return obs, info 
    
    def _place_obstacles_goal(self):
        num_small_obstacles = np.random.randint(2*self.grid_size, 4*self.grid_size)
        num_big_obstacles = np.random.randint(self.grid_size/2, self.grid_size)
        for _ in range(num_small_obstacles):
            pos = self._get_empty_position()
            self.grid[pos] = LunarObjects.SMALL_OBSTACLE.value
        for _ in range(num_big_obstacles):
            pos = self._get_empty_position()
            self.grid[pos] = LunarObjects.BIG_OBSTACLE.value
        pos = self._get_empty_position()
        self.grid[pos] = LunarObjects.BLENDER.value
        self.blender_pos = pos
    
    def _place_rovers_mines(self):
        rovers = []
        for rover_id, mine_id in self.rovers_mines_ids.items():
            rover_pos = self._get_empty_position()
            self.grid[rover_pos] = rover_id

            mine_pos = self._get_empty_position()
            self.grid[mine_pos] = mine_id

            # Se comprueba si se ha configurado o no para que el Rover 
            # sepa inicialmente la posición de su mina y el blender
            if self.know_pos:
                rover = Rover(self, rover_id, mine_id, rover_pos, self.vision_range, mine_pos, self.blender_pos)

            else:
                rover = Rover(self, rover_id, mine_id, rover_pos, self.vision_range)
            
            rovers.append(rover)
        self.rovers = rovers

    # Obtener una posición vacia, con la posibilidad de elegir unos límites en donde buscar
    def _get_empty_position(self, min_row=0, max_row=None, min_col=0, max_col=None):
        max_row = max_row if max_row is not None else self.grid_size
        max_col = max_col if max_col is not None else self.grid_size

        while True:
            pos = (np.random.randint(min_row, max_row), np.random.randint(min_col, max_col))
            if self.grid[pos] == 0:
                return pos
    
    # Obtiene la suma de las recompensas de todos los rovers
    def _get_total_reward(self):
        return sum([rover.reward for rover in self.rovers])
    
    # Obtiene un bool según si han terminado o no todos los rovers
    def _get_done(self):
        return all(rover.done for rover in self.rovers)   
    
    # Obtiene todas las observaciones de los rovers
    def _get_obs(self, update_pos):
        obs = []
        for rover in self.rovers:
            obs.append(rover.get_observation()[0])
            if not update_pos:
                rover.mine_pos = (-1,-1)
                rover.blender_pos = (-1,-1)
        return tuple(obs)
    
    # Obtiene una lista con el estado de si los rovers han terminado
    def _get_dones(self):
        return [rover.done for rover in self.rovers] 
    
    def _assign_colors(self):
        colors = [(0,234,255),      # Cian
                (239, 69, 191),   # Rosa
                (255, 165, 0),    # Naranja
                (145, 30, 180),   # Morado
                (0, 0, 255),      # Azul
                (154, 99, 36),    # Marrón
                (191, 239, 69),   # Pistacho
                (128, 128, 0),    # Oliva
                (245, 245, 220),  # Beige
                (0, 128, 128)]    # Verde azulado

        def generate_random_color(existing_colors):
            while True:
                new_color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                if new_color not in existing_colors:
                    return new_color

        for agent_id, mine_id in self.rovers_mines_ids.items():
            if colors:
                color = colors.pop(0)
            else:
                color = generate_random_color(self.agent_colors.values())
            self.agent_colors[agent_id] = color
            self.mine_colors[mine_id] = color
    
    # Si no hemos acabado, recorremos los rovers en orden para saber
    # cual es su siguiente turno
    def next_rover(self):
        if not self._get_done():
            actual_index = self.rovers.index(self.actual_rover)
            for i in range(1, self.n_agents):
                next_index = (actual_index + i) % self.n_agents
                if not self.rovers[next_index].done:
                    self.actual_rover = self.rovers[next_index]
                    break

    # Método para centralizar movimientos de los Rovers
    def step(self, actions):

        rewards = []

        for rover, action in zip(self.rovers, actions):
            if rover.done:
                pass
                # print(f"Rover {rover.agent_id} ya terminado con recompensa {rover.reward}")
            else:
                rover_step = rover.step(action)
                rewards.append(rover_step[1])
                # print(f"Rover {rover.agent_id} realiza movimiento {action} con recompensa {rover.reward}")

        # Devuelve todas las observaciones, la reward total del step y si han acabado todos
        obs = self._get_obs(True)
        done = self._get_done()
        # Como los formatos del return son fijos, para devolver las listas de recompensas
        # y acabados usaremos el dict info
        info = {"dones":self._get_dones(),
                "rewards":rewards}

        if self.render_mode == 'human':
            self.render()

        return obs, sum(rewards), done, False, info

    def render(self, mode='human'):
        if self.render_mode is None:
            return
        
        # Manejo de eventos de Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        done = self._get_done()
        if not done:
            rover_x, rover_y = self.actual_rover.position
        
        # Definir colores
        floor = (200, 200, 200) # Gris claro
        small_object = (255, 255, 0) # Amarillo
        big_object = (255, 0, 0) # Rojo
        blender_object = (0, 255, 0) # Verde
        border_color = (50, 50, 50) # Negro

        # Limpiar la pantalla
        self.screen.fill((0, 0, 0))

        font = pygame.font.Font(None, 36)

        # Se imprimen los colores y el texto según el objeto que haya en la casilla
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
                elif value == LunarObjects.BLENDER.value:
                    color = blender_object
                    text = "B"
                
                elif value in self.agent_colors.keys():
                    color = self.agent_colors[value]
                    text = "A"
                elif value in self.mine_colors.keys():
                    color = self.mine_colors[value]
                    text = "M"

                if not done:
                    # Verificar si la casilla está dentro del rango de visión del rover
                    if (abs(x - rover_x) <= self.vision_range and abs(y - rover_y) <= self.vision_range) and (x != rover_x or y != rover_y):
                        color = darken_color(color)

                pygame.draw.rect(self.screen, color, pygame.Rect(y * 50, x * 50, 50, 50))
                pygame.draw.rect(self.screen, border_color, pygame.Rect(y * 50, x * 50, 50, 50), 1)

                if text:
                    text_surface = font.render(text, True, (0, 0, 0))
                    text_rect = text_surface.get_rect(center=(y * 50 + 25, x * 50 + 25))
                    self.screen.blit(text_surface, text_rect)

        # Actualizar la pantalla
        pygame.display.flip()

        # Control de la tasa de actualización
        self.clock.tick(self.metadata['render_fps'])

    def close(self):
        if self.render_mode == 'human':
            pygame.display.quit()
            pygame.quit()