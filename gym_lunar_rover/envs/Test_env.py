from gym_lunar_rover.envs.Lunar_Rover_env import LunarEnv
import numpy as np

# Clase del entorno que utilizaremos durante los tests, se usa una semilla para que siempre
# se genere el mismo entorno inicial y mantener la igualdad de condiciones
class TestEnv(LunarEnv):
    def __init__(self, n_agents, grid_size, vision_range, know_pos=False, render_mode=None, seed=42):
        self.seed_value = seed
        super(TestEnv, self).__init__(n_agents, grid_size, vision_range, know_pos, render_mode)

    def reset(self, seed=None, options=None):
        np.random.seed(self.seed_value)
        return super(TestEnv, self).reset(seed=self.seed_value, options=options)

    def _place_obstacles_goal(self):
        np.random.seed(self.seed_value)  
        super(TestEnv, self)._place_obstacles_goal()

    def _place_rovers_mines(self):
        np.random.seed(self.seed_value)  
        super(TestEnv, self)._place_rovers_mines()