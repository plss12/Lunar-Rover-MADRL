import numpy as np
from gym_lunar_rover.envs.DDDQL import InferenceDDDQNAgent
from gym_lunar_rover.envs.Test_env import TestEnv
from gym_lunar_rover.envs.Utils import generate_filename, normalize_obs, check_file_exists

def test_dddql(steps, know_pos):
    # Parámetros para la creación del entorno
    n_agents = 3
    grid_size = 12
    vision_range = 3
    observation_shape = vision_range*2+1

    test_env = TestEnv(n_agents, grid_size, vision_range, know_pos=know_pos, render_mode='human', seed = 1)
    action_dim = test_env.action_space.nvec[0]

    model_filename = generate_filename('DDDQL','model_weights', steps, 'weights.h5')
    if not check_file_exists(model_filename):
        print("Faltan ficheros para el modelo que se quiere entrenar")
        return
    agent = InferenceDDDQNAgent(observation_shape, action_dim, model_filename)

    # Hay que probar que en todos los episodios se obtiene la misma recompensa
    # para ver que la inferencia es correcta, luego se podrá prescindir de 
    # varios episodios al ser el entorno de prueba siempre igual
    episodes = 10
    
    for i in range(episodes):
        observations = list(test_env.reset()[0])
        dones = [False]*test_env.n_agents
        while not all(dones):
            for i, rover in enumerate(test_env.unwrapped.rovers):
                # Si el Rover ha terminado saltamos al siguiente
                if rover.done:
                    continue
                available_actions = rover.get_movements()
                observation = observations[i]
                # Normalizamos la observación en el rango 0-1
                observation = normalize_obs(observation)
                # Normalizamos las posiciones en el rango 0-1

                action = agent.act(observation, available_actions)
                step_act = rover.step(action)

                observations[i] = step_act[0]
                dones[i] = step_act[2]
        print(f"Terminado episodio {i+1} con una recompensa total de {test_env.total_reward}")

def test_ppo(steps):
    pass

def main():
    # Número de steps del modelo que queremos testear y si los rovers 
    # incluyen o no la posición de la mina y la mezcladora desde un inicio
    model_steps = 600000
    initial_know_pos = False

    test_dddql(model_steps,initial_know_pos)
    
if __name__ == "__main__":
    main()