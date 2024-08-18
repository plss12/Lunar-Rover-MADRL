import numpy as np
from gym_lunar_rover.envs.DDDQL import InferenceDDDQNAgent
from gym_lunar_rover.envs.MAPPO import InferenceMAPPOAgent
from gym_lunar_rover.envs.Test_env import TestEnv
from gym_lunar_rover.envs.Utils import generate_filename, normalize_obs, normalize_pos, normalize_visits, check_file_exists

# Parámetros para la creación del entorno
n_agents = 3
grid_size = 12
vision_range = 3
observation_shape = vision_range*2+1
info_shape = 7

know_pos = False
test_env = TestEnv(n_agents, grid_size, vision_range, know_pos=know_pos, render_mode='human')
action_dim = test_env.action_space.nvec[0]

def test(algorithm, steps):
    match algorithm:
        case 'DDDQL':
            model_filename = generate_filename(algorithm,'model_weights', steps, 'weights.h5')
            if not check_file_exists(model_filename):
                print("Faltan ficheros para el modelo que se quiere entrenar")
                return
            agent = InferenceDDDQNAgent(observation_shape, info_shape, action_dim, model_filename)
        case 'MAPPO':
            model_filename = generate_filename(algorithm,'actor_weights', steps, 'weights.h5')
            if not check_file_exists(model_filename):
                print("Faltan ficheros para el modelo que se quiere entrenar")
                return
            agent = InferenceMAPPOAgent(observation_shape, info_shape, action_dim, model_filename)

    # Se prueba el modelo en distintos entornos fijados con semillas 
    # para igualar las comparaciones entre algoritmos y modelos
    seeds = [1,2,3,4,5]
    
    for i, seed in enumerate(seeds):
        test_env.reset(seed)
        dones = [False]*test_env.n_agents
        num_steps = 0
        while not all(dones):
            for i, rover in enumerate(test_env.unwrapped.rovers):
                # Si el Rover ha terminado saltamos al siguiente
                if rover.done:
                    continue
                available_actions = rover.get_movements()
                observation, visits = rover.get_observation()[0:2]
                # Normalizamos la observación en el rango 0-1
                norm_observation = normalize_obs(observation)
                # Normalizamos las visitas en el rango 0-1
                norm_visits = normalize_visits(visits)
                # Normalizamos las posiciones en el rango 0-1
                info = normalize_pos(rover.position + rover.mine_pos + rover.blender_pos, grid_size)
                info = np.append(info, int(rover.mined))

                action = agent.act(norm_observation, norm_visits, info, available_actions)
                step_act = rover.step(action)

                dones[i] = step_act[3]
                num_steps +=1

        print(f"Terminado episodio {i+1} (seed {seed}) con una recompensa total de {test_env.total_reward} en {num_steps} pasos")

def main():
    # Número de steps del modelo que queremos testear
    model_steps = 970000
    # Algoritmo que queremos testear (DDDQL o MAPPO)
    algorithm = 'DDDQL'
    # algorithm = 'MAPPO'

    test(algorithm, model_steps)

if __name__ == "__main__":
    main()