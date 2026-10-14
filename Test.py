import numpy as np
from gym_lunar_rover.algorithms.DDDQL import InferenceDDDQNAgent
from gym_lunar_rover.algorithms.MAPPO import InferenceMAPPOAgent
from gym_lunar_rover.envs.Test_env import TestEnv
from gym_lunar_rover.envs.Utils import *

# Parámetros para la creación del entorno
n_agents = 3
grid_size = 12
vision_range = 3
observation_shape = vision_range*2+1
info_shape = 7

know_pos = False
test_env = TestEnv(n_agents, grid_size, vision_range, know_pos=know_pos, render_mode='human')
action_dim = int(test_env.action_space.nvec[0])

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
        case _:
            print("El algoritmo seleccionado no existe")
            return

    # Se prueba el modelo en distintos entornos fijados con semillas 
    # para igualar las comparaciones entre algoritmos y modelos
    seeds = [1,2,4,5]

    # Límite de pasos por episodio, el mismo que en el entrenamiento, ya que una política
    # determinista puede quedarse atascada repitiendo las mismas acciones indefinidamente
    max_episode_steps = 5000

    for episode, seed in enumerate(seeds):
        test_env.reset(seed)
        dones = [False]*test_env.n_agents
        num_steps = 0
        while not all(dones) and num_steps < max_episode_steps:
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

                if num_steps >= max_episode_steps:
                    break

        if all(dones):
            print(f"Terminado episodio {episode+1} (seed {seed}) con una recompensa total de {test_env.total_reward} en {num_steps} pasos")
        else:
            print(f"Episodio {episode+1} (seed {seed}) sin terminar tras {num_steps} pasos con una recompensa total de {test_env.total_reward}")

def main():
    # Número de steps del modelo que queremos testear
    model_steps = 3750000
    # Algoritmo que queremos testear (DDDQL o MAPPO)
    algorithm = 'DDDQL'
    # algorithm = 'MAPPO'

    test(algorithm, model_steps)

if __name__ == "__main__":
    main()