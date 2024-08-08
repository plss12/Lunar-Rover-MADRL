import gymnasium as gym
import numpy as np
from gym_lunar_rover.envs.DDDQL import DoubleDuelingDQNAgent
from gym_lunar_rover.envs.MAPPO import MAPPOAgent
from gym_lunar_rover.envs.Utils import generate_filename, check_file_exists, csv_save_train_dddql, csv_save_train_mappo, normalize_pos, normalize_obs, normalize_map

# Parámetros comunes para la creación del entorno
n_agents = 4
grid_size = 12
vision_range = 3
know_pos = False
observation_shape = vision_range*2+1
info_shape = 7

env = gym.make('lunar-rover-v0', render_mode='train', n_agents=n_agents, grid_size=grid_size, vision_range=vision_range, know_pos=know_pos)
action_dim = env.action_space.nvec[0]

def train_dddql(total_steps, initial_steps, model_path=None, buffer_path=None, parameters_path=None):

    # Hiperparámetros
    buffer_size = 100000
    batch_size = 64

    gamma = 0.95

    max_lr = 1e-3
    min_lr = 1e-5
    lr_decay_factor = 0.8
    patiente = 25
    cooldown = 10

    max_epsilon = 1
    min_epsilon = 0.4
    epsilon_decay = 1e-5

    dropout_rate = 0.3
    l1_rate = 0
    l2_rate = 0.05

    update_target_freq = 500
    warm_up_steps = 100
    clip_rewards = False

    agent = DoubleDuelingDQNAgent(observation_shape, info_shape, action_dim, buffer_size, batch_size, warm_up_steps, clip_rewards,
                                max_epsilon, min_epsilon, epsilon_decay, gamma, max_lr, min_lr, lr_decay_factor, patiente, cooldown, 
                                dropout_rate, l1_rate, l2_rate, update_target_freq, 
                                model_path, buffer_path, parameters_path)

    max_iterations = 5000
    count_steps = 0

    total_rewards = []
    total_losses = []

    while count_steps < total_steps:
        observations = list(env.reset()[0])
        dones = [False]*n_agents
        iteration = 0
        episode_rewards = []
        episode_losses = []

        # Comprobamos que haya Rovers sin terminar y se limita el número de iteraciones
        # para no sobreentrenar situaciones inusuales
        while not all(dones) and iteration < max_iterations:
            iteration +=1
            for i, rover in enumerate(env.unwrapped.rovers):
                # Si el Rover ha terminado saltamos al siguiente
                if rover.done:
                    continue
                available_actions = rover.get_movements()
                observation = observations[i]
                # Normalizamos la observación en el rango 0-1
                norm_observation = normalize_obs(observation)
                # Normalizamos las posiciones en el rango 0-1
                info = normalize_pos(rover.position + rover.mine_pos + rover.blender_pos, grid_size)
                info = np.append(info, int(rover.mined))

                action = agent.act(norm_observation, info, available_actions)
                step_act = rover.step(action)

                # Una vez realizada la acción obtenemos el nuevo estado para 
                # añadir la experiencia completa al buffer
                next_observation, reward, done = step_act[0:3]
                # Normalizamos la observación en el rango 0-1
                norm_next_observation = normalize_obs(next_observation)
                # Normalizamos las posiciones en el rango 0-1
                next_info = normalize_pos(rover.position + rover.mine_pos + rover.blender_pos, grid_size)
                next_info = np.append(next_info, int(rover.mined))
                next_availables_actions = rover.get_movements()
                agent.add_experience(norm_observation, info, action, reward, norm_next_observation, next_info, done, next_availables_actions)
                
                # Con la nueva experiencia añadida entrenamos y obtenemos el loss
                loss = agent.train()

                observations[i] = next_observation
                dones[i] = done

                episode_rewards.append(reward)
                if loss:
                    episode_losses.append(loss)

                count_steps +=1

                if count_steps >= total_steps:
                    break

            if count_steps >= total_steps:
                break

        episode_total_reward = sum(episode_rewards)
        episode_average_reward = round(np.mean(episode_rewards),2)
        episode_average_loss = round(np.mean(episode_losses), 4) if episode_losses else 0
        
        total_rewards.extend(episode_rewards)
        total_losses.extend(episode_losses)

        print(f'Episodio acabado en la iteración {iteration} con una recompensa total de {episode_total_reward},',
              f'una recompensa promedio de {episode_average_reward} y una pérdida promedio de {episode_average_loss}')

    model_filename = generate_filename('DDDQL', 'model_weights', initial_steps+count_steps, 'weights.h5')
    buffer_filename = generate_filename('DDDQL', 'replay_buffer', initial_steps+count_steps, 'pkl')
    parameters_filename = generate_filename('DDDQL', 'training_state', initial_steps+count_steps, 'pkl')

    agent.save_train(model_filename, buffer_filename, parameters_filename)

    total_reward = sum(total_rewards)
    average_reward = round(total_reward / count_steps, 2)
    average_loss = round(np.mean(total_losses), 4)

    print(f'\nEntrenamiento guardado tras {count_steps} steps con una recompensa',
          f'y loss promedio de {average_reward} y {average_loss}\n')

    return total_reward, average_reward, average_loss

def train_mappo(total_steps, initial_steps, actor_path=None, critic_path=None, parameters_path=None):
    # Hiperparámetros
    gamma = 0.95
    lamda = 0.9
    clip = 0.2

    max_lr = 1e-3
    min_lr = 1e-5
    lr_decay_factor = 0.8
    patiente = 25
    cooldown = 10

    dropout_rate = 0.0
    l1_rate = 0
    l2_rate = 0.05

    clip_rewards = False

    # Inicialización del agente MAPPO
    agent = MAPPOAgent(grid_size, observation_shape, info_shape, action_dim, n_agents, clip_rewards,
                       gamma, lamda, clip, max_lr, min_lr, lr_decay_factor, patiente, cooldown, 
                       dropout_rate, l1_rate, l2_rate, 
                       actor_path, critic_path, parameters_path)
    
    max_iterations = 5000
    train_freq = 100 # 2000
    count_steps = 0

    total_rewards = []
    total_actor_losses = []
    total_critic_losses = []

    while count_steps < total_steps:
        observations = list(env.reset()[0])
        dones = [False]*n_agents
        iteration = 0
        episode_rewards = []
        episode_actor_losses = []
        episode_critic_losses = []

        while not all(dones) and iteration < max_iterations:
            iteration += 1
            for i, rover in enumerate(env.unwrapped.rovers):
                if rover.done:
                    continue

                available_actions = rover.get_movements()
                observation = observations[i]
                # Normalizamos la observación en el rango 0-1
                norm_observation = normalize_obs(observation)
                # Normalizamos las posiciones en el rango 0-1
                info = normalize_pos(rover.position + rover.mine_pos + rover.blender_pos, grid_size)
                info = np.append(info, int(rover.mined))
                # Normalizamos el mapa en el rango 0-1
                norm_state = normalize_map(env.unwrapped.grid, env.unwrapped.rovers_mines_ids)

                action, act_prob, state_value = agent.act(norm_observation, norm_state, info, available_actions)
                step_act = rover.step(action)

                # Una vez realizada la acción obtenemos el nuevo estado para 
                # añadir la experiencia completa al buffer
                next_observation, reward, done = step_act[0:3]

                agent.add_experience(i, norm_observation, info, action, reward, done, available_actions, norm_state, state_value, act_prob)

                observations[i] = next_observation
                dones[i] = done

                episode_rewards.append(reward)

                count_steps +=1
                
                if count_steps % train_freq == 0:
                    # Entrenamos si ya hemos alcanzado el número de steps máximo
                    actor_loss, critic_loss = agent.train()
                    if actor_loss:
                        episode_actor_losses.append(actor_loss)
                    if critic_loss:
                        episode_critic_losses.append(critic_loss)

                if count_steps >= total_steps:
                    break

            if count_steps >= total_steps:
                break

        episode_total_reward = sum(episode_rewards)
        episode_average_reward = round(np.mean(episode_rewards),2)
        episode_average_actor_loss = round(np.mean(episode_actor_losses), 4) if episode_actor_losses else 0
        episode_average_critic_loss = round(np.mean(episode_critic_losses), 4) if episode_critic_losses else 0

        total_rewards.extend(episode_rewards)
        total_actor_losses.extend(episode_actor_losses)
        total_critic_losses.extend(episode_critic_losses)

        print(f'Episodio acabado en la iteración {iteration} con una recompensa total de {episode_total_reward},',
              f'una recompensa promedio de {episode_average_reward} y una pérdidas promedio de {episode_average_actor_loss}',
              f'para el actor y {episode_average_critic_loss} para el critic')

    actor_filename = generate_filename('MAPPO', 'actor_weights', initial_steps+count_steps, 'weights.h5')
    critic_filename = generate_filename('MAPPO', 'critic_weights', initial_steps+count_steps, 'weights.h5')
    parameters_filename = generate_filename('MAPPO', 'training_state', initial_steps+count_steps, 'pkl')

    agent.save_train(actor_filename, critic_filename, parameters_filename)

    total_reward = sum(total_rewards)
    average_reward = round(total_reward / count_steps, 2)
    average_actor_loss = round(np.mean(total_actor_losses), 4)
    average_critic_loss = round(np.mean(total_critic_losses), 4)

    print(f'\nEntrenamiento guardado tras {count_steps} steps con una recompensa',
          f'y loss promedio de {average_reward} y {average_actor_loss} para el actor y {average_critic_loss} para el critic\n')

    return total_reward, average_reward, average_actor_loss, average_critic_loss

def train_by_steps(steps_before_save, initial_steps, total_train_steps, algorithm):

    # Steps totales que lleva el entrenamiento
    count_steps = 0

    first_train = False
    
    if initial_steps==0:
        first_train = True

    match algorithm:
        case 'DDDQL':
            # Mientras llevemos menos steps que el total que queremos realizar
            while count_steps < total_train_steps:
                # Si no hay un modelo previo que entrenar se empieza desde 0
                if first_train:
                    total_reward, average_reward, average_loss = train_dddql(steps_before_save, initial_steps)
                    first_train = False
                # Si hay un modelo previo se carga y se entrena desde ese punto
                else:
                    model_filename = generate_filename(algorithm, 'model_weights', initial_steps, 'weights.h5')
                    buffer_filename = generate_filename(algorithm, 'replay_buffer', initial_steps, 'pkl')
                    parameters_filename = generate_filename(algorithm, 'training_state', initial_steps, 'pkl')

                    # Se debe comprobar que todos los ficheros necesarios para la carga del modelo existen
                    if not all(check_file_exists(fname) for fname in [model_filename, buffer_filename, parameters_filename]):
                        print("Faltan ficheros para el modelo que se quiere entrenar")
                        return

                    # Si todos sus ficheros existen se realiza el entrenamiento desde el modelo dado
                    total_reward, average_reward, average_loss = train_dddql(steps_before_save, initial_steps, model_filename, buffer_filename, parameters_filename)

                # Guardamos los datos de como ha ido el entrenamiento para ir viendo su evolución
                csv_save_train_dddql(algorithm, initial_steps, steps_before_save, total_reward, average_reward, average_loss)

                # Sumamos los steps realizados al count total y a los iniciales para llevar el recuento
                # de steps totales entrenados en esta llamada y los totales entrenados por el modelo
                count_steps += steps_before_save
                initial_steps += steps_before_save
        
        case 'MAPPO':
            # Mientras llevemos menos steps que el total que queremos realizar
            while count_steps < total_train_steps:
                # Si no hay un modelo previo que entrenar se empieza desde 0
                if first_train:
                    total_reward, average_reward, average_actor_loss, average_critic_loss= train_mappo(steps_before_save, initial_steps)
                    first_train = False
                # Si hay un modelo previo se carga y se entrena desde ese punto
                else:
                    actor_filename = generate_filename(algorithm, 'actor_weights', initial_steps, 'weights.h5')
                    critic_filename = generate_filename(algorithm, 'critic_weights', initial_steps, 'weights.h5')
                    parameters_filename = generate_filename(algorithm, 'training_state', initial_steps, 'pkl')

                    # Se debe comprobar que todos los ficheros necesarios para la carga del modelo existen
                    if not all(check_file_exists(fname) for fname in [actor_filename, critic_filename, parameters_filename]):
                        print("Faltan ficheros para el modelo que se quiere entrenar")
                        return

                    # Si todos sus ficheros existen se realiza el entrenamiento desde el modelo dado
                    total_reward, average_reward, average_actor_loss, average_critic_loss = train_mappo(steps_before_save, initial_steps, actor_filename, critic_filename, parameters_filename)

                # Guardamos los datos de como ha ido el entrenamiento para ir viendo su evolución
                csv_save_train_mappo(algorithm, initial_steps, steps_before_save, total_reward, average_reward, average_actor_loss, average_critic_loss)

                # Sumamos los steps realizados al count total y a los iniciales para llevar el recuento
                # de steps totales entrenados en esta llamada y los totales entrenados por el modelo
                count_steps += steps_before_save
                initial_steps += steps_before_save

        case _:
            print("El algoritmo seleccionado no existe")

def main():
    # Steps que queremos realizar antes de cada guardado
    steps_before_save = 50000
    # Steps del modelo que queremos continuar entrenando
    # o iniciar un entrenamiento con 0 steps
    initial_steps = 0
    # Steps totales que queremos alcanzar
    total_train_steps = 1000000
    # Algoritmo que queremos usar (DDDQL o PPO)
    # algorithm = 'DDDQL'
    algorithm = 'MAPPO'

    train_by_steps(steps_before_save, initial_steps, total_train_steps, algorithm)

if __name__ == "__main__":
    main()