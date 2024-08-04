import gymnasium as gym
import os
import csv
from gym_lunar_rover.envs.DDDQL import DoubleDuelingDQNAgent, InferenceDDDQNAgent
from gym_lunar_rover.envs.Test_env import TestEnv

# Función para generar un nombre de archivo único
def generate_filename(algorithm, base_name, steps, extension):
    return f"saved_trains/{algorithm}/{base_name}_steps_{steps}.{extension}"

# Función para comprobar si existe un archivo
def check_file_exists(filename):
    return os.path.exists(filename)

# Función para escribir en un csv la evolución de las métricas del entrenamiento
def csv_save_train(algorithm, initial_steps, total_steps, total_reward, average_reward, average_loss):
    file_path = 'training_metrics.csv'
    fieldnames = ['algorithm', 'initial_steps', 'total_steps', 'total_reward', 'average_reward', 'average_loss']
    
    if not os.path.isfile(file_path):
        # Inicializar archivo CSV si no existe para guardar métricas
        with open('training_metrics.csv', mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

    # Si ya existe el archivo solo se escribe una nueva fila para no sobrescribir nada
    with open(file_path, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow({'algorithm': algorithm, 'initial_steps': initial_steps, 'total_steps': total_steps, 'total_reward': total_reward, 'average_reward': average_reward,'average_loss': average_loss})


def train_dddql(total_steps, model_path=None, buffer_path=None, parameters_path=None):

    # Parámetros para la creación del entorno
    n_agents = 4
    grid_size = 15
    vision_range = 3
    know_pos = False
    observation_shape = vision_range*2+1
    info_shape = 7

    env = gym.make('lunar-rover-v0', render_mode='train', n_agents=n_agents, grid_size=grid_size, vision_range=vision_range, know_pos=know_pos)
    action_dim = env.action_space.nvec[0]

    # Hiperparámetros
    buffer_size = 100000
    batch_size = 64
    gamma = 0.99
    lr = 1e-4
    update_target_freq = 1000

    agent = DoubleDuelingDQNAgent(observation_shape, info_shape, action_dim, buffer_size, batch_size, gamma, lr, update_target_freq, model_path, buffer_path, parameters_path)

    max_iterations = 1000
    global_steps = 0

    total_rewards = []
    averages_rewards = []
    averages_losses = []

    while global_steps < total_steps:
        observations = list(env.reset()[0])
        dones = [False]*n_agents
        iteration = 0
        rewards = []
        losses = []

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
                info = rover.position + rover.mine_pos + rover.blender_pos + (int(rover.mined),)

                action = agent.act(observation, info, available_actions)
                step_act = rover.step(action)

                next_observation, reward, done = step_act[0:3]
                next_info = rover.position + rover.mine_pos + rover.blender_pos + (int(rover.mined),)
                agent.add_experience(observation, info, action, reward, next_observation, next_info, done)
                loss = agent.train()

                observations[i] = next_observation
                dones[i] = done

                rewards.append(reward)
                losses.append(loss)

                global_steps +=1

                if global_steps >= total_steps:
                    break

            if global_steps >= total_steps:
                break

        total_reward = sum(rewards)
        average_reward = round(sum(rewards) / len(rewards),2)
        average_loss = round(sum(losses) / len(losses),4)
        
        total_rewards.append(total_reward)
        averages_rewards.append(average_reward)
        averages_losses.append(average_loss)

        print(f'Episodio acabado con una recompensa total de {total_reward}, una recompensa',
              f'promedio de {average_reward} y una pérdida promedio de {average_loss}')

    model_filename = generate_filename('DDDQL', 'model_weights', agent.update_counter, 'h5')
    buffer_filename = generate_filename('DDDQL', 'replay_buffer', agent.update_counter, 'pkl')
    parameters_filename = generate_filename('DDDQL', 'training_state', agent.update_counter, 'pkl')

    agent.save_train(model_filename, buffer_filename, parameters_filename)

    total_reward = sum(total_rewards)
    average_reward = round(sum(averages_rewards) / len(averages_rewards),2)
    average_loss = round(sum(averages_losses) / len(averages_losses),4)

    print(f'Entrenamiento guardado tras {global_steps} steps con una recompensa',
          f'y loss promedio por episodio de {average_reward} y {average_loss}')

    return total_reward, average_reward, average_loss

def train_ppo():
    pass

def train_by_steps(steps_befor_save, initial_steps, total_train_steps, algorithm):

    # Steps totales que lleva el entrenamiento
    total_steps = 0

    first_train = False
    
    if initial_steps==0:
        first_train = True

    match algorithm:
        case 'DDDQL':
            while total_steps < total_train_steps:
                # Si no hay un modelo previo que entrenar se empieza desde 0
                if first_train:
                    total_reward, average_reward, average_loss = train_dddql(steps_befor_save)
                    first_train = False
                # Si hay un modelo previo se carga y se entrena desde ese punto
                else:
                    model_filename = generate_filename(algorithm, 'model_weights', initial_steps, 'h5')
                    buffer_filename = generate_filename(algorithm, 'replay_buffer', initial_steps, 'pkl')
                    parameters_filename = generate_filename(algorithm, 'training_state', initial_steps, 'pkl')

                    # Se debe comprobar que todos los ficheros necesarios para la carga del modelo existen
                    if not all(check_file_exists(fname) for fname in [model_filename, buffer_filename, parameters_filename]):
                        print("Faltan ficheros para el modelo que se quiere entrenar")
                        return

                    # Si todos sus ficheros existen se realiza el entrenamiento desde el modelo dado
                    total_reward, average_reward, average_loss = train_dddql(steps_befor_save, model_filename, buffer_filename, parameters_filename)

                # Guardamos los datos de como ha ido el entrenamiento para ir viendo su evolución
                csv_save_train(algorithm, initial_steps, total_steps, total_reward, average_reward, average_loss)

                # Sumamos los steps realizados al total y a los iniciales para llevar el recuento
                # de steps totales entrenados en esta llamada y los totales entrenados por el modelo
                total_steps += steps_befor_save
                initial_steps += steps_befor_save
        
        case 'PPO':
            pass

        case _:
            print("El algoritmo seleccionado no existe")

def main():
    # Steps que queremos realizar antes de cada guardado
    steps_before_save = 100000
    # Steps del modelo que queremos continuar entrenando
    # o iniciar un entrenamiento con 0 steps
    initial_steps = 0
    # Steps totales que queremos alcanzar
    total_train_steps = 1000000
    # Algoritmo que queremos usar (DDDQL o PPO)
    algorithm = 'DDDQL'

    train_by_steps(steps_before_save, initial_steps, total_train_steps, algorithm)

if __name__ == "__main__":
    main()