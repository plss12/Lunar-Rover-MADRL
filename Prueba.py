import random
import gymnasium as gym
import pygame
from gym_lunar_rover.envs.Lunar_Rover_env import RoverAction
from gym_lunar_rover.envs.Info_tk import RoverInfoWindow
import threading
import tkinter as tk

def prueba_manual(n_agents, grid_size, vision_range):   

    # Iniciamos un nuevo entorno
    env = gym.make('lunar-rover-v0', render_mode='human', n_agents=n_agents, grid_size=grid_size, vision_range=vision_range)
    env.reset()

    # Creamos una ventana independiente con información de los Rovers
    # Esta ventana se debe ejecutar en un hilo separado para no
    # interferir en el hilo principal de ejecución de la simulación
    root = tk.Tk()
    info_window = RoverInfoWindow(root, env)

    tk_thread = threading.Thread(target=root.mainloop)
    tk_thread.start()

    # Establecer la lista de agentes completados
    dones = [False] * n_agents
    current_agent = 0

    while not all(dones):
        # Capturar eventos de teclado
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                # Asignar acción basada en la tecla presionada
                if event.key == pygame.K_UP:
                    user_action = RoverAction.UP.value
                elif event.key == pygame.K_DOWN:
                    user_action = RoverAction.DOWN.value
                elif event.key == pygame.K_LEFT:
                    user_action = RoverAction.LEFT.value
                elif event.key == pygame.K_RIGHT:
                    user_action = RoverAction.RIGHT.value
                elif event.key == pygame.K_SPACE:
                    user_action = RoverAction.WAIT.value
                else:
                    user_action = None

                if user_action is not None:
                    rover = env.unwrapped.rovers[current_agent]

                    # Si el rover a terminado buscamos uno que no haya terminado
                    if rover.done:
                        while rover.done:
                            current_agent = (current_agent + 1) % n_agents
                            rover = env.unwrapped.rovers[current_agent]

                    # Comprobamos si la acción es válida en el agente
                    if user_action in rover.get_movements():
                        obs, reward, done, info = rover.step(user_action)
                        # print(f"Rover {current_agent} realiza movimiento {user_action} con recompensa {reward}")
                        dones[current_agent] = done
                        current_agent = (current_agent + 1) % n_agents

                        # Actualizamos la información de la ventana de información
                        info_window.update_info()

    print("Simulación completada")
    env.close()

def prueba_conjunta(n_agents, grid_size, vision_range):
    
    # Iniciamos un nuevo entorno
    env = gym.make('lunar-rover-v0', render_mode='human', n_agents=n_agents, grid_size=grid_size, vision_range=vision_range)
    env.reset()
    done = False

    # Acciones realizadas conjuntamente por el env
    while not done:
        actions = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(actions)
        # print(f"Finalizada fase de ejecución con {reward} de recompensa")
    
    print("Simulación completada")
    env.close()

def prueba_individual(n_agents, grid_size, vision_range):
    # Iniciamos un nuevo entorno
    env = gym.make('lunar-rover-v0', render_mode='human', n_agents=n_agents, grid_size=grid_size, vision_range=vision_range)
    env.reset()

    # Acciones realizadas independientemente por cada Rover
    dones = [False]*n_agents
    while not all(dones):
        for i, rover in enumerate(env.unwrapped.rovers):
            if rover.done == False:
                action = random.choice(rover.get_movements())
                obs, reward, done, info = rover.step(action)
                # print(f"Rover {i} realiza movimiento {action} con recompensa {reward}")
            else:
                dones[i] = True
                # print(f"Rover {i} ya terminado con recompensa {rover.reward}")

    print("Simulación completada")
    env.close()

def main():
    n_agents = 5
    grid_size = 15 
    vision_range = 4

    prueba_manual(n_agents, grid_size, vision_range)
    # prueba_individual(n_agents, grid_size, vision_range)
    # prueba_conjunta(n_agents, grid_size, vision_range)

if __name__ == "__main__":
    main()