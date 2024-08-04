from gym_lunar_rover.envs.DDDQL import InferenceDDDQNAgent
from gym_lunar_rover.envs.Test_env import TestEnv

# Función para generar un nombre de archivo único
def generate_filename(algorithm, base_name, steps, extension):
    return f"saved_trains/{algorithm}/{base_name}_steps_{steps}.{extension}"

def test_dddql(steps, know_pos):
    # Parámetros para la creación del entorno
    n_agents = 4
    grid_size = 15
    vision_range = 3
    observation_shape = vision_range*2+1
    info_shape = 7

    test_env = TestEnv(n_agents, grid_size, vision_range, know_pos=know_pos, render_mode='human')
    action_dim = test_env.action_space.nvec[0]
    
    model_filename = generate_filename('DDDQL','model_weights', steps, 'h5')
    agent = InferenceDDDQNAgent(observation_shape, info_shape, action_dim, model_filename)

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
                observation = observations[i]
                info = rover.position + rover.mine_pos + rover.blender_pos + (int(rover.mined),)

                action = agent.act(observation, info)
                step_act = rover.step(action)

                observations[i] = step_act[0]
                dones[i] = step_act[2]
        print(f"Terminado episodio {i+1} con una recompensa total de {test_env.total_reward}")

def test_ppo(steps):
    pass

def main():
    # Número de steps del modelo que queremos testear y 
    model_steps = 10
    initial_know_pos = True

    test_dddql(model_steps,initial_know_pos)

if __name__ == "__main__":
    main()
