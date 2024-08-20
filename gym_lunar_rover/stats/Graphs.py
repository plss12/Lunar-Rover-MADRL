import pandas as pd
import matplotlib.pyplot as plt

def loss_graph(algorithm):

    match algorithm:
        case 'DDDQL':
            df = pd.read_csv('training_metrics_dddql.csv')
        case 'MAPPO':
            df = pd.read_csv('training_metrics_mappo.csv')

    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(df['final_steps'], df['average_loss'], marker='o')

    # Asignamos los límites del eje y
    plt.ylim(0, df['average_loss'].max() * 1.1) 

    # Agregar títulos y etiquetas
    plt.title('Evolución de la pérdida media por step')
    plt.xlabel('Initial Steps')
    plt.ylabel('Average Loss')

    # Mostrar la gráfica
    plt.grid(True)
    plt.show()

def avg_reward_graph():

    df_dddql = pd.read_csv('training_metrics_dddql.csv')
    df_mappo = pd.read_csv('training_metrics_mappo.csv')
    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(df_dddql['final_steps'], df_dddql['average_reward'], marker='o')
    plt.plot(df_mappo['final_steps'], df_mappo['average_reward'], marker='o')

    # Asignamos los límites del eje y
    max_y = max(df_mappo['average_reward'].max() , df_dddql['average_reward'].max())
    min_y = min(df_mappo['average_reward'].min() , df_dddql['average_reward'].min())
    plt.ylim(min_y * 1.1, max_y * 1.1) 

    # Agregar títulos y etiquetas
    plt.title('Evolución de la recompensa media obtenida en cada step')
    plt.xlabel('Initial Steps')
    plt.ylabel('Average Reward')

    # Mostrar la gráfica
    plt.grid(True)
    plt.show()

def finished_episodes_graph():
    df_dddql = pd.read_csv('training_metrics_dddql.csv')
    df_mappo = pd.read_csv('training_metrics_mappo.csv')
    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(df_dddql['final_steps'], df_dddql['num_episodes'], marker='o')
    plt.plot(df_mappo['final_steps'], df_mappo['num_episodes'], marker='o')

    # Asignamos los límites del eje y
    max_y = max(df_mappo['num_episodes'].max() , df_dddql['num_episodes'].max())
    plt.ylim(0, max_y * 1.1) 

    # Agregar títulos y etiquetas
    plt.title('Evolución en el número de episodios acabados entre puntos de control')
    plt.xlabel('Initial Steps')
    plt.ylabel('Finished Episodes')

    # Mostrar la gráfica
    plt.grid(True)
    plt.show()

def max_steps_graph():
    df_dddql = pd.read_csv('training_metrics_dddql.csv')
    df_mappo = pd.read_csv('training_metrics_mappo.csv')
    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(df_dddql['final_steps'], df_dddql['max_steps'], marker='o')
    plt.plot(df_mappo['final_steps'], df_mappo['max_steps'], marker='o')

    # Asignamos los límites del eje y
    max_y = max(df_mappo['max_steps'].max() , df_dddql['max_steps'].max())
    plt.ylim(0, max_y * 1.1) 

    # Agregar títulos y etiquetas
    plt.title('Evolución en el número de steps máximos utilizados entre puntos de control')
    plt.xlabel('Initial Steps')
    plt.ylabel('Maximum Steps')

    # Mostrar la gráfica
    plt.grid(True)
    plt.show()

def main():
    algorithm = 'DDDQL'
    # algorithm = 'MAPPO'
    loss_graph(algorithm)
    avg_reward_graph()
    finished_episodes_graph()
    max_steps_graph()

if __name__ == "__main__":
    main()