import pandas as pd
import matplotlib.pyplot as plt

def loss_graph(df, loss, algorithm):

    # Creamos la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(df['final_steps'], df[loss])

    # Desactivamos la notación científica en el eje x
    plt.ticklabel_format(style='plain', axis='x')

    # Asignamos los límites del eje y
    min_loss = df[loss].min() * 1.1
    max_loss = df[loss].max() * 1.1

    if max_loss < 0:
        max_loss = 0

    if min_loss > 0:
        min_loss = 0

    plt.ylim(min_loss, max_loss) 

    # Agregamos títulos y etiquetas
    plt.title(f'Evolución de la pérdida media por step - {algorithm}')
    plt.xlabel('Steps')
    plt.ylabel('Loss')

    # Mostramos la gráfica
    plt.grid(True)
    plt.show()

def avg_reward_graph(df, algorithm):

    # Creamos la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(df['final_steps'], df['average_reward'])

    # Desactivamos la notación científica en el eje x
    plt.ticklabel_format(style='plain', axis='x')

    # Asignamos los límites del eje y
    min_avg = df['average_reward'].min() * 1.1
    max_avg = df['average_reward'].max() * 1.1

    if max_avg < 0:
        max_avg = 0

    # Asignamos los límites del eje y
    plt.ylim(min_avg, max_avg) 

    # Agregamos títulos y etiquetas
    plt.title(f'Evolución de la recompensa media obtenida por step - {algorithm}')
    plt.xlabel('Steps')
    plt.ylabel('Reward')

    # Mostramos la gráfica
    plt.grid(True)
    plt.show()

def finished_episodes_graph(df, algorithm):

    # Creamos la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(df['final_steps'], df['num_episodes'])

    # Desactivamos la notación científica en el eje x
    plt.ticklabel_format(style='plain', axis='x')

    # Asignamos los límites del eje y
    max_avg = df['num_episodes'].max() * 1.1

    # Asignamos los límites del eje y
    plt.ylim(0, max_avg) 

    # Agregamos títulos y etiquetas
    plt.title(f'Evolución en el número de episodios finalizados entre puntos de control - {algorithm}')
    plt.xlabel('Steps')
    plt.ylabel('Finished Episodes')

    # Mostramos la gráfica
    plt.grid(True)
    plt.show()

def max_steps_graph(df, algorithm):

    # Creamos la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(df['final_steps'], df['max_steps'])

    # Desactivamos la notación científica en el eje x
    plt.ticklabel_format(style='plain', axis='x')

    # Asignamos los límites del eje y
    max_avg = df['max_steps'].max() * 1.1

    # Asignamos los límites del eje y
    plt.ylim(0, max_avg) 

    # Agregamos títulos y etiquetas
    plt.title(f'Evolución en el número de steps máximos utilizados \npara finalizar un solo episodio entre puntos de control - {algorithm}')
    plt.xlabel('Steps')
    plt.ylabel('Maximum Steps')

    # Mostramos la gráfica
    plt.grid(True)
    plt.show()

def main():
    algorithm = 'DDDQL'
    # algorithm = 'MAPPO'
    
    match algorithm:
        case 'DDDQL':
            df = pd.read_csv('saved_trains/training_metrics_dddql.csv')
            loss = 'average_loss'
        case 'MAPPO':
            df = pd.read_csv('saved_trains/training_metrics_mappo.csv')
            loss = 'average_actor_loss'

    loss_graph(df, loss, algorithm)
    avg_reward_graph(df, algorithm)
    finished_episodes_graph(df, algorithm)
    max_steps_graph(df, algorithm)

if __name__ == "__main__":
    main()