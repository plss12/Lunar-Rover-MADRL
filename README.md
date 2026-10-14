# Lunar-Rover-MADRL

[![CI](https://github.com/plss12/Lunar-Rover-MADRL/actions/workflows/ci.yml/badge.svg)](https://github.com/plss12/Lunar-Rover-MADRL/actions/workflows/ci.yml)

## Introducción

Este repositorio contiene la parte práctica de mi Trabajo de Fin de Máster, donde se desarrolla una solución de planificación de rutas multiagente basada en algoritmos de Deep Reinforcement Learning (DRL). El objetivo de este proyecto es optimizar la navegación de múltiples lunar rovers en un entorno lunar dinámico y complejo, facilitando la recolección y el transporte de minerales para la construcción de la primera Estación Internacional de Investigación Lunar en 2030.

## Contenido del Repositorio

- **Simulador**: Un simulador de entornos lunares, desarrollado con las bibliotecas Gymnasium y Pygame, que reproduce fielmente las condiciones del problema a resolver.
- **Algoritmos**: Implementación completa de dos algoritmos de Aprendizaje por Refuerzo Profundo:
  - **Double Dueling Deep Q-Learning (DDDQL)**
  - **Multi-Agent Proximal Policy Optimization (MAPPO)**
- **Código del Proyecto**: Scripts diseñados para la prueba, el entrenamiento y la evaluación de los algoritmos sobre el simulador.
- **Modelos Entrenados**: Los archivos generados tras la fase de entrenamiento, junto con las métricas recopiladas para ambos algoritmos.
- **Documentación**: El documento final de Trabajo de Fín de Máster y la presentación utilizada para la defensa final.

## Instalación y uso

Requiere Python 3.10–3.13 (las versiones con soporte de TensorFlow) y [uv](https://docs.astral.sh/uv/). La ventana de información de `Prueba.py` usa `tkinter`, que viene con Python pero en algunas distribuciones de Linux se instala aparte (p. ej. `python3-tk`).

```bash
uv sync                  # crea el entorno e instala las dependencias
uv run python Prueba.py  # simulador en modo manual o aleatorio
uv run python Train.py   # entrenamiento (DDDQL o MAPPO, se elige en main())
uv run python Test.py    # evaluación de los modelos de saved_trains/
```

El entrenamiento no muestra la simulación por defecto, porque el renderizado limita la velocidad a 120 pasos por segundo; para verla, cambia `render_mode` a `'human'` al principio de `Train.py`. En la evaluación, cada episodio se corta a los 5000 pasos (como en el entrenamiento) por si la política se queda atascada.

Los pesos del modelo DDDQL de `saved_trains/` se guardaron con Keras 2 (TensorFlow 2.10) y los de MAPPO con Keras 3; ambos se cargan con la versión actual gracias a `load_weights_compat`. Las métricas de entrenamiento se guardan en `saved_trains/training_metrics_*.csv`, que es donde las lee `gym_lunar_rover/stats/Graphs.py`.

## Cambios posteriores al TFM

Tras la entrega se corrigió la implementación de MAPPO, por lo que el modelo MAPPO de `saved_trains/` se entrenó con la versión anterior:

- La pérdida del critic comparaba cada valor con todos los retornos del lote en lugar de con el suyo.
- Las trayectorias que se cortan sin terminar (cada `train_freq` pasos o al llegar al máximo de pasos) se completan con el valor del critic (bootstrap) en vez de suponer que no quedan recompensas.
- Los gradientes se promedian solo entre los agentes que han aportado experiencias.
- Cada lote se recorre durante varias épocas (`epochs`), como en PPO y MAPPO. Con una sola actualización el ratio entre la nueva y la antigua política siempre era 1 y el clipping nunca actuaba.

El algoritmo DDDQL y su modelo no han cambiado.

## Desarrollo

```bash
uv run ruff check .  # lint
uv run pytest        # tests
```

El CI de GitHub Actions ejecuta ambos en cada push y pull request, con Python 3.10 y 3.13.

## Contacto

En caso de tener alguna duda, idea o aportación sobre el proyecto por favor contactar al siguiente correo: [pepoluis712@gmail.com](mailto:pepoluis712@gmail.com)
