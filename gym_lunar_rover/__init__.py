from gym.envs.registration import register

# Registrar el entorno
register(
    id='lunar-rover-v0',
    entry_point='Lunar-Rover:LunarEnv',)
    