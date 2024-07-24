from gymnasium import register

# Registrar el entorno
register(
    id='lunar-rover-v0',
    entry_point='gym_lunar_rover.envs.Lunar_Rover_env:LunarEnv',)
    