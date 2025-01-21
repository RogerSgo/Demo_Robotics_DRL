# Registro del entorno

from gymnasium.envs.registration import register


register(
    id="DemoC_RL/TareaRL-v0",
    entry_point="DemoC_RL.envs:TareaRL",
)