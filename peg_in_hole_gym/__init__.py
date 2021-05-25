from gym.envs.registration import register

register(
    id='peg-in-hole-v0',
    entry_point='peg_in_hole_gym.envs.base_env:BaseEnv',
)

register(
    id='peg-in-hole-mp-v0',
    entry_point='peg_in_hole_gym.envs.base_env_mp:BaseEnvMp',
)
