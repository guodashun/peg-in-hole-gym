from gym.envs.registration import register

register(
    id='peg-in-hole-v0',
    entry_point='peg_in_hole_gym.envs.panda_env:PandaEnv',
)