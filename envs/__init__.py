import gym
from gym.envs.registration import registry, register, make, spec

from envs.humanoid_walker import HumanoidWalker, HumanoidWalkerHardcore

# Box2d envs
# ------------------------------------------
#register(
#    id='HumanoidWalker-v0', 
#    entry_point='envs:humanoid_walker',
#    max_episode_steps=1600,
#    reward_threshold=300,
#)
#gym.make('HumanoidWalker-v0')
