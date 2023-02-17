import warnings

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore

import torch
from matplotlib import pyplot as plt
from tensordict import TensorDict
from torchrl.envs.libs.gym import GymEnv

for env_name in GymEnv.available_envs:
    print(env_name)

env = GymEnv("Cartpole-v1")
env.set_seed(0)
print("Env observation_spec: \n", env.observation_spec)
print("Env action_spec: \n", env.action_spec)
print("Env reward_spec: \n", env.reward_spec)

