import warnings

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore

import torch
from matplotlib import pyplot as plt
from tensordict import TensorDict
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.libs.dm_control import DMControlEnv, DMControlWrapper

import torchrl
print(torchrl.__version__)

for env_name in GymEnv.available_envs:
    print(env_name)

print("!!!!!!!!!!!!!")

for env_name in DMControlEnv.available_envs:
    print(env_name)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = GymEnv("CartPole-v1")
env.to(DEVICE)
env.set_seed(0)
print("Env observation_spec: \n", env.observation_spec)
print("Env action_spec: \n", env.action_spec)
print("Env reward_spec: \n", env.reward_spec)

