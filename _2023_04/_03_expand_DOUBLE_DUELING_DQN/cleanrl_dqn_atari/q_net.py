from torch import nn
import random
import numpy as np
import torch
import os
import sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

MODEL_DIR = os.path.join(PROJECT_HOME, "cleanrl_dqn_atari", "models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )
        self.advantage = nn.Linear(512, env.single_action_space.n)
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.0   # 입력 이미지 데이터 픽셀 값을 (0~1) 범위로 정규화
        x = self.network(x)
        advantage = self.advantage(x)
        value = self.value(x)
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values

    def get_action(self, obs, epsilon):
        a = random.random()
        if a < epsilon:
            action = np.array([self.env.single_action_space.sample() for _ in range(self.env.num_envs)])
        else:
            q_values = self.forward(obs)
            action = torch.argmax(q_values, dim=-1).cpu().numpy()
        return action
