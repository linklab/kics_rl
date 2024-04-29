
import collections
import os
import sys
from torch import nn
import torch
import numpy as np
from torch.distributions import Normal
import torch.nn.functional as F

print("TORCH VERSION:", torch.__version__)

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

MODEL_DIR = os.path.join(PROJECT_HOME, "_07_DDPG", "models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, env, n_features=3, n_actions=1):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, n_actions)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.to(DEVICE)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu_v = F.tanh(self.mu(x))
        return mu_v * self.action_scale + self.action_bias

    # def get_action(self, x, training=True):
    #     mu_v = self.forward(x)
    #
    #     if training:
    #
    #     return mu_v


class Critic(nn.Module):
    '''
    Value network V(s_t) = E[G_t | s_t] to use as a baseline in the reinforce
    update. This a Neural Net with 1 hidden layer
    '''

    def __init__(self, n_features=4):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


Transition = collections.namedtuple(
    typename='Transition',
    field_names=['observation', 'action', 'next_observation', 'reward', 'done']
)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def size(self):
        return len(self.buffer)

    def append(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def pop(self):
        return self.buffer.pop()

    def clear(self):
        self.buffer.clear()

    def sample(self, batch_size):
        # Get random index
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        # Sample
        observations, actions, next_observations, rewards, dones = zip(*[self.buffer[idx] for idx in indices])

        # Convert to ndarray for speed up cuda
        observations = np.array(observations)
        next_observations = np.array(next_observations)
        # observations.shape, next_observations.shape: (32, 4), (32, 4)

        actions = np.array(actions)
        actions = np.expand_dims(actions, axis=-1) if actions.ndim == 1 else actions
        rewards = np.array(rewards)
        rewards = np.expand_dims(rewards, axis=-1) if rewards.ndim == 1 else rewards
        dones = np.array(dones, dtype=int)
        # actions.shape, rewards.shape, dones.shape: (32, 1) (32, 1) (32,)

        # Convert to tensor
        observations = torch.tensor(observations, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.int64, device=DEVICE)
        next_observations = torch.tensor(next_observations, dtype=torch.float32, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones, dtype=torch.int, device=DEVICE)

        return observations, actions, next_observations, rewards, dones
