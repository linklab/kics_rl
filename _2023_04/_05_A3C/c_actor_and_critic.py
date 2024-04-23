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

MODEL_DIR = os.path.join(PROJECT_HOME, "_05_A3C", "models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, n_features=3, n_actions=1):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mu = nn.Linear(128, n_actions)
        # self.log_std = nn.Parameter(torch.zeros(n_actions))   # Starting with small std deviation

        # ln_e(x) = 1.0 --> x = e^1.0 = 2.71
        log_std_param = nn.Parameter(torch.full((n_actions,), 1.0))
        self.register_parameter("log_std", log_std_param)
        self.to(DEVICE)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
        # x = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = F.tanh(self.mu(x))
        # mu = F.tanh(self.mu(x)) * 2  # Scale output to -2 ro 2 (action space bounds)
        std = self.log_std.exp().clamp(min=2.0, max=50)  # Clamping for numerical stability

        return mu, std

    def get_action(self, x, exploration=True):
        mu, std = self.forward(x)

        if exploration:
            dist = Normal(loc=mu, scale=std)
            action = dist.sample()
            action = torch.clamp(action, min=-1.0, max=1.0).detach().numpy()
        else:
            action = mu.detach().numpy()
        return action

class Critic(nn.Module):
    '''
       Value network V(s_t) = E[G_t | s_t] to use as a baseline in the reinforce
       update. This a Neural Net with 1 hidden layer
    '''

    def __init__(self, n_features=3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
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


class Buffer:
    def __init__(self):
        self.buffer = collections.deque()

    def size(self):
        return len(self.buffer)

    def append(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def pop(self):
        return self.buffer.pop()

    def clear(self):
        self.buffer.clear()

    def get(self):
        # Sample
        observations, actions, next_observations, rewards, dones = zip(*self.buffer)

        # Convert to ndarray for speed up cuda
        observations = np.array(observations)
        next_observations = np.array(next_observations)
        # observations.shape, next_observations.shape: (32, 4), (32, 4)

        actions = np.array(actions)
        actions = np.expand_dims(actions, axis=-1) if actions.ndim == 1 else actions
        rewards = np.array(rewards)
        rewards = np.expand_dims(rewards, axis=-1) if rewards.ndim == 1 else rewards
        dones = np.array(dones, dtype=bool)
        # actions.shape, rewards.shape, dones.shape: (32, 1) (32, 1) (32,)

        # Convert to tensor
        observations = torch.tensor(observations, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.int64, device=DEVICE)
        next_observations = torch.tensor(next_observations, dtype=torch.float32, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool, device=DEVICE)

        return observations, actions, next_observations, rewards, dones
