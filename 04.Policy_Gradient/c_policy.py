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

MODEL_DIR = os.path.join(PROJECT_HOME, "04.Policy_Gradient", "models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    def __init__(self, n_features=3, n_actions=1):
        super(Policy, self).__init__()
        self.shared_fc1 = nn.Linear(n_features, 128)
        self.shared_fc2 = nn.Linear(128, 128)
        self.mu = nn.Linear(128, n_actions)
        self.log_std = nn.Linear(128, n_actions)
        self.to(DEVICE)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))
        mu_v = F.tanh(self.mu(x))
        log_std_v = self.log_std(x)
        log_std_v = torch.clamp(log_std_v, min=-3.0, max=2.0)
        std_v = log_std_v.exp()
        # var_v = torch.clamp(var_v, 1e-4, 50)
        # print(
        #     "count_positive_mu: {0:>3}, mean_mu: {1:5.2f}, mean_log_std: {2:5.2f}, mean_std: {3:5.2f}".format(
        #         torch.count_nonzero(mu_v > 0.0).item(),
        #         torch.mean(mu_v).item(),
        #         torch.mean(log_std_v).item(),
        #         torch.mean(std_v).item()
        # ))
        return mu_v, 2.0

    def get_action(self, x, exploration=True):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        mu_v, std_v = self.forward(x)

        if exploration:
            dist = Normal(loc=mu_v, scale=std_v)
            action = dist.sample()
            action = torch.clamp(action, min=-1.0, max=1.0).detach().numpy()
        else:
            action = mu_v.detach().numpy()
        return action

    def get_action_with_action_prob(self, x):
        mu_v, var_v = self.forward(x)
        dist = Normal(loc=mu_v, scale=torch.sqrt(var_v))
        action = dist.sample()
        action = torch.clip(input=action, min=-2.0, max=2.0)

        action_log_prob = dist.log_prob(action)
        action = action.cpu().numpy()

        return action, action_log_prob


class StateValueNet(nn.Module):
    '''
    Value network V(s_t) = E[G_t | s_t] to use as a baseline in the reinforce
    update. This a Neural Net with 1 hidden layer
    '''

    def __init__(self, n_features=3):
        super(StateValueNet, self).__init__()
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
