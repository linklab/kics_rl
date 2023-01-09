import os
import sys
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.distributions import Categorical

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
    def __init__(self, n_features=4, n_actions=2, device=torch.device("cpu")):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
        self.device = device

    def forward(self, x):
        # x = [1.0, 0.5, 0.8, 0.8]  --> [1.7, 2.3] --> [0.3, 0.7]

        # x = [
        #  [1.0, 0.5, 0.8, 0.8]
        #  [1.0, 0.5, 0.8, 0.8]
        #  [1.0, 0.5, 0.8, 0.8]
        #  ...
        #  [1.0, 0.5, 0.8, 0.8]
        # ]

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1 if x.dim() == 2 else 0)
        return x

    def get_action(self, x, mode="train"):
        action_prob = self.forward(x)
        m = Categorical(probs=action_prob)

        if mode == "train":
            action = m.sample()
        else:
            action = torch.argmax(m.probs)
        return action.cpu().numpy()

    def get_action_with_action_prob(self, x, mode="train"):
        action_prob = self.forward(x)   # [0.3, 0.7]
        m = Categorical(probs=action_prob)

        if mode == "train":
            action = m.sample()
            action_prob_selected = action_prob[action]
        else:
            action = torch.argmax(m.probs, dim=1 if action_prob.dim() == 2 else 0)
            action_prob_selected = None
        return action.cpu().numpy(), action_prob_selected


class ActorCritic(nn.Module):
    def __init__(self, n_features=4, n_actions=2, device=torch.device("cpu")):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_pi = nn.Linear(128, n_actions)
        self.fc_v = nn.Linear(128, 1)
        self.device = device

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def pi(self, x):
        x = self.forward(x)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=-1)
        return prob

    def v(self, x):
        x = self.forward(x)
        v = self.fc_v(x)
        return v

    def get_action(self, x, mode="train"):
        action_prob = self.pi(x)
        m = Categorical(probs=action_prob)
        if mode == "train":
            action = m.sample()
        else:
            action = torch.argmax(m.probs, dim=-1)
        return action.cpu().numpy()
