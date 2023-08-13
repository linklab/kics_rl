import os
import sys
import random

from torch import nn
import collections
import torch
import numpy as np

from a_multi_head_attention import MultiHeadSelfAttention

print("TORCH VERSION:", torch.__version__)


class QNetAttn(nn.Module):
    def __init__(
            self,
            n_features: int,
            n_actions: int,
            hiddens: int = 32,
            n_heads: int = 1,
            device='cpu'
    ):
        super(QNetAttn, self).__init__()
        self.n_actions = n_actions

        n_tasks = n_actions
        n_resource = n_features // n_tasks - 1
        # n_feature: n_tasks * (n_resource + 1)

        d_model = n_resource * n_heads

        # (batch_size, n_tasks, n_resource + 1) -> (batch_size, n_tasks, d_model)
        self.emd = nn.Linear(n_resource+1, d_model)

        # (batch_size, n_tasks, d_model) -> (batch_size, n_tasks, d_model)
        self.attn_layers = MultiHeadSelfAttention(d_model=d_model, num_heads=n_heads)

        # (batch_size, n_tasks, d_model) -> (batch_size, n_tasks, 1)
        self.linear_layers = nn.Sequential(
            nn.Linear(d_model, hiddens),
            nn.LayerNorm(hiddens),
            nn.LeakyReLU(),
            nn.Linear(hiddens, 1)
        )

        self.device = device
        self.to(device)

    def forward(self, x):
        if isinstance(x, list):
            x = np.array(x, dtype=np.float32)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        elif isinstance(x, torch.Tensor):
            x = x.float().to(self.device)
        else:
            raise TypeError(f"unknown type: {type(x)}")

        # (n_feature, ) -> (batch_size, n_feature)
        no_batch = False
        if x.ndim == 1:
            no_batch = True
            x = x.unsqueeze(0)

        # (batch_size, n_feature) -> (batch_size, n_tasks, n_resoure + 1)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.n_actions, -1)

        # (batch_size, n_tasks, n_resoure + 1) -> (batch_size, n_tasks, d_model)
        x = self.emd(x)

        # (batch_size, n_tasks, d_model) -> (batch_size, n_tasks, d_model)
        x = x + self.attn_layers(x)

        # (batch_size, n_tasks, d_model) -> (batch_size, n_tasks, 1)
        x = self.linear_layers(x)

        # (batch_size, n_tasks, 1) -> (batch_size, n_tasks)
        x = x.squeeze(-1)

        if no_batch:
            x = x.squeeze(0)

        return x

    def get_action(self, obs, epsilon, action_mask):
        if isinstance(action_mask, list):
            action_mask = np.array(action_mask, dtype=np.float32)
        if isinstance(action_mask, np.ndarray):
            action_mask = torch.from_numpy(action_mask).bool().to(self.device)
        elif isinstance(action_mask, torch.Tensor):
            action_mask = action_mask.bool().to(self.device)
        else:
            raise TypeError(f"unknown type: {type(action_mask)}")

        # obs.shape: (batch_size or n_envs, num_features) or (num_features,)
        # action_mask.shape: (batch_size or n_envs, num_tasks) or (num_tasks,)
        if random.random() < epsilon:
            if action_mask.ndim == 1:
                available_actions = np.where(action_mask == 0.0)[0]
                actions = np.random.choice(available_actions)
            else:
                actions = []
                for env_id in range(self.n_envs):
                    available_actions = np.where(action_mask[env_id] == 0.0)[0]
                    action = np.random.choice(available_actions)
                    actions.append(action)
                actions = np.array(actions)
        else:
            q_values = self.forward(obs)
            q_values = q_values.masked_fill(action_mask, -float('inf'))
            actions = torch.argmax(q_values, dim=-1)
            actions = actions.cpu().numpy()

        return actions  # argmax: 가장 큰 값에 대응되는 인덱스 반환


Transition = collections.namedtuple(
    typename='Transition',
    field_names=['observation', 'action', 'next_observation', 'reward', 'done', 'action_mask']
)


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
        self.device = device

    def size(self):
        return len(self.buffer)

    def is_full(self):
        return self.size() >= self.capacity

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
        observations, actions, next_observations, rewards, dones, action_masks = zip(
            *[self.buffer[idx] for idx in indices]
        )

        # Convert to ndarray for speed up cuda
        observations = np.array(observations)
        next_observations = np.array(next_observations)
        # observations.shape, next_observations.shape: (32, 4), (32, 4)

        actions = np.array(actions)
        actions = np.expand_dims(actions, axis=-1) if actions.ndim == 1 else actions
        rewards = np.array(rewards)
        rewards = np.expand_dims(rewards, axis=-1) if rewards.ndim == 1 else rewards
        dones = np.array(dones, dtype=bool)
        action_masks = np.array(action_masks)
        # actions.shape, rewards.shape, dones.shape: (32, 1) (32, 1) (32,)

        # Convert to tensor
        observations = torch.tensor(observations, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        next_observations = torch.tensor(next_observations, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        action_masks = torch.tensor(action_masks, dtype=torch.bool, device=self.device)

        return observations, actions, next_observations, rewards, dones, action_masks