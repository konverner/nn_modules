from random import choices, sample
from collections import deque
import torch
from torch import nn


class Transition:
    def __init__(self, state: torch.tensor, action_idx: torch.tensor, next_state: torch.tensor, reward: torch.tensor):
        self.state = state
        self.action_idx = action_idx
        self.next_state = next_state
        self.reward = reward


class ReplayMemory(object):
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, trans: Transition):
        self.memory.append(trans)

    def sample(self, batch_size: int):
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, n_actions: int):
        super(DQN, self).__init__()
        self.pipeline = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x: torch.tensor):
        q_values = self.pipeline(x)
        return q_values


class Agent:
    def __init__(self, state_dim: int, hidden_dim: int, n_actions: int, memory_size: int = 10000):
        super().__init__()
        self.policy_model = DQN(state_dim, hidden_dim, n_actions)
        self.target_model = DQN(state_dim, hidden_dim, n_actions)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.memory = ReplayMemory(memory_size)

    def load_model(self, path: str):
        self.policy_model.load_state_dict(torch.load(path))

    def save_model(self, path: str):
        torch.save(self.policy_model.state_dict(), path)

    def next_action(self, state: torch.tensor, eps: float = 0.2):
        q_values = self.policy_model(state)
        choice = choices([0, 1], weights=[1 - eps, eps])[0]
        if choice == 0:  # exploit
            return q_values.argmax(1)[0].view(1, 1)
        else:  # explore
            return torch.tensor(choices(range(len(q_values))), dtype=torch.long).unsqueeze(0)
