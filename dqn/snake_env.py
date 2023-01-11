from random import randint
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


def distance(point1: List[int], point2: List[int]) -> int:
    return abs(point2[0] - point1[0]) + abs(point2[1] - point1[1])


class SnakeEnv:
    """
    An environment for Snake Game
    
    Actions are moves up, down, right, left. States are represented
    as 4-dim vectors [x1, y1, x2, y2] where x1, y1 are coordiantes of
    the snake's head and x2, y2 are coordinates of target (food).
    """
    def __init__(self, size: Tuple[int, int]):
        self.size = size
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.states = []
        for i in range(size[0]):
            for j in range(size[1]):
                for k in range(size[0]):
                    for l in range(size[1]):
                        self.states.append([i, j, k, l])

        self.n_states = len(self.states)
        self.n_action = len(self.actions)

    def get_initial_state(self) -> List[int]:
        state_idx = randint(0, self.n_states - 1)
        return self.states[state_idx]

    # check that we are inside the size of environment
    def is_legal(self, position: List[int]) -> bool:
        # out of bounds north-south
        if position[0] < 0 or position[0] >= self.size[0]:
            return False
        # out of bounds east-west
        elif position[1] < 0 or position[1] >= self.size[1]:
            return False
        return True

    def step(self, state: List[int], action_idx: int) -> Tuple[List[int], float]:

        action = self.actions[action_idx]
        curr_position, curr_target = state[:2], state[2:]

        new_position = [curr_position[0] + action[0], curr_position[1] + action[1]]

        # agent gets target
        if new_position == curr_target:
            reward = 2
            new_target = [randint(0, self.size[0] - 1), randint(0, self.size[1] - 1)]

        # agent is out of boundaries
        elif not self.is_legal(new_position):
            reward = -2
            new_position, new_target = curr_position, curr_target
        # agent is closer to target
        elif distance(new_position, curr_target) < distance(curr_position, curr_target):
            reward = 1
            new_target = curr_target
        # agent is further away from target
        elif np.linalg.norm(new_position - curr_target) > distance(curr_position, curr_target):
            reward = -1
            new_target = curr_target
        else:
            reward = 0
            new_target = curr_target

        new_state = new_position + new_target
        return new_state, reward

    # visualize state
    def render(self, state: List[int]) -> sn.heatmap:
        agent_pos, target_pos = state[:2], state[2:]
        fig, ax = plt.subplots(figsize=self.size)
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        map = np.zeros(self.size)
        map[agent_pos[0], agent_pos[1]] = -1
        map[target_pos[0], target_pos[1]] = 1
        return sn.heatmap(map, fmt=".1f", linewidths=1, linecolor="black", cbar=False, cmap="PiYG")
