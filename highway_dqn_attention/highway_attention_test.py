import copy

import gym
import highway_env
import numpy as np
import torch
from matplotlib import pyplot as plt
from stable_baselines3 import DQN

from base_function import set_seed, from_obs_to_figure_batch
from highway_safety_test import eul, eul_attack


def baseline_train():
    env = gym.make("highway-fast-v0")
    config = {
        "action": {
            "type": 'DiscreteMetaAction',
            'reward_speed_range': [20, 33],
        },
        "high_speed_reward": 0.4,
        "right_lane_reward": 0,
        "observation": {
            "type": "OccupancyGrid",
            "vehicles_count": 10,
            "features": ["x", "y", "vx", "vy"],
            "features_range": {
                "vx": [-30, 30],
                "vy": [-30, 30],
                "x": [-100, 100],
                "y": [-100, 100],
            },
            "grid_size": [[0, 120], [-6, 6]],
            "grid_step": [20, 4],
            "normalize": 0,  # 对"OccupancyGrid"无用
            "absolute": 0,
        },
        "duration": 100,
        # "ego_spacing": 3,
        # "vehicles_density": 2.5,
        "reward_speed_range": [20, 33],
        "policy_frequency": 1,
    }
    env.configure(config)
    env.reset()
    env.render()
    action=2
    env.step(action)
    env.render()
    env.step(action)
    env.render()


if __name__ == '__main__':
    baseline_train()
