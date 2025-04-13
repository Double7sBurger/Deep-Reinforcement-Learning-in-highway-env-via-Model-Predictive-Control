import copy
import time

import gym
import highway_env
import numpy as np
import torch
from matplotlib import pyplot as plt
from stable_baselines3 import DQN

from base_function import set_seed, from_obs_to_figure_batch
from highway_safety_test import eul, eul_attack


def baseline_train(config):
    env = gym.make("highway-fast-v0")
    env.configure(config)
    env.reset()
    model = DQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[1024, 256]),
                learning_rate=1e-3,
                buffer_size=15000,
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=100,
                verbose=2,
                seed=10,
                exploration_final_eps=0.05,
                device="cuda:0",
                tensorboard_log="./tensorboard/highway_com/",
                )
    # D:\Anaconda3\envs\xhs_highway\Scripts\tensorboard --logdir="C:\Users\28063\Desktop\xhs\highway_dqn_attention\tensorboard\highway_com"
    # tensorboard位置+文件位置
    model.learn(total_timesteps=int(8e5), n_eval_episodes=20)
    model.save("model/highway/dqn_model_attention_3e5")  # 32

if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    set_seed(random_seed=11)
    config = {
        "action": {
            "type": 'DiscreteMetaAction',
            'reward_speed_range': [20, 30],
        },
        "high_speed_reward": 0.4,
        "collision_reward": -10,
        "right_lane_reward": 0,
        "observation": {
            "type": "OccupancyGrid",
            "vehicles_count": 10,
            "features": ["x", "y", "vx", "vy"],
            "features_range": {
                "vx": [-30, 30],
                "vy": [-10, 10],
                "x": [-100, 100],
                "y": [-12, 12],
            },
            "grid_size": [[0, 120], [-6, 6]],
            "grid_step": [20, 4],
            "normalize": 0,  # 对"OccupancyGrid"无用
            "absolute": 0,
        },
        "duration": 40,
        "ego_spacing": 3,
        "vehicles_density": 1.5,
        "reward_speed_range": [20, 30],
        "policy_frequency": 1,
    }
    baseline_train(config)
    # 测试的代码
    # model = DQN.load("model/highway/dqn_model_attention_8e5")
    # env = gym.make("highway-v0")
    # env.configure(config)
    # obs = env.reset()
    # eul(seq=100, env=env, model=model, seq_num=40)
