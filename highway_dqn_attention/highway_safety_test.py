import copy
import random

import gym
import highway_env
import imageio
import numpy as np
import torch
from matplotlib import pyplot as plt
from stable_baselines3 import SAC

from base_function import set_seed, from_Kin_to_Occ

start = 900


def eulvlation(env, model, seeed, seq_num):
    # 环境，模型，随机种子
    random_seed = seeed + start  # 730有一个诡异的碰撞
    env.seed(random_seed)
    set_seed(random_seed)
    done = False
    obs = env.reset()
    i = 0
    frames = []
    speed_all = 0
    # 0左变道，1保持，2右变道，3快，4慢
    while not done and i < seq_num:
        action, _states = model.predict(obs, deterministic=True)
        # if i>21:
        # print(obs[0])
        # print(obs[2])
        # print(action)
        # print(i)
        obs, reward, done, info = env.step(int(action))
        speed_all += info['speed']
        # env.render()
        # im = env.render(mode="rgb_array")
        # frames.append(im)
        i += 1
    # imageio.mimsave("test.gif", frames, fps=1)
    if i < seq_num:
        issuccess = 0
    else:
        issuccess = 1
    return issuccess, i, speed_all / i


def eul(seq, env, model, seq_num):
    success_rate = np.zeros(100)
    mean_speed_all = np.zeros(100)
    for i in range(seq):
        [issuccess, step, mean_speed] = eulvlation(env, model, i, seq_num)
        success_rate[i % 100] = issuccess
        mean_speed_all[i % 100] = mean_speed
        # if np.mean(success_rate)>0.9:
        print(np.mean(success_rate))
        print(np.mean(mean_speed_all))
        print(i)


# 受到攻击
def eul_attack(seq, env, model, seq_num, error, p):
    success_rate = np.zeros(100)
    mean_speed_all = np.zeros(100)
    for i in range(seq):
        [issuccess, step, mean_speed] = eulvlation_attack(env, model, i, seq_num, error, p)
        success_rate[i % 100] = issuccess
        mean_speed_all[i % 100] = mean_speed
        # if np.mean(success_rate)>0.9:
        print(np.mean(success_rate))
        print(np.mean(mean_speed_all))
        print(i)


def eulvlation_attack(env, model, seeed, seq_num, error, p):
    # 环境，模型，随机种子
    random_seed = seeed + start  # 730有一个诡异的碰撞
    set_seed(random_seed)
    done = False
    env.seed(random_seed)
    obs = env.reset()
    i = 0
    speed_all = 0
    # 0左变道，1保持，2右变道，3快，4慢
    while not done and i < seq_num:
        if np.random.rand() < p:
            obs_attack = FDI_attack(copy.deepcopy(obs), error)
        else:
            obs_attack = obs
        obs_fin = from_Kin_to_Occ(obs_attack)
        action, _states = model.predict(obs_fin, deterministic=True)
        obs, reward, done, info = env.step(int(action))
        speed_all += info['speed']
        i += 1
    if i < seq_num:
        issuccess = 0
    else:
        issuccess = 1
    return issuccess, i, speed_all / i


def FDI_attack(obs, error):
    data_size = np.shape(obs)
    for i in range(1, data_size[0]):
        for j in range(1, data_size[1]):
            if obs[i, 0] == 1:
                # obs[i, j] = obs[i, j] + random.gauss(0, error)
                a = (obs[i, j] - obs[0, j]) * random.uniform(-error, error)
                obs[i, j] = obs[i, j] + a
    return obs


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    config = {
        "action": {
            "type": "ContinuousAction",
            'reward_speed_range': [20, 33],
        },
        "high_speed_reward": 0.4,
        "right_lane_reward": 0,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy"],
            "features_range": {
                "vx": [-30, 30],
                "vy": [-30, 30],
                "x": [-100, 100],
                "y": [-100, 100],
            },
            "normalize": 1,
            "absolute": 0,
        },
        "duration": 200,
        "policy_frequency": 5,
    }
    env = gym.make("highway-v0")
    env.configure(config)
    obs = env.reset()
    model = SAC.load("model/highway_dqn_have_v/sac_model_2")
    done = False
    i = 0
    speed_all = 0
    speed = []
    frames = []
    while not done and i < 200:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(reward)
        speed_all += info['speed']
        speed.append(info['speed'])
        im = env.render(mode="rgb_array")
        frames.append(im)
        i += 1
    imageio.mimsave("test.gif", frames, fps=1)
    print(i)
    plt.plot(speed)
    plt.show()
