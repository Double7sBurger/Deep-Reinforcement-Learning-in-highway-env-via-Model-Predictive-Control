import copy
import time
import gym
import highway_env
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy import io
from stable_baselines3 import DQN
from torch import optim, nn
from torch.optim import lr_scheduler
from base_function import set_seed, from_obs_to_figure_batch, to_tensor
from highway_safety_test import eul

##
import gym
from gym import spaces

from my_model import my_conv_lstm


class my_env(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, config):
        super().__init__()
        self.is_train = False
        self.start_time = time.time()
        self.end_time = 0
        self.rnn_time = []
        self.nor_time = []
        self.data_index = -1
        self.label_index = -1
        self.done = None
        self.obs = None
        self.capacity = 3000
        # 预测的参数
        self.pre_length = 3  # 预测的长度
        self.vision = 2  # 看的是2步之后有没有碰撞
        #
        self.process_step = 0  # 场景步数
        self.tot_step = 0  # 总步数
        self.env = gym.make("highway-fast-v0")
        self.env.configure(config)
        self.env.reset()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.obs_space = np.empty((self.pre_length, 4, 7, 3))
        self.device = 'cuda:0'
        self.pre_model = my_conv_lstm().to(self.device)
        self.loss_function = nn.MSELoss().to(self.device)
        self.data = np.empty((self.capacity, self.pre_length, 4, 7, 3))
        self.label = np.empty((self.capacity, 2))
        self.optimizer = torch.optim.Adam(self.pre_model.parameters(), lr=1e-3, weight_decay=0.01)
        self.loss_target = 0.16  # 目标loss
        self.learn_size = 200
        self.test_length = 200

    def step(self, action):
        if self.process_step < self.pre_length - 1:
            self.obs_space[self.process_step] = self.obs
        else:
            for i in range(self.pre_length - 1):
                self.obs_space[i] = self.obs_space[i + 1]
        observation, reward, done, info = self.env.step(int(action))
        self.process_step = self.process_step + 1
        self.tot_step = self.tot_step + 1
        self.obs = observation
        self.done = done
        # 得到了观测
        if self.process_step < self.pre_length - 1:
            self.obs_space[self.process_step] = self.obs
        else:
            self.obs_space[self.pre_length - 1] = self.obs
        # save
        if self.process_step > self.pre_length - 2:
            self.data_save_obs()
        if self.process_step > self.pre_length - 2 + self.vision:
            self.data_save_label()

        if self.data_index > self.capacity:
            if self.is_train:
                self.is_train = True
                # print("训练完成！")
                # self.learn()
            else:
                self.is_train = True
                for i in range(10000):
                    self.learn()
                    if self.test():
                        break

        # 预测
        occ = self.pre_model(to_tensor(self.obs_space).to(self.device).unsqueeze(0))
        if self.is_train:
            if occ[0, 1] > 0.7 and self.tot_step > self.capacity:
                info["pre_reward"] = -1
            else:
                info["pre_reward"] = 0
        reward = reward + info.get("pre_reward", 0)  # 增加这个属性
        return observation, reward, done, info

    def data_save_obs(self):
        self.data_index = self.data_index + 1
        index = self.data_index % self.capacity
        self.data[index] = copy.deepcopy(self.obs_space)
        self.label[index] = np.array([1, 0])  # 要是没改回来默认碰撞了再说

    def data_save_label(self):
        index = (self.data_index - self.vision + 1) % self.capacity
        if not self.done:  # 没有碰撞
            self.label[index] = np.array([1, 0])  # one-hot编码没有碰撞
        else:
            for i in range(self.vision):  # 0,1,2
                self.label[(index + i) % self.capacity] = np.array([0, 1])  # one-hot编码有碰撞

    def reset(self):
        observation = self.env.reset()
        self.obs = observation
        self.process_step = 0
        return observation

    def render(self, mode="human"):
        return self.env.render(mode="human")

    def close(self):
        return self.env.close()

    def learn(self):
        indices1 = np.where(self.label[0:self.capacity - self.test_length, 1] == 1)[0]  # 碰的
        indices2 = np.where(self.label[0:self.capacity - self.test_length, 0] == 1)[0]  # 安全
        selected_indices1 = np.random.choice(indices1, int(self.learn_size / 2), replace=False)
        selected_indices2 = np.random.choice(indices2, self.learn_size - int(self.learn_size / 2), replace=False)
        index = np.concatenate((selected_indices1, selected_indices2))
        inputVar = to_tensor(self.data[index])
        targetVar = to_tensor(self.label[index])
        inputs = inputVar.to(self.device)  # B,S,5,4,18
        label = targetVar.to(self.device)
        pred = self.pre_model(inputs)  # B,S,C,H,W
        loss = self.loss_function(pred, label)
        loss.backward()
        # 梯度裁剪，超过10减掉
        torch.nn.utils.clip_grad_value_(self.pre_model.parameters(), clip_value=10.0)
        self.optimizer.step()

    def test(self):
        inputVar = to_tensor(self.data[self.capacity - self.test_length:])
        targetVar = to_tensor(self.label[self.capacity - self.test_length:])
        inputs = inputVar.to(self.device)  # B,S,5,4,18
        label = targetVar.to(self.device)
        pred = self.pre_model(inputs)  # B,S,C,H,W
        loss = self.loss_function(pred, label)
        # 找到所有 label[:, 0] < label[:, 1] 的索引
        positive_indices = label[:, 0] < label[:, 1]  # 有危险
        # 在这些索引上找到 pred[:, 1] > pred[:, 0] 的数量
        count = torch.sum((pred[:, 1] > 0.5)[positive_indices]) / sum(positive_indices)
        print(
            f"验证集 loss: {loss.item()}, 准确率:{sum((pred[:, 0] > pred[:, 1]) == (label[:, 0] > label[:, 1])) / self.test_length},"
            f"Recall:{count.item()}")
        if loss.item() < self.loss_target:
            print("训练提前完成")
            return True
        else:
            return False


def baseline_train(config):
    env = my_env(config)
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
    # tensorboard --logdir=highway_dqn_attention\tensorboard\highway_com
    TIMESTEPS = 8000
    for i in range(100):
        model.learn(total_timesteps=int(8e3), reset_num_timesteps=False, tb_log_name="First_Run_1126", n_eval_episodes=20)
        model.save(f"model/highway_dqn_rnn/dqn_model_attention_mpc_{TIMESTEPS*i}")  # 每 8000 timesteps save一次model,这里设置错了，应该是i+1


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    set_seed(random_seed=11)
    config = {
        "action": {
            "type": 'DiscreteMetaAction',
            'reward_speed_range': [20, 30],
            "low_level": "AMPC",
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

    if torch.cuda.is_available():
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name('cuda:0'))
        # print(torch.cuda.get_device_name(1))
    baseline_train(config)


    # 测试的代码q
    # model = DQN.load("model/highway_dqn_rnn/dqn_model_attention8e5")
    # env = gym.make("highway-v0")
    # env.configure(config)
    # obs = env.reset()
    # eul(seq=100, env=env, model=model, seq_num=40)
