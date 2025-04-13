from typing import Any, Dict, List, Optional, Type
import torch.nn.functional as F
import gym
import torch as th
from torch import nn
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule
from torch.nn import init
import numpy as np
import copy

lane = 3
out_channels = 16
figure_step = 20  # 22和2的区别只在于按20米一格还是15米一格
figure_length = round(120 / figure_step) + 1  # 1是自我车辆的,用自己的转化函数不需要这个


def to_tensor(data):
    return th.tensor(data, dtype=th.float)


def svm_rule_hard(obs, action, last_action):
    #########
    # 不能和前车同时变道
    if obs[3, 1, 1] < -0.14 and action == 0:
        return 0
    if obs[3, 1, 1] > 0.14 and action == 2:
        return 0
    # 后方有车辆，没拉开距离之前禁止变道
    if obs[0, 0, 0] != 0 and ((-obs[2, 0, 0] * 30) - 8 - obs[0, 0, 0] * 100 < 0) and action == 0:
        return 0
    if obs[0, 0, 2] != 0 and ((-obs[2, 0, 2] * 30) - 8 - obs[0, 0, 2] * 100 < 0) and action == 2:
        return 0
    ######禁止连续变道
    if last_action == 0 and action == 0:
        return 0
    if last_action == 2 and action == 2:
        return 0
    ######### 无效动作
    if obs[1, 0, 1] < 0.008 and action == 0:
        return 0
    if obs[1, 0, 1] > 0.115 and action == 2:
        return 0
    ###########鼓励加速
    if obs[0, 1, 1] == 0 and obs[0, 2, 1] == 0 and obs[0, 3, 1] == 0:
        if action != 3:
            return 0
    # 左右变道
    if obs[0, 1, 1] != 0 and (obs[2, 1, 1] * 30 * 0.86 + obs[0, 1, 1] * 100) < 0 and (action == 0 or action == 2):
        return 0
    # 0.24比较保险，21也行
    if ((obs[0, 1, 1] != 0 or (obs[0, 2, 1] != 0 and obs[0, 2, 1] < 0.24)) or (
            (obs[0, 2, 1] * 100 + obs[2, 2, 1] * 30 * 2.2 - 5 < 0) and obs[0, 2, 1] != 0)) and action == 3:
        return 0
    if (((obs[0, 1, 1] * 100 + obs[2, 1, 1] * 30 * 2) < 0) or (
            (obs[0, 2, 1] * 100 + obs[2, 2, 1] * 30 * 2.2) < 0)) and action == 1:
        return 0
    ###变完道之后，需要保持安全距离
    if (((obs[0, 1, 2] * 100 + obs[2, 1, 2] * 30 * 2) < 0) or (
            (obs[0, 2, 2] * 100 + obs[2, 2, 2] * 30 * 2.2) < 0)) and action == 2:
        return 0
    if (((obs[0, 1, 0] * 100 + obs[2, 1, 0] * 30 * 2) < 0) or (
            (obs[0, 2, 0] * 100 + obs[2, 2, 0] * 30 * 2.2) < 0)) and action == 0:
        return 0
    # #####################################保守了一点
    # if (obs[0, 1, 1] != 0 or obs[0, 2, 1] < thre) and action == 3:
    #     return 0
    # if (obs[0, 1, 0] != 0 or obs[0, 2, 0] < thre) and action == 0:
    #     return 0
    # if (obs[0, 1, 2] != 0 or obs[0, 2, 2] < thre) and action == 2:
    #     return 0
    # ######################################
    if obs[0, 1, 0] != 0 and action == 0:
        return 0
    if obs[0, 1, 2] != 0 and action == 2:
        return 0
    return 1


class QNetwork(BasePolicy):
    """
    Action-Value (Q-Value) network for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            features_extractor: nn.Module,
            features_dim: int,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_extractor = features_extractor
        ##########卷积层
        self.features_dim = lane * out_channels * (figure_length - 2)
        self.normalize_images = normalize_images
        action_dim = self.action_space.n  # number of actions
        q_net = create_mlp(self.features_dim, action_dim, self.net_arch, self.activation_fn)
        self.q_net = nn.Sequential(*q_net)
        self.last_action = -1
        self.convlayer = nn.Conv2d(in_channels=4, out_channels=out_channels, kernel_size=3, stride=1, padding=(0, 1))
        init.kaiming_normal_(self.convlayer.weight, mode='fan_out')

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        obs = self.convlayer(obs)
        obs = self.extract_features(obs)
        return self.q_net(obs)

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        # satefy_value = highway_rule(observation[0])
        # print(satefy_value)
        q_value = F.softmax(self(observation), dim=1)
        # q_value = F.softmax(th.rand(1,5), dim=1)
        action = q_value.argmax(dim=1)
        # # Greedy action
        # flag = 0
        # while not flag:  # 直到被接受
        #     action = q_value.argmax(dim=1).squeeze()
        #     if action == 4:
        #         break  # 不需要判断了
        #     flag = svm_rule_hard(observation[0], action, self.last_action)
        #     q_value[0, action] = -1
        # self.last_action = action
        # action = action.reshape(-1)
        return action

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data


class DQNPolicy(BasePolicy):
    """
    Policy class with Q-Value Net and target net for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.normalize_images = normalize_images

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        self.q_net, self.q_net_target = None, None
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.

        Put the target network into evaluation mode.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.set_training_mode(False)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def make_q_net(self) -> QNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return QNetwork(**net_args).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self.q_net._predict(obs, deterministic=deterministic)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_args["net_arch"],
                activation_fn=self.net_args["activation_fn"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.q_net.set_training_mode(mode)
        self.training = mode


MlpPolicy = DQNPolicy


class CnnPolicy(DQNPolicy):
    """
    Policy class for DQN when using images as input.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


class MultiInputPolicy(DQNPolicy):
    """
    Policy class for DQN when using dict observations as input.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
