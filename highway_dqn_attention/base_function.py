import copy

import numpy as np
import torch
from torch.utils.data import DataLoader


class GetLoader(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    def __len__(self):
        return len(self.data)


def normalized_data(data, max_data, min_data):
    data_base = max_data - min_data
    normalized_data_local = (data - min_data) / data_base
    return normalized_data_local


def recover_data(data, max_data, min_data):
    data_base = max_data - min_data
    recover_data_local = (data * data_base + min_data)
    return recover_data_local


def to_tensor(data):
    return torch.tensor(data, dtype=torch.float)


def to_array(data):
    return data.detach().numpy()


def set_seed(random_seed=11):
    np.set_printoptions(suppress=True)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(random_seed)
    else:
        torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def transfor_loader(train_x, train_y, test_x, test_y, batch_size, train_shuffle=1, test_shuffle=1):
    torch_data_train = GetLoader(train_x, train_y)
    train_loader = DataLoader(torch_data_train, batch_size=batch_size, shuffle=train_shuffle)  # , drop_last=True)

    torch_data_test = GetLoader(test_x, test_y)
    test_loader = DataLoader(torch_data_test, batch_size=batch_size, shuffle=test_shuffle)  # , drop_last=True)
    return train_loader, test_loader


def normalized_five_size(obs_seq, data_max, data_min):
    data_size = np.shape(obs_seq)
    for i in range(data_size[0]):
        for j in range(data_size[1]):
            for k in range(data_size[2]):
                for n in range(data_size[3]):
                    for m in range(1, data_size[4]):
                        obs_seq[i, j, k, n, m] = normalized_data(obs_seq[i, j, k, n, m], data_max[m - 1],
                                                                 data_min[m - 1])
    return obs_seq


def normalized_highway(obs_pre, normal_value=np.array([100, 100, 30, 30])):
    data_size = np.shape(obs_pre)
    # 这个函数用来归一化和相对化
    # 相对化
    for n in range(1, data_size[0]):
        for m in range(1, data_size[1]):
            if obs_pre[n, 0] > 0:  # 有这个车
                obs_pre[n, m] = (obs_pre[n, m] - obs_pre[0, m])
    # 归一化
    for n in range(data_size[0]):
        for m in range(1, data_size[1]):
            if obs_pre[n, 0] > 0:  # 有这个车
                obs_pre[n, m] = obs_pre[n, m] / normal_value[m - 1]
                if obs_pre[n, m] > 1:
                    obs_pre[n, m] = 1
    return obs_pre


def inv_normalized_highway(obs_pre, normal_value=np.array([100, 100, 30, 30])):
    data_size = np.shape(obs_pre)
    # 这个函数用来逆归一化和逆相对化
    # 逆归一化
    for n in range(data_size[0]):
        for m in range(1, data_size[1]):
            obs_pre[n, m] = obs_pre[n, m] * normal_value[m - 1]
    # 逆相对化
    for n in range(1, data_size[0]):
        for m in range(1, data_size[1]):
            obs_pre[n, m] = (obs_pre[n, m] + obs_pre[0, m])
    return obs_pre


def recover_data_two_size(obs_pre, data_max, data_min):
    # 用来给二维数据反归一化
    data_size = np.shape(obs_pre)
    for n in range(data_size[0]):
        for m in range(data_size[1]):
            if m == 0:
                if obs_pre[n, m] > 0.1:
                    obs_pre[n, m] = 1
            else:
                obs_pre[n, m] = recover_data(obs_pre[n, m], data_max[m - 1], data_min[m - 1])
    return obs_pre


def from_obs_to_figure_batch(data_input):
    #  H W C S
    # 这个函数把观测数据转化为在位置上可以一一对应的图数据
    figure_step = 20
    figure_width = 4
    figure_length = round(200 / figure_step) + 1  # 1是自我车辆的
    features = 4
    data_size = np.shape(data_input)  # b 10*5
    figure_batch = np.zeros((data_size[0], figure_width, figure_length, features))  # B H W C
    for k in range(data_size[0]):
        origin_data = data_input[k, :, :]
        figure_data = from_obs_to_figure_data(origin_data)
        figure_batch[k, :, :, :] = figure_data
    return figure_batch  # 返回批数量的结果


def from_obs_to_figure_data(origin_data, figure_width=4, figure_length=11, features=4, figure_step=20, ori_width=10):
    vx_max = 25
    vy_max = 6
    figure_data = np.zeros((figure_width, figure_length, features))
    ego_vehicle = origin_data[0, 1:]  # 这个是第一幕自我车辆的数据
    ego_x = ego_vehicle[0]
    figure_data[int(ego_vehicle[1] // 4), 0, 0] = 0
    figure_data[int(ego_vehicle[1] // 4), 0, 1] = (ego_vehicle[1] % 4) / 4
    figure_data[int(ego_vehicle[1] // 4), 0, 2] = ego_vehicle[2] / vx_max
    figure_data[int(ego_vehicle[1] // 4), 0, 3] = ego_vehicle[3] / vy_max

    for i in range(1, ori_width):  # 第一行是自我车辆，不算
        if origin_data[i, 0] == 0:  # 没有车
            break
        Vehicle_Information = origin_data[i, 1:]  # x y vx vy
        x = Vehicle_Information[0]
        y = Vehicle_Information[1]
        # 与matlab不同，python从0开始索引
        ind_x = int((x - ego_x) // figure_step) + 1
        ind_y = int(y // 4)
        if figure_data[ind_y, ind_x, 0] != 0:
            print("!")
        # figure_data[ind_y, ind_x, 0] = ((x - ego_x) % figure_step) / figure_step
        figure_data[ind_y, ind_x, 0] = (x - ego_x) / 100
        figure_data[ind_y, ind_x, 1] = (y % 4) / 4
        figure_data[ind_y, ind_x, 2] = Vehicle_Information[2] / vx_max
        figure_data[ind_y, ind_x, 3] = Vehicle_Information[3] / vy_max
    return copy.deepcopy(figure_data)


def from_ori_to_figure_seq(data_input, seq_len=5):
    #  H W C S
    figure_step = 20
    figure_width = 4
    figure_length = round((200 + seq_len * 25) / figure_step)
    features = 3
    data_size = np.shape(data_input)
    ego_vehicle = data_input[0, 1:, 0]  # 这个是第一幕自我车辆的数据
    ego_x = ego_vehicle[0]
    figure_seq = np.zeros((figure_width, figure_length, features, seq_len))
    for k in range(data_size[2]):
        origin_data = data_input[:, :, k]
        figure_data = from_ori_to_figure_data(origin_data, ego_x)
        figure_seq[:, :, :, k] = figure_data
    return figure_seq, ego_vehicle


def from_ori_to_figure_data(origin_data, ego_x, figure_width=4, figure_length=16, features=3, figure_step=20,
                            ori_width=10):
    figure_data = np.zeros((figure_width, figure_length, features))
    for i in range(1, ori_width):  # 第一行是自我车辆，不算
        if origin_data[i, 0] == 0:  # 没有车
            break
        Vehicle_Information = origin_data[i, 1:]  # x y vx vy
        x = Vehicle_Information[0]
        y = Vehicle_Information[1]
        # 与matlab不同，python从0开始索引
        ind_x = int((x - ego_x) // figure_step)
        ind_y = int(y // 4)
        figure_data[ind_y, ind_x, 0] = ((x - ego_x) % figure_step) / figure_step
        figure_data[ind_y, ind_x, 1] = (y % 4) / 4
        figure_data[ind_y, ind_x, 2] = 1  # 只在有车的格子设为1
    return copy.deepcopy(figure_data)


def from_figure_data_to_ori(figure_data, ego_vehicle, data_last, frequency=1, figure_step=20, yunzhi=0.4):
    # 利用第一辆车的数组还原原始的数据
    # data_last是车辆数据最后的情况，差分得到速度
    ego_number = 10
    obs_size = np.array([ego_number, 5])  # 每个数据是5个属性，用差分把x和y的速度还原回去
    origin_data = np.zeros((obs_size[0], obs_size[1]))
    index = 1  # 因为第一行是自我车辆，且是当前的自我车辆，不需要测量
    figure_size = np.shape(figure_data)
    for i in range(figure_size[0]):  # 无所谓，后面再根据第二列排序即可
        for j in range(figure_size[1]):
            if figure_data[i, j, 2] > yunzhi:  # 图数据的第三个维度是标志位，说明这个格子有没有车
                origin_data[index, 0] = 1
                x = j * figure_step + ego_vehicle[0] + figure_data[i, j, 0] * figure_step
                y = i * 4 + figure_data[i, j, 1] * 4
                origin_data[index, 1:3] = np.hstack((x, y))
                index += 1
    result = origin_data[np.argsort(origin_data[0:index], axis=0)[:, 1], :]
    seq = np.zeros((1, 5))
    while index < ego_number:  # 如果不够
        result = np.vstack((result, seq))
        index = index + 1
    # f是频率，相当于两个位置一减除以采样时间得到速度
    # 只要预测没漏车，这个结果就是正确的
    result[:, 3] = data_last[:, 3]
    result[:, 4] = (result[:, 2] - data_last[:, 2]) * frequency
    return result


def pre_seq(data_input, model_road):
    pre_model = torch.load(model_road).cpu()
    data_last = data_input[:, :, -1]  # 最新的数据，希望用差分还原出速度
    [figure_seq, ego_vehicle] = from_ori_to_figure_seq(data_input, seq_len=5)  # 转化为图数据
    figure_seq = figure_seq[:, :, :, :, np.newaxis].transpose(4, 3, 2, 0, 1)  # 和matlab的顺序一样转化为跟训练的数据一样
    pre_figure_data = pre_model(to_tensor(figure_seq))  # 跟训练一样的输入
    pre_figure_data = to_array(pre_figure_data.squeeze()).transpose(1, 2, 0)  # from CHW to HWC
    pre_ori_data = from_figure_data_to_ori(pre_figure_data, ego_vehicle, data_last, frequency=1,
                                           yunzhi=0.4)  # 转化为原来的数据
    obs_pre = copy.deepcopy(pre_ori_data)
    return obs_pre


def pre_all_seq(obs_seq, model_road="model/encode-decode-gridding_sample_1.pth"):
    obs_seq = obs_seq.transpose(1, 2, 0)
    seq_len = 30
    obs_pre = np.zeros((30, 10, 5))
    for i in range(seq_len):
        if i < 1:
            obs_pre[i, :, :] = np.zeros((10, 5))
            continue
        if i <= 4:
            obs_pre[i, :, :] = obs_seq[:, :, i - 1]
            continue
        if i > 4:
            data_input = obs_seq[:, :, i - 5:i]
            obs_pre[i, :, :] = pre_seq(data_input=data_input, model_road=model_road)
            obs_pre[i, 0, :] = obs_seq[0, :, i]  # 自我车辆没有时延
    return obs_pre


def from_Kin_to_Occ(Kin, grid_step_y=4, grid_step_x=20, normal_value=np.array([100, 100, 30, 30])):
    # 从Kin的形式转化为Occ的形式
    # 输入参数模仿离散化的过程
    num = 10  # 车的数量
    feature_num = 4  # 特征的数量
    ego = Kin[0, :]
    # 这个就是返回的grid数据
    Occ = np.zeros((4, 7, 3))
    # 自我车辆的数据
    Occ[1, 0, 1] = ego[2] / normal_value[1]  # y
    Occ[2, 0, 1] = ego[3] / normal_value[2]
    Occ[3, 0, 1] = ego[4] / normal_value[3]
    for i in range(1, num):
        if Kin[i, 0] == 0:
            continue
        vehicle = Kin[i, :]
        ind_x = int(np.floor((vehicle[1] - ego[1]) / grid_step_x) + 1)
        ind_y = int(np.floor((vehicle[2] - ego[2] + 6) / grid_step_y))  # +6把-6到-2转化为0~4，后同
        if (0 <= ind_x <= 6 and 0 <= ind_y <= 2) and (ind_x != 0 or ind_y != 1) and Occ[0, ind_x, ind_y] == 0:  # 不在正后方
            for j in range(feature_num):
                fea = (vehicle[j + 1] - ego[j + 1]) / normal_value[j]
                if fea > 1:
                    fea = 1
                Occ[j, ind_x, ind_y] = fea
    return copy.deepcopy(Occ.astype(np.float32))
