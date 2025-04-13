import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from my_attention.SelfAttention import ScaledDotProductAttention
from base_function import to_tensor
from convlstm import ConvLSTM
from my_attention.ExternalAttention import ExternalAttention
from my_attention.CBAM import CBAMBlock
from my_attention.SEAttention import SEAttention


class my_conv_lstm(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=None, kernel_size=None, num_layers=1,
                 H=7, W=3, batch_first=True, bias=True, return_all_layers=True):
        super().__init__()
        # input_dim: Number of channels in input 就是特征的数量
        if hidden_dim is None:
            hidden_dim = [1024]
        if kernel_size is None:
            kernel_size = [(3, 3)]
        self.hid_num = int(hidden_dim[num_layers - 1] * H * W)
        # self.hid_num = int(input_dim * H * W)
        self.out_num = 2  # 是否碰撞，两种可能性，one-hot编码
        self.num_layers = num_layers
        self.feature = input_dim
        self.H = H
        self.W = W
        self.l1 = ConvLSTM(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, num_layers=num_layers,
                           batch_first=batch_first, bias=bias, return_all_layers=return_all_layers)
        self.out = nn.Linear(self.hid_num, self.out_num)
        self.out = nn.Sequential(
            nn.Linear(self.hid_num, 128),  # 第一个全连接层
            nn.ReLU(),
            nn.Linear(128, self.out_num)  # 第二个全连接层
        )
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def forward(self, x):
        layer_output_list, last_state_list = self.l1(x)
        x2 = self.out(last_state_list[self.num_layers - 1][0].reshape(-1, self.hid_num))
        # x2 = self.out(x.view(x.size(0),-1))
        output = self.softmax(x2)
        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


if __name__ == "__main__":
    data = torch.randn((1, 3, 4, 3, 7))  # B, T（seq), C, H, W
    pre_model = my_conv_lstm()
    out = pre_model(data)
    print(out)
