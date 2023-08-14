# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 11:21:28 2018

@author: yang an
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm, InstanceNorm2d
from utils import ST_BLOCK_0  # STGCN

"""
the parameters:
x-> [batch_num,in_channels,num_nodes,tem_size],
输入x-> [ batch数, 通道数, 节点数, 时间],
"""


# 5、周、日、邻近中的一个整体模型 两个时空块+FC
class STGCN_block(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt):
        super(STGCN_block, self).__init__()
        self.block1 = ST_BLOCK_0(c_in, c_out, num_nodes, tem_size, K, Kt)
        self.block2 = ST_BLOCK_0(c_out, c_out, num_nodes, tem_size, K, Kt)
        self.final_conv = Conv2d(tem_size, 12, kernel_size=(1, c_out), padding=(0, 0),
                                 stride=(1, 1), bias=True)
        self.w = Parameter(torch.zeros(num_nodes, 12), requires_grad=True)
        nn.init.xavier_uniform_(self.w)

    # 前向传播
    def forward(self, x, supports):
        x, _, _ = self.block1(x, supports)
        x, d_adj, t_adj = self.block2(x, supports)
        x = x.permute(0, 3, 2, 1)
        x = self.final_conv(x).squeeze().permute(0, 2, 1)  # b,n,12
        x = x * self.w
        return x, d_adj, t_adj


# 6、周、日、邻近整体三个模型
class STGCN(nn.Module):
    # c_in 特征数，速度、流量、时间占有率=3
    # c_out 输出通道数
    # num_nodes 节点数，170/307
    # week 周数据 24=12*2（12位一个小时数据数）
    # day 日数据 12=12*1
    # recent 邻近数据 24=12*2
    # K 图卷积维度=3
    # Kt 对时间普通卷积的维度=3
    def __init__(self, c_in, c_out, num_nodes, week, day, recent, K, Kt):
        super(STGCN, self).__init__()
        # 周数据模块
        self.block_w = STGCN_block(c_in, c_out, num_nodes, week, K, Kt)
        # 日数据模块
        self.block_d = STGCN_block(c_in, c_out, num_nodes, day, K, Kt)
        # 邻近数据模块
        self.block_r = STGCN_block(c_in, c_out, num_nodes, recent, K, Kt)
        # self.block_gru = GRU(c_in, c_out, num_nodes, week, day, recent, K, Kt)
        # 数据归一化
        self.bn = BatchNorm2d(c_in, affine=False)

    # supports:拉普拉斯矩阵
    def forward(self, x_w, x_d, x_r, supports):
        x_w = self.bn(x_w)
        x_d = self.bn(x_d)
        x_r = self.bn(x_r)
        # print(supports[0])
        # print(supports.shape)
        x_w, _, _ = self.block_w(x_w, supports)
        x_d, _, _ = self.block_d(x_d, supports)
        x_r, d_adj_r, t_adj_r = self.block_r(x_r, supports)
        # x_gru, _, _  = self.block_gru(x_w, x_d, x_r, supports)
        # print(x_gru.shape)
        out = x_w + x_d + x_r
        return out, d_adj_r, t_adj_r
