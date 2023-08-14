# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 13:20:23 2018

@author: yang an
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm, BatchNorm1d

"""
x-> [batch_num,in_channels,num_nodes,tem_size],
"""


# 1、空间注意力层 当执行GCN时，我们将邻接矩阵A和空间注意力矩阵S结合起来动态调整节点之间的权重。
class SATT(nn.Module):
    # def __init__(self, c_in='3', num_nodes='170/307', tem_size='24/12/24'):
    def __init__(self, c_in, num_nodes, tem_size):
        # print('c_in=',c_in)
        super(SATT, self).__init__()
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(tem_size, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)

        # nn.Parameter 一组可训练参数
        self.w = nn.Parameter(torch.rand(tem_size, c_in), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w)
        # 对self.w的参数进行初始化，且服从Xavier均匀分布

        self.b = nn.Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True)
        # 全为0，因此不存在初始化

        self.v = nn.Parameter(torch.rand(num_nodes, num_nodes), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.v)
        # 对self.v的参数进行初始化，且服从Xavier均匀分布

    # seq：序列
    def forward(self, seq):
        # print('seq的维度：',seq.shape)，seq哪里来的？

        c1 = seq
        f1 = self.conv1(c1).squeeze(1)  # batch_size,num_nodes,length_time
        # print('f1的维度：',f1.shape)

        c2 = seq.permute(0, 3, 1, 2)  # b,c,n,l->b,l,n,c
        # print('c2的维度：', c2.shape)

        f2 = self.conv2(c2).squeeze(1)  # b,c,n
        # print('f2的维度：', f2.shape)

        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        # print('logits的维度：', logits.shape)
        logits = torch.matmul(self.v, logits)
        # print('logits的维度：', logits.shape, '\n')
        ##normalization
        a, _ = torch.max(logits, 1, True)
        logits = logits - a
        coefs = torch.softmax(logits, -1)
        # print('coefs=',coefs.shape)
        return coefs


# 2、图卷积层 具有空间注意分数的K阶chebyshev图卷积
class cheby_conv_ds(nn.Module):
    def __init__(self, c_in, c_out, K):
        super(cheby_conv_ds, self).__init__()
        c_in_new = (K) * c_in
        self.conv1 = Conv2d(c_in_new, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)
        self.K = K

    def forward(self, x, adj, ds):
        nSample, feat_in, nNode, length = x.shape # 32, 3/64 , 307 , 24/12
        Ls = []
        L0 = torch.eye(nNode).cuda()
        L1 = adj

        L = ds * adj
        I = ds * torch.eye(nNode).cuda()
        Ls.append(I)
        Ls.append(L)
        for k in range(2, self.K):
            L2 = 2 * torch.matmul(adj, L1) - L0
            L0, L1 = L1, L2
            L3 = ds * L2
            Ls.append(L3)

        Lap = torch.stack(Ls, 1)  # [B, K,nNode, nNode]
        # print(Lap)
        Lap = Lap.transpose(-1, -2)
        x = torch.einsum('bcnl,bknq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        # print('out=', out.shape)
        return out

    ###ASTGCN_block


# 3、时间注意力层
class TATT(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        super(TATT, self).__init__()
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(num_nodes, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True)
        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.v)

    def forward(self, seq):
        # print('seq的维度：', seq.shape)
        c1 = seq.permute(0, 1, 3, 2)  # b,c,n,l->b,c,l,n
        # print('c1的维度：', c1.shape)
        f1 = self.conv1(c1).squeeze(1)  # b,l,n
        # print('f1的维度：', f1.shape)

        c2 = seq.permute(0, 2, 1, 3)  # b,c,n,l->b,n,c,l
        # print('c2的维度：', c2.shape)
        f2 = self.conv2(c2).squeeze(1)  # b,c,l
        # print('f2的维度：', f2.shape)

        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        # print('logits的维度：', logits.shape)
        logits = torch.matmul(self.v, logits)
        # print('logits的维度：', logits.shape, '\n')
        ##normalization
        a, _ = torch.max(logits, 1, True)
        logits = logits - a
        coefs = torch.softmax(logits, -1)
        # print('coefs=', coefs.shape)
        return coefs


# 4、时空块 整体时空块的搭建，用到了前面的1、2、3
class ST_BLOCK_0(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt):
        super(ST_BLOCK_0, self).__init__()

        self.conv1 = Conv2d(c_in, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)
        self.TATT = TATT(c_in, num_nodes, tem_size)
        self.SATT = SATT(c_in, num_nodes, tem_size)
        self.dynamic_gcn = cheby_conv_ds(c_in, c_out, K)
        self.K = K
        self.time_conv = Conv2d(c_out, c_out, kernel_size=(1, Kt), padding=(0, 1),
                                stride=(1, 1), bias=True)
        # self.bn=BatchNorm2d(c_out)
        self.bn = LayerNorm([c_out, num_nodes, tem_size])

    def forward(self, x, supports):
        x_input = self.conv1(x)
        T_coef = self.TATT(x)
        T_coef = T_coef.transpose(-1, -2)
        x_TAt = torch.einsum('bcnl,blq->bcnq', x, T_coef)
        S_coef = self.SATT(x)  # B x N x N
        # print('supports.shape =',supports.shape)
        # print('S_coef.shape =',S_coef.shape)
        # print('x_TAt.shape =',x_TAt.shape)
        # print('T_coef.shape =',T_coef.shape,'\n')
        spatial_gcn = self.dynamic_gcn(x_TAt, supports, S_coef)
        spatial_gcn = torch.relu(spatial_gcn)
        time_conv_output = self.time_conv(spatial_gcn)
        out = self.bn(torch.relu(time_conv_output + x_input))
        # print('out=', out.shape)
        # print('ST_BLOCK_0:supports=', supports[1][1:3])
        # print('ST_BLOCK_0:supports=', supports.shape,'\n')
        return out, S_coef, T_coef
