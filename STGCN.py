# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 15:12:27 2019

@author: yang an
"""

import os
import shutil
from time import time
from datetime import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
# import configparser
# import torch.nn.functional as F
# from torch.autograd import Variable
from lib.utils import compute_val_loss, evaluate, predict
from lib.data_preparation import read_and_generate_dataset
from lib.utils import scaled_Laplacian, cheb_polynomial, get_adjacency_matrix
from model import STGCN as model

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')

# parser.add_argument('--max_epoch', type=int, default=40, help='Epoch to run [default: 250]')
parser.add_argument('--max_epoch', type=int, default=40, help='Epoch to run [default: 250]')

# parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 32]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')

parser.add_argument('--learning_rate', type=float, default=0.0005, help='Initial learning rate [default: 0.001]')

# momentum 加速梯度下降的一种方法,加速收敛，消除初始权值带来的不确定性
parser.add_argument('--momentum', type=float, default=0.99, help='Initial learning rate [default: 0.9]')

parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--length', type=int, default=60, help='Size of temporal : 6')

parser.add_argument("--force", type=str, default=True,
                    help="remove params dir", required=False)
parser.add_argument("--data_name", type=str, default=4,
                    help="the number of data documents [8/4]", required=False)
parser.add_argument('--num_point', type=int, default=307, help='road Point Number [170/307] ', required=False)

# decay rate of learning rate 学习率的衰减
parser.add_argument('--decay', type=float, default=0.92, help='decay rate of learning rate [0.97/0.92]')

FLAGS = parser.parse_args()
f = FLAGS.data_name
decay = FLAGS.decay

# 获取数据位置 （adj_filename 邻接矩阵）（graph_signal_matrix_filename 5分钟一次的交通数据）
adj_filename = 'data/PEMS0%s/distance.csv' % f
graph_signal_matrix_filename = 'data/PEMS0%s/pems0%s.npz' % (f, f)

Length = FLAGS.length
batch_size = FLAGS.batch_size

# 节点数量，这里是307或170
num_nodes = FLAGS.num_point

epochs = FLAGS.max_epoch
learning_rate = FLAGS.learning_rate
optimizer = FLAGS.optimizer

# 每小时12个数据
points_per_hour = 12

# 预测12个数据
num_for_predict = 12

# week day hour分别是周、日、邻近，每个数据的起始和结束点
# num_of_weeks = 2就是一个初始一个结束，num_of_days = 1就是初始和结束都相同
num_of_weeks = 2
num_of_days = 1
num_of_hours = 2

# 节点数 307或170
num_of_vertices = FLAGS.num_point

# 3个特征 速度、流量、时间占有率
num_of_features = 3

# 设置是否需要将训练集和验证集合并到一起训练
merge = False
model_name = 'STGCN_%s' % f

# 训练参数存放地址，以及文件夹名
params_dir = 'result/exp/STGCN'

# 预测结果存放地址，以及文件名
prediction_path = 'result/prediction/STGCN_0%s' % f

# wdecay：weight decay（权值衰减）= L2 Regularization，pytorch中的L2正则项
wdecay = 0.00

device = torch.device(FLAGS.device)

# read laplace matrix
# 获得、处理邻接矩阵数据
# (adj_filename, num_nodes)=('data/PEMS08/distance.csv',170)
adj = get_adjacency_matrix(adj_filename, num_nodes)
# 计算拉普拉斯
adjs = scaled_Laplacian(adj)
supports = (torch.tensor(adjs)).type(torch.float32).to(device)
# print(adj.shape)
# print(adjs.shape)
# print(supports.shape)
# print(adj[0])
# print(adjs[0])
# print(supports[0])
# print(supports)
# print('train:supports=', supports[1][1:3])
# print('train:supports=', supports.shape, '\n')
print('Model is %s' % (model_name))

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
# print(timestamp)
if params_dir != "None":
    params_path = os.path.join(params_dir, model_name)
else:
    params_path = 'params/%s_%s/' % (model_name, timestamp)

# check parameters file
if os.path.exists(params_path) and not FLAGS.force:
    raise SystemExit("Params folder exists! Select a new params path please!")
else:
    if os.path.exists(params_path):
        shutil.rmtree(params_path)
    os.makedirs(params_path)
    print('Create params directory %s' % (params_path))

# main函数
if __name__ == "__main__":
    # read all data from graph signal matrix file
    print("Reading data...")
    # Input: train / valid  / test : length x 3 x NUM_POINT x 12
    all_data = read_and_generate_dataset(graph_signal_matrix_filename,
                                         num_of_weeks,
                                         num_of_days,
                                         num_of_hours,
                                         num_for_predict,
                                         points_per_hour,
                                         merge)

    # test set ground truth
    print(all_data['train']['week'][0][0][0][0])
    true_value = all_data['test']['target']
    print('true_value.shape=',true_value.shape)

    # 设置数据训练的一些参数，将数据转换为torch.Tensor的格式
    # training set data loader
    train_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['train']['week']),
            torch.Tensor(all_data['train']['day']),
            torch.Tensor(all_data['train']['recent']),
            torch.Tensor(all_data['train']['target'])
        ),
        batch_size=batch_size,
        shuffle=True
    )

    # validation set data loader
    val_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['val']['week']),
            torch.Tensor(all_data['val']['day']),
            torch.Tensor(all_data['val']['recent']),
            torch.Tensor(all_data['val']['target'])
        ),
        batch_size=batch_size,
        shuffle=False
    )

    # testing set data loader
    test_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['test']['week']),
            torch.Tensor(all_data['test']['day']),
            torch.Tensor(all_data['test']['recent']),
            torch.Tensor(all_data['test']['target'])
        ),
        batch_size=batch_size,
        shuffle=False
    )

    # save Z-score mean and std
    stats_data = {}
    for type_ in ['week', 'day', 'recent']:
        stats = all_data['stats'][type_]
        stats_data[type_ + '_mean'] = stats['mean']
        stats_data[type_ + '_std'] = stats['std']
    # print('stats_data[week_mean].shape=',stats_data['week_mean'].shape)
    # print('stats_data[day_mean].shape=', stats_data['day_mean'].shape)
    # print('stats_data[recent_mean].shape=', stats_data['recent_mean'].shape)
    # print('stats_data[week_std].shape=',stats_data['week_std'].shape)
    # print('stats_data[day_std].shape=', stats_data['day_std'].shape)
    # print('stats_data[recent_std].shape=', stats_data['recent_std'].shape)
    np.savez_compressed(
        os.path.join(params_path, 'stats_data'),
        **stats_data
    )

    # loss function MSE
    loss_function = nn.MSELoss()

    # get model's structure
    net = model(c_in=num_of_features, c_out=64,
                num_nodes=num_nodes, week=24,
                day=12, recent=24,
                K=3, Kt=3)
    net.to(device)  # to cuda

    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=wdecay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10,20], gamma=0.1, last_epoch=-1)

    # calculate origin loss in epoch 0
    compute_val_loss(net, val_loader, loss_function, supports, device, epoch=0)

    # compute testing set MAE, RMSE, MAPE before training
    evaluate(net, test_loader, true_value, supports, device, epoch=0)

    clip = 5
    his_loss = []
    train_time = []
    for epoch in range(1, epochs + 1):
        train_l = []
        start_time_train = time()
        for train_w, train_d, train_r, train_t in train_loader:
            train_w = train_w.to(device)
            train_d = train_d.to(device)
            train_r = train_r.to(device)
            train_t = train_t.to(device)
            net.train()  # train pattern
            optimizer.zero_grad()  # grad to 0

            output, _, _ = net(train_w, train_d, train_r, supports)
            loss = loss_function(output, train_t)
            # backward p
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(net.parameters(), clip)

            # update parameter
            optimizer.step()

            training_loss = loss.item()
            train_l.append(training_loss)
        scheduler.step()
        end_time_train = time()
        train_l = np.mean(train_l)
        print('epoch step: %s, training loss: %.2f, time: %.2fs'
              % (epoch, train_l, end_time_train - start_time_train))
        train_time.append(end_time_train - start_time_train)

        # compute validation loss
        valid_loss = compute_val_loss(net, val_loader, loss_function, supports, device, epoch)
        his_loss.append(valid_loss)

        # evaluate the model on testing set
        evaluate(net, test_loader, true_value, supports, device, epoch)

        params_filename = os.path.join(params_path,
                                       '%s_epoch_%s_%s.params' % (model_name,
                                                                  epoch, str(round(valid_loss, 2))))
        torch.save(net.state_dict(), params_filename)
        print('save parameters to file: %s' % (params_filename))

    print("Training finished")
    print("Training time/epoch: %.2f secs/epoch" % np.mean(train_time))

    bestid = np.argmin(his_loss)

    print("The valid loss on best model is epoch%s_%s" % (str(bestid + 1), str(round(his_loss[bestid], 4))))
    best_params_filename = os.path.join(params_path,
                                        '%s_epoch_%s_%s.params' % (model_name,
                                                                   str(bestid + 1), str(round(his_loss[bestid], 2))))
    net.load_state_dict(torch.load(best_params_filename))
    start_time_test = time()
    prediction, spatial_at, temporal_at = predict(net, test_loader, supports, device)
    end_time_test = time()
    evaluate(net, test_loader, true_value, supports, device, epoch)
    test_time = np.mean(end_time_test - start_time_test)
    print("Test time: %.2f" % test_time)

    np.savez_compressed(
        os.path.normpath(prediction_path),
        prediction=prediction,
        spatial_at=spatial_at,
        temporal_at=temporal_at,
        ground_truth=all_data['test']['target']
    )
print(timestamp)
