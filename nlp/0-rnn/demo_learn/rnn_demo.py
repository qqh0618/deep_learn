# import torch
# X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
# H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))
# out = torch.matmul(X, W_xh) + torch.matmul(H, W_hh)
# out2 = torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0))
# print(out2)
import torch.nn as nn
import torch

rnn = nn.RNN(3, 5, 2)  # (input_size, hiden_size, num_layers)
input = torch.randn(5, 3, 3) # (seq_len, batch_size, input=input_size)
h0 = torch.randn(2, 3, 5) #(d*num_layers, batch_size, out=hidden_size)
outpit, hn = rnn(input, h0)
print(outpit, hn)

def self_rnn(input, weight_ih, weigth_hh, bias_ih, bis_hh, h_prev):
    """
    :param input: 输入的内容
    :param weight_ih: 输入层与隐藏层的权重矩阵[h_dim, input_size]
    :param weigth_hh:上一个隐藏层与当前隐藏层的权重矩阵[h_dim, h_dim]
    :param bias_ih:输入层的偏置
    :param bis_hh:隐藏层的偏置矩阵
    :param h_prev:上一时刻的状态
    :return:
    """
    batch_size, seq_len, input_size = input.shape
    h_dim = weight_ih.shape[0]
    h_out = torch.zeros(batch_size, seq_len, h_dim) # 初始化一个输出矩阵

    for t in range(seq_len):
        x = input[:, t]  # 获取当前时刻的输入特征, bs*input_size
        w_ih_batch = weight_ih.unsqueeze(0).tile(batch_size, 1, 1)  # 扩充一个维度，并复制batch_size次
        w_hh_batch = weigth_hh.unsqueeze(0).tile(batch_size, 1, 1)

        w_time_x = torch.bmm(w_ih_batch, x).squeeze(-1)  # bs*h_dim
        w_time_h = torch.bmm(w_hh_batch, h_prev.unsqueeze(2)).squeeze(-1)  # bs*h_dim
        h_prev = torch.tanh(w_time_x + bias_ih + w_time_h+bis_hh)

        h_out[:, t, :] = h_prev

    return h_out, h_prev.unsqueeze(0)

