import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from LS import LS_fit
from LS import gen_data
from LS import gen_res
from LS import gen_test
from data.data_loader import create_dataLoader
from data.data_loader import create_inout_sequences
from model.WZPNet import WZPNet
from utils_ANN import WeightedL1Loss




def train_WZPNet(start_time, end_time, EOP, pred_len, seq_len, seq_out, d_model, dropout,
                 seq_ar, seq_cnn, cnn_kernel, cnn_stride, cnn_channel, seq_gru, gru_layer, gru_hidden,
                 skip_num, skip_len, skip_layer, skip_hidden, num_epoch, batch_size):

    data = gen_data('./data/data_origin.txt', start_time, end_time)
    # weights = (1 / np.arange(len(data), 0, -1))**2
    LS_weights = None
    train_data = gen_res(data, EOP, LS_weights)



    train_loader, val_loader = create_dataLoader(train_data, seq_len, seq_out, val_num=50, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WZPNet(seq_out=seq_out, d_model=d_model, dropout=dropout, seq_ar=seq_ar,
                seq_cnn=seq_cnn, cnn_kernel=cnn_kernel, cnn_stride=cnn_stride, cnn_channel=cnn_channel,
                seq_gru=seq_gru, gru_layer=gru_layer, gru_hidden=gru_hidden,
                skip_num=skip_num, skip_len=skip_len, skip_layer=skip_layer, skip_hidden=skip_hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    # 定义权重，假设第一个预测点的权重大，逐渐减小
    weights = torch.linspace(1.0, 0.1, steps=seq_out).to(device)
    # weights = torch.tensor((1 / np.arange(1, pred_len+1))).to(device)
    # weights = torch.tensor( np.tile((1 / np.arange(1, seg_len+1)), pred_len // seg_len) ).to(device)
    # weights = torch.tensor( np.repeat((1 / np.arange(1, (pred_len//seg_len)+1)), seg_len) ).to(device)
    # weights = torch.linspace(1.0, 0.1, steps=pred_len).to(device)
    # 创建损失函数实例
    criterion = WeightedL1Loss(weights)

    temp_loss = 10
    epoch_save = 0
    temp_train = 10
    for epoch in range(num_epoch):
        model.train()
        train_loss = []
        # input [batch, seq_len, channel] labels [batch, seq_out]
        for input, labels in train_loader:
            input = input.to(device)
            labels = labels.to(device)    # [batch, seq_out]
            output = model(input)
            # 反向传播和优化
            optimizer.zero_grad()
            loss = criterion(output, labels)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        train_loss = np.mean(train_loss)
        # 每个epoch验证一次模型
        model.eval()
        val_loss = []
        with torch.no_grad():
            for input, labels in val_loader:
                input = input.to(device)
                labels = labels.to(device)
                output = model(input)
                loss = criterion(output, labels)
                val_loss.append(loss.item())
        val_loss = np.mean(val_loss)

        print(f'Epoch [{epoch + 1}], 训练集上的平均损失(MSE)为：{train_loss*1000} (RMSE)')
        print(f'Epoch [{epoch + 1}], 验证集上的平均损失(MSE)为：{val_loss*1000} (RMSE)')
        if train_loss < temp_train:
            temp_train = train_loss
        else:
            print("训练损失没有下降……………………\n")
        if val_loss < temp_loss:
            temp_loss = val_loss
            print("保存当前模型\n")
            torch.save(model.state_dict(), 'checkpoint.pt')
            epoch_save = epoch

    model.load_state_dict(torch.load('checkpoint.pt', weights_only=True))

    print(epoch_save)
    print(temp_loss*1000)


    train_res = torch.tensor(train_data[-seq_len:], dtype=torch.float32).reshape(1, seq_len, 1).to(torch.device(device))
    _, LS_forecast = LS_fit(data, 360, EOP, LS_weights)
    model.eval()
    forecast = []
    with torch.no_grad():
        for i in range(pred_len // seq_out):
            res_output = model(train_res)
            forecast.append(res_output.view(-1).to('cpu'))
            train_res = torch.cat((train_res[:, seq_out:, :], res_output.unsqueeze(-1)), dim=1)
        forecast = np.ravel(np.array(forecast))

    final = LS_forecast+forecast

    for num in final:
        print(f'{num:.8f}')


if __name__ == '__main__':
    start_time = '2005-5-13'
    end_time = '2025-5-13'
    EOP = 'X'

    pred_len = 360
    seq_len = 256

    seq_out = 20
    d_model = 64
    dropout = 0
    seq_ar = 256

    seq_cnn = 0
    cnn_kernel = 4
    cnn_stride = 1
    cnn_channel = 1


    seq_gru = 0
    gru_layer = 0
    gru_hidden = 0

    skip_num = 200//3
    skip_len = 3
    skip_layer = 1
    skip_hidden = 64

    num_epoch = 20
    batch_size = 64
    train_WZPNet(start_time, end_time, EOP, pred_len, seq_len, seq_out, d_model, dropout,
                 seq_ar, seq_cnn, cnn_kernel, cnn_stride, cnn_channel, seq_gru, gru_layer, gru_hidden,
                 skip_num, skip_len, skip_layer, skip_hidden, num_epoch, batch_size)