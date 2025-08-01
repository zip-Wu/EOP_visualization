import torch
import torch.nn as nn
import numpy as np

class WeightedL1Loss(nn.Module):
    def __init__(self, weights):
        super(WeightedL1Loss, self).__init__()
        self.weights = weights

    def forward(self, predictions, targets):
        # 计算L1损失
        l1_loss = torch.pow((predictions - targets), 2)
        # 扩展权重到与损失形状匹配
        weights = self.weights.view(1, -1, 1)  # (1, pred_len, 1)
        weights = self.weights.unsqueeze(0).expand_as(l1_loss)
        # 应用权重
        weighted_loss = l1_loss * weights
        # 返回加权损失的平均值
        return weighted_loss.mean()

# 实现早停的类
class EarlyStopping:
    def __init__(self, patience=20, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score + self.delta:      # 如果loss比最好的loss要大的话
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''验证集的损失在该轮训练成功降低，正在保存当前模型……'''
        if self.verbose:
            print(f'验证集损失降低幅度为： ({self.val_loss_min:.8f} --> {val_loss:.8f}).  Saving model ...\n')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

# 先生成训练数据的序列值
def create_inout_sequences(input_data, tw, predict_len):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw - predict_len + 1):
        train_seq = input_data[i: i+tw]
        train_label = input_data[i+tw:i+tw+predict_len]
        inout_seq.append((train_seq, train_label))
    return inout_seq