import torch
import torch.utils
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Subset


def create_inout_sequences(data, seq_len, pred_len):
    inout_seq = []
    for i in range(len(data) - seq_len - pred_len + 1):
        train_seq = data[i: i+seq_len]
        train_label = data[i+seq_len: i+seq_len+pred_len]
        inout_seq.append((train_seq, train_label))
    return inout_seq



def create_dataLoader(data, seq_len, pred_len, val_num=0, batch_size=1):
    # data: np.array(data_len)，一维
    inout_seq = create_inout_sequences(data, seq_len, pred_len)
    # 很多个[seq_len, channel]组成的list， 此处channel=1
    trains = [torch.tensor(seq[:, None], dtype=torch.float32) for seq, _ in inout_seq]
    # 很多个[pred_len,]组成的list
    labels = [torch.tensor(label, dtype=torch.float32) for _, label in inout_seq]
    
    trains = torch.stack(trains)     # [train_num, seq_len, 1]
    labels = torch.stack(labels)     # [train_num, seq_len]
    dataset = TensorDataset(trains, labels)

    train_num = len(dataset) - val_num

    if val_num == 0:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return train_loader
    
    else:
        # 直接从末尾选取验证集
        train_dataset = Subset(dataset, range(train_num))
        val_dataset = Subset(dataset, range(train_num, train_num + val_num))

        # 创建dataloader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        return train_loader, val_loader