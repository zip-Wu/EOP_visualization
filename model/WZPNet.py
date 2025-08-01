import torch
import torch.nn as nn
import torch.nn.functional as F




class WZPNet(nn.Module):
    def __init__(self, seq_out, d_model, dropout, seq_ar, 
                 seq_cnn, cnn_kernel, cnn_stride, cnn_channel, 
                 seq_gru, gru_layer, gru_hidden, 
                 skip_num, skip_len, skip_layer, skip_hidden
                 ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.seq_out = seq_out

        self.seq_ar = seq_ar
        self.seq_cnn = seq_cnn
        self.seq_gru = seq_gru
        self.skip_len = skip_len

        # 定义可学习的权重参数 w1, w2, w3, w4
        self.w1 = nn.Parameter(torch.tensor(1.0))  # 初始化为1
        self.w2 = nn.Parameter(torch.tensor(1.0))
        self.w3 = nn.Parameter(torch.tensor(1.0))
        self.w4 = nn.Parameter(torch.tensor(1.0))

        # AR
        if seq_ar > 0:
            # self.seq_ar = seq_ar
            self.ARLinear = nn.Linear(seq_ar, seq_out, bias=True)

        # CNN + GRU
        if seq_cnn > 0:
            # self.seq_cnn = seq_cnn
            self.cnn_kernel = cnn_kernel
            self.cnn_stride = cnn_stride
            self.cnn_channel = cnn_channel
            self.conv1 = nn.Conv2d(1, cnn_channel, (cnn_kernel, 1), cnn_stride)
            self.cnn_GRU = nn.GRU(cnn_channel, d_model, batch_first=True)
            self.cnnLinear = nn.Linear(d_model, seq_out, bias=True)

        # GRU
        if seq_gru > 0:
            # self.seq_gru = seq_gru
            self.gru_layer = gru_layer
            self.gru_hidden = gru_hidden
            self.gru1 = nn.GRU(1, gru_hidden, gru_layer, batch_first=True)
            self.gruLinear = nn.Linear(gru_hidden, seq_out, bias=True)

        # skip-GRU
        if skip_len > 0:
            self.skip_num = skip_num
            # self.skip_len = skip_len
            self.skip_layer = skip_layer
            self.skip_hidden = skip_hidden
            self.skip_gru = nn.GRU(1, skip_hidden, skip_layer, batch_first=True)
            self.skipLinear = nn.Linear(skip_len*skip_hidden, seq_out, bias=True)

    # [batch, seq_len, channel]
    def forward(self, x):
        batch = x.size(0)
        y = torch.zeros(batch, self.seq_out).to(x.device)    # [batch, seq_out]
        # AR层
        if self.seq_ar > 0:
            ar_out = self.AR_layer(x)     # [batch, seq_out]
            y = y + ar_out
        
        # CNN+GRU
        if self.seq_cnn > 0:
            cnn_out = self.CNN_layer(x)
            y = y + cnn_out

        # GRU
        if self.seq_gru > 0:
            gru_out = self.GRU_layer(x)
            y = y + gru_out

        # skip-GRU
        if self.skip_len > 0:
            skip_out = self.SKIP_layer(x)
            y = y + skip_out

        return y
        
        
        

            


    # [batch, seq_len, channel] Tensor
    def AR_layer(self, x):
        # [batch, seq_len, 1]
        x = torch.squeeze(x, -1)    # [batch, seq_len]
        x = x[:, -self.seq_ar:]     # [batch, seq_ar]
        x = self.ARLinear(x)        # [batch, seq_out]
        return x
    
    # [batch, seq_len, channel]
    def CNN_layer(self, x):
        # [batch, seq_len, 1]
        x = x[:, -self.seq_cnn:, :]    # [batch, seq_cnn, 1]
        x = torch.unsqueeze(x, 1)      # [batch, 1, seq_cnn, 1]
        x = F.relu(self.conv1(x))      # [batch, cnn_channel, seq_rnn*, 1]
        x = self.dropout(x)
        x = torch.squeeze(x, -1)       # [batch, cnn_channel, seq_rnn*]
        x = x.permute(0, 2, 1).contiguous()     # [batch, seq_rnn*, cnn_channel]
        _, x = self.cnn_GRU(x)           # [num_layers, batch, d_model]
        x = x[-1, :, :]                # [batch, d_model]
        x = self.dropout(x)
        x = self.cnnLinear(x)          # [batch, seq_out]
        return x
        
    # [batch, seq_len, channel]
    def GRU_layer(self, x):
        # [batch, seq_len, channel]
        x = x[:, -self.seq_gru:, :]    # [batch, seq_gru, 1]
        _, x = self.gru1(x)               # [num_layers, batch, gru_hidden]
        x = x[-1, :, :]             # [batch, gru_hidden]
        x = self.dropout(x)
        x = self.gruLinear(x)       # [batch, seq_out]
        return x
    
    # [batch, seq_len, channel]
    def SKIP_layer(self, x):
        # [batch, seq_len, channel]
        batch = x.size(0)
        x = x[:, -(self.skip_num*self.skip_len):, :].contiguous()    # [batch, seq_skip, 1]
        x = x.view(batch, self.skip_num, self.skip_len)            # [batch, skip_num, skip_len]
        x = x.permute(0, 2, 1).contiguous()                   # [batch, skip_len, skip_num]
        x = x.view(batch*self.skip_len, self.skip_num)       # [batch*skip_len, skip_num]
        x = torch.unsqueeze(x, -1)                         # [batch*skip_len, skip_num, 1]
        _, x = self.skip_gru(x)                       # [num_layers, batch*skip_len, skip_hidden]
        x = x[-1, :, :]                               # [batch*skip_len, skip_hidden]
        x = x.view(batch, self.skip_len*self.skip_hidden)    # [batch, skip_len*skip_hidden]
        x = self.dropout(x)
        x = self.skipLinear(x)                        # [batch, seq_out]
        return x


            


        
        
