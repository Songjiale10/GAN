import torch
import torch.nn as nn
from main import train_data,hidden_size
import torch.nn.functional as F

"""序列预测模型"""
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.rnn=nn.Lstm(train_data.shape[-1],
                         hidden_size,
                         batch_first=True
                         )
        self.fc1=nn.Linear(hidden_size,64)
        self.fc2=nn.Linear(64,1)

    def forward(self,inputs):
        _,(hs,cs)=self.rnn(inputs)
        x=F.dropout(hs[0])
        x=F.dropout(F.relu(self.fc1(x)))
        x=self.fc2(x)

        return torch.squeeze(1)#去掉维度为1的维度