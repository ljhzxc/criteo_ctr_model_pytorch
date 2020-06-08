import time
import gc
import joblib
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader

#local code
import util
from dataset import CriteoDataset


class AFM(torch.nn.Module):
    def __init__(self, feature_size, field_size, embedding_size, attention_size):
        super(AFM, self).__init__()
        self.field_size = field_size
        self.fm_bias = torch.nn.Parameter(torch.tensor(0.0))
        self.fm_first_order_weight_emb = torch.nn.Embedding(feature_size, 1)
        #这里指的是FM二阶项的向量
        self.fm_emb = torch.nn.Embedding(feature_size, embedding_size)
        self.attention_w = torch.nn.Linear(embedding_size, attention_size)
        self.attention_b = torch.nn.Parameter(torch.Tensor(attention_size))
        self.attention_h = torch.nn.Linear(attention_size, 1)
        self.fc_p = torch.nn.Linear(embedding_size, 1)

    #idx和vals的shape = (batch_size, field_size)
    def forward(self, idxs, vals):
        #fm 二次型-带attention
        emb = self.fm_emb(idxs) * vals.unsqueeze(-1)

        '''
        #计算attention的naive版本，速度慢很多
        emb_list, attention_list = [], []
        for i in range(self.field_size - 1):
            for j in range(i+1, self.field_size):
                tmp = emb[:,i,:] * emb[:,j,:]
                att_e = self.attention_h(F.relu(self.attention_w(tmp)) + self.attention_b)
                emb_list.append(tmp.tolist()), attention_list.append(att_e.tolist())

        emb_cross = torch.tensor(emb_list).permute(1,0,2)
        #计算attention单项数值e_{ij}
        att_e_cross = torch.tensor(attention_list).permute(1,0,2).squeeze(-1)
        '''

        #参考其他人的版本
        row, col = list(), list()
        for i in range(self.field_size - 1):
            for j in range(i + 1, self.field_size):
                row.append(i), col.append(j)
        p, q = emb[:, row], emb[:, col]
        emb_cross = p * q
        #计算attention单项数值e_{ij}
        att_e_cross = self.attention_h(F.relu(self.attention_w(emb_cross)) + self.attention_b).squeeze(-1)
        #计算attention系数a_{ij}
        att_a_cross = F.softmax(att_e_cross, dim=1)
        #计算embedding做一个sumpooling
        sum_emb = torch.sum(att_a_cross.unsqueeze(-1) * emb_cross, dim=1)
        fm_second_order_with_att = self.fc_p(sum_emb).squeeze(-1)

        #fm 一次项
        fm_first_order_weight = self.fm_first_order_weight_emb(idxs)
        fm_first_order = torch.sum(fm_first_order_weight.squeeze(-1) * vals, dim=-1)

        return torch.sigmoid(self.fm_bias + fm_first_order + fm_second_order_with_att)

    
batch_size = 512
embedding_size = 32
attention_size = 64
lr = 0.001
weight_l2 = 1e-4
epoch = 10
device = torch.device('cuda:0')
#device = torch.device('cpu')

# dataset = CriteoDataset('../data/train_100w.txt')
# with open('../store/criteo_dataset_100w.jb', 'wb') as file:
#     joblib.dump(dataset, file)
dataset = joblib.load('../store/criteo_dataset_100w.jb')

dataset_num = len(dataset)
train_num = int(dataset_num * 0.8)
valid_num = int(dataset_num * 0.1)
test_num = dataset_num - train_num - valid_num
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    dataset, (train_num, valid_num, test_num))
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=32)
valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=32)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=32)

model = AFM(dataset.feature_size, dataset.field_size, embedding_size, attention_size).to(device)
st = time.time()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_l2)
for epoch_i in range(epoch):
    util.train(model, optimizer, train_data_loader, criterion, device, verbose=200)
    auc = util.test(model, valid_data_loader, device)
    print('epoch:', epoch_i, 'validation: auc:', auc)
    print('cost total time: %d'%(time.time() - st))
auc = util.test(model, test_data_loader, device)
print('test auc:', auc)