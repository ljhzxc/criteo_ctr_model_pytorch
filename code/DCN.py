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


class DCN(torch.nn.Module):
    def __init__(self, feature_size, field_size, embedding_size, cross_layer_num, dnn_hidden = [128, 64]):
        super(DCN, self).__init__()
        self.field_size = field_size
        self.cross_layer_num = cross_layer_num
        self.dnn_layer_num = len(dnn_hidden)
        input_size = field_size * embedding_size
        self.fm_emb = torch.nn.Embedding(feature_size, embedding_size)
        self.cross_w = torch.nn.ModuleList(
            [torch.nn.Linear(input_size, 1, bias=False) for _ in range(cross_layer_num)]
        )
        self.cross_b = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(input_size,)) for _ in range(cross_layer_num)]
        )
        self.dnn_layer = torch.nn.ModuleList(
            [torch.nn.Linear(dnn_hidden[i-1] if i > 0 else input_size, dnn_hidden[i]) for i in range(self.dnn_layer_num)]
        )
        self.fc = torch.nn.Linear(input_size + (dnn_hidden[-1] if self.dnn_layer_num > 0 else input_size), 1)
    
    def forward(self, idxs, vals):
        #fm 二次型-带attention
        emb = self.fm_emb(idxs) * vals.unsqueeze(-1)
        emb = torch.flatten(emb, start_dim=1)
        
        #cross部分
        x0 = emb
        cross = x0
        for i in range(cross_layer_num):
            #每层的cross计算时，w_l可以先和x_l进行，减少计算量
            cross = self.cross_w[i](cross) * x0 + self.cross_b[0] + x0
        
        dnn = emb
        #dnn部分
        for i in range(self.dnn_layer_num):
            #每层的cross计算时，w_l可以先和x_l进行，减少计算量
            dnn = F.relu(self.dnn_layer[i](dnn))
        merge = torch.cat((cross, dnn), dim=1)
        return torch.sigmoid(self.fc(merge))

# dataset = CriteoDataset('../data/train_100w.txt')
# with open('../store/criteo_dataset_100w.jb', 'wb') as file:
#     joblib.dump(dataset, file)
dataset = joblib.load('../store/criteo_dataset_100w.jb')

batch_size = 512
lr = 0.001
weight_l2 = 1e-4
epoch = 10
dataset_num = len(dataset)
train_num = int(dataset_num * 0.8)
valid_num = int(dataset_num * 0.1)
test_num = dataset_num - train_num - valid_num
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    dataset, (train_num, valid_num, test_num))
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=32)
valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=32)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=32)


embedding_size = 32
cross_layer_num = 3
dnn_hidden = [128, 64]
device = torch.device('cuda:0')
#device = torch.device('cpu')
model = DCN(dataset.feature_size, dataset.field_size, embedding_size, cross_layer_num, dnn_hidden).to(device)


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
