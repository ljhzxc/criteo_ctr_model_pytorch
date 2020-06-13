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


class NFM(torch.nn.Module):
    def __init__(self, feature_size, field_size, embedding_size, dnn_hidden):
        super(NFM, self).__init__()
        self.field_size = field_size
        self.dnn_layer_num = len(dnn_hidden)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.fm_first_order_weight_emb = torch.nn.Embedding(feature_size, 1)
        self.emb = torch.nn.Embedding(feature_size, embedding_size)
        self.dnn_layer = torch.nn.ModuleList(
            [torch.nn.Linear(dnn_hidden[i-1] if i > 0 else embedding_size, dnn_hidden[i]) for i in range(self.dnn_layer_num)]
        )
        self.fc = torch.nn.Linear(dnn_hidden[-1] if self.dnn_layer_num > 0 else embedding_size, 1)

    def forward(self, idxs, vals):
        emb = self.emb(idxs) * vals.unsqueeze(-1)
        
        #fm 一次项
        fm_first_order_weight = self.fm_first_order_weight_emb(idxs)
        fm_first_order = torch.sum(fm_first_order_weight.squeeze(-1) * vals, dim=-1).squeeze(-1)
        
        #fm 二次型
        tmp = torch.sum(emb, dim=1)
        square_of_sum = tmp * tmp
        sum_of_square = torch.sum(emb * emb, dim=1)
        fm_second_order = 0.5 * (square_of_sum - sum_of_square)

        x = fm_second_order
        #dnn部分
        for i in range(self.dnn_layer_num):
            x = self.dnn_layer[i](x)
            x = F.relu(x)
        dnn_out = self.fc(x).squeeze(-1)
        return torch.sigmoid(dnn_out + fm_first_order + self.bias)
    

# dataset = CriteoDataset('../data/train_100w.txt')
# with open('../store/criteo_dataset_100w.jb', 'wb') as file:
#     joblib.dump(dataset, file)
dataset = joblib.load('../store/criteo_dataset_100w.jb')

batch_size = 512
dataset_num = len(dataset)
train_num = int(dataset_num * 0.8)
valid_num = int(dataset_num * 0.1)
test_num = dataset_num - train_num - valid_num
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    dataset, (train_num, valid_num, test_num))
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=32)
valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=32)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=32)

embedding_size = 10
dnn_hidden = [200, 100]
device = torch.device('cuda:0')
#device = torch.device('cpu')
model = NFM(dataset.feature_size, dataset.field_size, embedding_size, dnn_hidden).to(device)

lr = 0.001
weight_l2 = 1e-4
epoch = 10
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