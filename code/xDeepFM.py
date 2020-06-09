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


class xDeepFM(torch.nn.Module):
    def __init__(self, feature_size, field_size, embedding_size, cin_layer_nums=[30, 20], dnn_hidden = [128, 64]):
        super(xDeepFM, self).__init__()
        self.field_size = field_size
        self.cin_layer_num = len(cin_layer_nums)
        self.dnn_layer_num = len(dnn_hidden)
        self.fm_emb = torch.nn.Embedding(feature_size, embedding_size)
        self.fm_first_order_weight_emb = torch.nn.Embedding(feature_size, 1)
        self.dnn_layer = torch.nn.ModuleList(
            [torch.nn.Linear(dnn_hidden[i-1] if i > 0 else field_size * embedding_size, dnn_hidden[i]) for i in range(self.dnn_layer_num)]
        )
        #也有用卷积的，我觉得都可以
        self.cin_layer = torch.nn.ModuleList(
            [torch.nn.Linear(field_size * (cin_layer_nums[i-1] if i > 0 else field_size), cin_layer_nums[i]) for i in range(self.cin_layer_num)]
        )
        if self.cin_layer_num <= 0 or self.dnn_layer_num <= 0:
            print('model init fail, cin_layer_num or dnn_layer_num <= 0')
        self.fc = torch.nn.Linear(dnn_hidden[-1] + (field_size + np.sum(cin_layer_nums)) * embedding_size, 1)
    
    def forward(self, idxs, vals):
        emb = self.fm_emb(idxs) * vals.unsqueeze(-1)
        emb = emb.permute(0,2,1)
        
        #fm 一次项
        fm_first_order_weight = self.fm_first_order_weight_emb(idxs)
        fm_first_order = torch.sum(fm_first_order_weight.squeeze(-1) * vals, dim=-1, keepdim=True)
        
        #cin部分
        x, x0 = emb, emb.unsqueeze(-1)
        cin_list = [x]
        for i in range(self.cin_layer_num):
            #做两两交叉
            x = x0 * x.unsqueeze(-2)
            x = torch.flatten(x, start_dim = 2)
            x = self.cin_layer[i](x)
            cin_list.append(x)
        cin = torch.cat(cin_list, dim=2)
        cin = torch.flatten(cin, start_dim=1)

        dnn = torch.flatten(emb, start_dim=1)
        #dnn部分
        for i in range(self.dnn_layer_num):
            dnn = F.relu(self.dnn_layer[i](dnn))
        merge = torch.cat((cin, dnn), dim=1)
        return torch.sigmoid(fm_first_order + self.fc(merge))

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
cin_layer_nums = [40, 30, 20]
dnn_hidden = [200, 200]
device = torch.device('cuda:0')
#device = torch.device('cpu')

model = xDeepFM(dataset.feature_size, dataset.field_size, embedding_size, cin_layer_nums, dnn_hidden).to(device)


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
