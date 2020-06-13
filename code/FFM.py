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


class FFM(torch.nn.Module):
    def __init__(self, feature_size, field_size, embedding_size):
        super(FFM, self).__init__()
        self.emb = torch.nn.ModuleList(
            [torch.nn.Embedding(feature_size, embedding_size) for _ in range(field_size)]
        )
        self.fm_first_order_weight = torch.nn.Embedding(feature_size, 1)
        self.field_size = field_size


    def forward(self, idxs, vals):
        #fm 一次项
        fm_first_order = (self.fm_first_order_weight(idxs) * vals.unsqueeze(-1))
        fm_first_order = torch.sum(fm_first_order.squeeze(-1), dim = 1)

        ffm_second_order_embs = []
        for i in range(self.field_size):
            ffm_second_order_embs.append(self.emb[i](idxs) * vals.unsqueeze(-1))

        ffm_second_order_list = list()
        for i in range(self.field_size - 1):
            for j in range(i + 1 , self.field_size - 1):
                ffm_second_order_list.append(ffm_second_order_embs[i][:, j] * ffm_second_order_embs[j][:, i])

        fm_second_order = torch.cat(ffm_second_order_list, dim = 1)
        fm_second_order = torch.sum(fm_second_order, dim = 1)

        return torch.sigmoid(fm_first_order + fm_second_order)

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
device = torch.device('cuda:0')
#device = torch.device('cpu')
model = FFM(dataset.feature_size, dataset.field_size, embedding_size).to(device)


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