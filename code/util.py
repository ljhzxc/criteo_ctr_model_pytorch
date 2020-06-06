import torch
from sklearn.metrics import roc_auc_score

def train(model, optimizer, data_loader, criterion, device, verbose=200):
    model.train()
    total_loss = 0
    for i, (feat_idxs, values, label) in enumerate(data_loader):
        feat_idxs, values, label = feat_idxs.to(device), values.to(device) ,label.to(device)
        y_pred = model(feat_idxs, values)
        loss = criterion(y_pred, label.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % verbose == 0:
            print('    - loss:', total_loss / verbose)
            total_loss = 0


def test(model, data_loader, device):
    model.eval()
    labels, predicts = list(), list()
    with torch.no_grad():
        for feat_idxs, values, label in data_loader:
            feat_idxs, values, label = feat_idxs.to(device), values.to(device) ,label.to(device)
            y_pred = model(feat_idxs, values)
            labels.extend(label.tolist())
            predicts.extend(y_pred.tolist())
    return roc_auc_score(labels, predicts)