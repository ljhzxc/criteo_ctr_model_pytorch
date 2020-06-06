import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import gc
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import log_loss, roc_auc_score

def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df



df = pd.read_csv('../data/train_100w.txt', header=None, sep='\t')
num_cols = [f'I{i+1}' for i in range(13)]
cate_cols = [f'C{i+1}' for i in range(26)]
columns = ['label'] + num_cols + cate_cols
df.columns = columns

scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

le = LabelEncoder()
for cate_col in cate_cols:
    df[cate_col].fillna("", inplace=True)
    df[cate_col]  = le.fit_transform(df[cate_col])

df = reduce_mem(df)
train_num = (int)(df.shape[0]*0.8)
dev_num = (int)(df.shape[0]*0.1)
train_df = df.iloc[:train_num]
dev_df = df.iloc[train_num:train_num+dev_num]
test_df = df.iloc[train_num+dev_num:]

st_time = time.time()
lgb_clf = lgb.LGBMClassifier(
    learning_rate=0.003,
    n_estimators=2000, 
    num_leaves=32,
    subsample=0.9, 
    colsample_bytree=0.7,
    objective='binary',
    random_state=3,
)
print('************** training **************')
print(train_df.shape)
lgb_clf.fit(
    train_df.iloc[:, 1:],
    train_df.iloc[:, 0],
    categorical_feature=cate_cols,
    eval_set=[(dev_df.iloc[:, 1:], dev_df.iloc[:, 0])],
    early_stopping_rounds=100,
    verbose=100,
)
print(f' train using time: {time.time() - st_time}')


dev_pred = lgb_clf.predict_proba(dev_df.iloc[:, 1:])[:, 1]
dev_labels = dev_df.iloc[:, 0]
print(f' log_loss: {log_loss(dev_labels, dev_pred)}    auc: {roc_auc_score(dev_labels, dev_pred)}')

test_pred = lgb_clf.predict_proba(test_df.iloc[:, 1:])[:, 1]
test_labels = test_df.iloc[:, 0]
print(f' log_loss: {log_loss(test_labels, test_pred)}    auc: {roc_auc_score(test_labels, test_pred)}')