"""工具函数，按需添加"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer
from scipy.interpolate import lagrange
from imblearn.over_sampling import SMOTE, ADASYN
import os
import pickle
from sklearn.preprocessing import robust_scale


def fill_missing_for_train(data, strategy='most_frequent'):
    my_imputer = Imputer(missing_values=-99, strategy=strategy, axis=1)
    for each_group in ['group_1', 'group_2', 'group_3']:
        for each_y in [0, 1]:
            tmp_idx = data[(data['cust_group'] == each_group) & (data['y'] == each_y)].index
            tmp_data = data[(data['cust_group'] == each_group) & (data['y'] == each_y)].iloc[:, 2:]
            tmp_data = my_imputer.fit_transform(tmp_data)

            data.ix[tmp_idx, 2:] = tmp_data
    return data

def fill_missing_for_test(data, strategy='most_frequent'):
    imp = Imputer(missing_values=-99, strategy=strategy, axis=1)
    for each_group in ['group_1', 'group_2', 'group_3']:
        tmp_idx = data[(data['cust_group'] == each_group)].index
        tmp_data = data[(data['cust_group'] == each_group)].iloc[:, 2:]
        tmp_data = imp.fit_transform(tmp_data)
        data.ix[tmp_idx, 2:] = tmp_data

    return data


def load_data(data_path='../input/train_xy.csv', flag='train', test_size=0.3, random_state=123):
    """对数据data_path进行分割并返回，若返回全部令test_size=0.0"""
    # 读取数据
    assert flag in ('train, test'), print('flag must in ("train", "test")')

    if flag == 'train':
        data = pd.read_csv(data_path, index_col=0)
        data.drop(['x_110', 'x_112'], axis=1, inplace=True)  # x_110 x_112 在训练集和验证中全为空
        data = fill_missing_for_train(data)
        X = data.iloc[:, 2:]
        y = data.loc[:, 'y']
        cust_group = data[['cust_group']]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_val, y_train, y_val, cust_group
    else: # test data
        data = pd.read_csv(data_path)
        data.drop(['x_110', 'x_112'], axis=1, inplace=True)  # x_110 x_112 在训练集和验证中全为空
        data = fill_missing_for_test(data)
        # print(data.columns)
        return data[['cust_id']].values, data.iloc[:, 2:]


def load_data_resampled(data_path='../input/train_xy.csv', test_size=0.3, random_state=123):
    """对数据data_path进行分割并返回，若返回全部令test_size=0.0"""
    # 读取数据
    data = pd.read_csv(data_path, index_col=0)
    data.drop(['x_110', 'x_112'], axis=1, inplace=True) # x_110 x_112 在训练集和验证中全为空
    res_dtypes = data.iloc[:,2:].dtypes
    data = fill_missing_for_train(data)

    X = data.iloc[:, 2:]
    Y = data['y']

    # 分组的上采样
    data_resampled = []
    for each_group in ['group_1', 'group_2', 'group_3']:
        X_grouped = data[data['cust_group'] == each_group].iloc[:,2:]
        y_grouped = data[data['cust_group'] == each_group].loc[:,'y']

        X_resampled, y_resampled = SMOTE().fit_sample(X_grouped, y_grouped)
        # print(each_group)
        # print('X_resampled: ', X_resampled)
        # print('y_resampled: ', y_resampled)

        tmp_group = np.array([each_group] * X_resampled.shape[0]).reshape(-1, 1)
        tmp = np.concatenate([tmp_group, y_resampled[:, np.newaxis], X_resampled], axis=1)
        data_resampled.append(tmp)

    ## TODO: 这一坨东西写的太复杂了！！！
    data_resampled = np.concatenate(data_resampled, axis=0)
    data_resampled = pd.DataFrame(data_resampled, columns=data.columns)

    data_resampled.index = range(data_resampled.shape[0])

    # 分割
    X = data_resampled.iloc[:, 2:]
    for i in X.columns:
        X[[i]] = X[[i]].astype(float)
    for i, j in zip(X.columns, res_dtypes):
        X[[i]] = X[[i]].astype(j)
    y = data_resampled.loc[:, 'y']
    y = y.astype(np.int)

    cust_group = data_resampled[['cust_group']]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_val, y_train, y_val, cust_group


def auc_score(y_true, y_preds, cust_group):
    """注意这里的cust_group是所有数据的分组信息， y_true和y_preds只包含少数测试数据"""
    groups = cust_group.loc[y_true.index]['cust_group']
    tmp = pd.DataFrame({'cust_group': groups, 'y_true': y_true, 'y_pred': y_preds})

    auc_list = []
    for g in ['group_1', 'group_2', 'group_3']:
        group = tmp[tmp['cust_group'] == g]
        tmp_auc = roc_auc_score(group['y_true'], group['y_pred'])
        auc_list.append(tmp_auc)

    auc = np.average(auc_list, weights=[0.3, 0.3, 0.4])
    return auc


def evaluate(y_train, y_train_pred, y_val, y_val_pred, cust_group):
    print('模型评估...')
    auc_train = auc_score(y_train, y_train_pred, cust_group)
    auc_test = auc_score(y_val, y_val_pred, cust_group)
    print('训练auc: %.6f ------ 测试auc: %.6f' %(auc_train, auc_test))


def make_submission(file_name, preds, cust_id):
    assert type(file_name) == str, print('filename必须是str')
    res = pd.DataFrame({'cust_id': cust_id, 'pred_prob': preds})
    res.to_csv(f'../output/{file_name}', index=None)

def load_model_if_exist(pkl_fp):
    model = None
    if os.path.exists(pkl_fp):
        with open(pkl_fp, 'rb') as f:
            model = pickle.load(f)
    return model

def save_model(model, save_fp):
    if not os.path.exists(save_fp):
        print('Saving model...')
        with open(save_fp, 'wb') as f:
            pickle(model, f)
    else:
        print(f'model {save_fp} always exist')