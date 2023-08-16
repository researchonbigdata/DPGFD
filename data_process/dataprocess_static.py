
import numpy as np
import copy

import re
import pandas as pd
import copy
import time as timee
from contextlib import contextmanager


from scipy import stats
import networkx as nx
from collections import defaultdict


@contextmanager
def timer(title):
    t0 = timee.time()
    yield
    print("{} - done in {:.0f}s".format(title, timee.time() - t0))


def normalize(mx: np.ndarray):  # 归一化
    shape = mx.shape

    for k in range(mx.shape[-1]):
        mx[:, k] = (mx[:, k]-np.mean(mx[:, k]))/np.std(mx[:, k])

    return mx


def behavior_feature_extract():
    train_ids = np.load(
        '/home/m21_huangzijun/pythonprojs/sichuan/data_after/mymodel5_2/train_ids.npy')
    test_ids = np.load(
        '/home/m21_huangzijun/pythonprojs/sichuan/data_after/mymodel5_2/test_ids.npy')

    ids = np.concatenate([train_ids, test_ids])

    user = pd.read_csv(
        '/home/m21_huangzijun/pythonprojs/sichuan/data/0527/train/train_user.csv')
    voc = pd.read_csv(
        '/home/m21_huangzijun/pythonprojs/sichuan/data/0527/train/train_voc.csv')

    voc['start_datetime'] = pd.to_datetime(voc['start_datetime'])
    voc['hour'] = voc['start_datetime'].dt.hour

    # 平均通话时长
    voc['mean_dur'] = voc.groupby('phone_no_m')[
        'call_dur'].transform(np.nanmean)
    # 通话时长方差
    voc['var_dur'] = voc.groupby('phone_no_m')[
        'call_dur'].transform(np.nanvar)
    # 精力分散度
    voc['sum_call_times'] = voc.groupby(
        ['phone_no_m'])['phone_no_m'].transform('count')
    voc['every_one_calltimes'] = voc.groupby(['phone_no_m', 'opposite_no_m'])['phone_no_m'].transform(
        'count')
    voc['energy_dispersion'] = voc['every_one_calltimes'] / voc['sum_call_times']
    del voc['sum_call_times']
    del voc['every_one_calltimes']

    net = nx.Graph()
    net.add_nodes_from(ids)
    for row in voc.to_dict(orient='records'):
        calltype_id = row['calltype_id']
        source = row['phone_no_m']
        target = row['opposite_no_m']
        if calltype_id == 1:
            net.add_edge(source, target, weight=1)

        elif calltype_id == 2:
            net.add_edge(source, target, weight=-1)

    # 联系人重复率无法计算

    users_voc = [g for _, g in voc.groupby('phone_no_m')]

    person_feature = dict()

    tmp_ids = set(ids.copy())
    for user_voc in users_voc:
        id = user_voc['phone_no_m'].iloc[0]

        # 入度/出度
        indegree = 0
        outdegree = 0
        for k, v in net[id].items():
            if v['weight'] == 1:
                outdegree += 1
            elif v['weight'] == -1:
                indegree += 1
        # 邻居平均度
        neighbor_degree = list()
        for neigh in net.neighbors(id):
            neighbor_degree.append(net.degree(id))
        neighbor_degree = np.mean(neighbor_degree)

        # 聚类系数
        coefficient = nx.clustering(net, id)

        # 通话时间分布
        time_dis = [0 for i in range(24)]
        for time, count in user_voc['hour'].value_counts(normalize=True).to_dict().items():
            time_dis[time] = count

        # 平均通话时长
        mean_dur = user_voc['mean_dur'].iloc[0]
        # 通话时长方差
        var_dur = user_voc['var_dur'].iloc[0]
        # 能量分布
        ed = user_voc['energy_dispersion'].iloc[0]

        person_feature[id] = [indegree, outdegree,
                              neighbor_degree, coefficient, mean_dur, var_dur, ed]+time_dis
        tmp_ids.remove(id)

    for id in list(tmp_ids):
        person_feature[id] = [0 for i in range(31)]

    person_feature_inorder = []
    for id in ids:
        person_feature_inorder.append(person_feature[id])

    person_feature_inorder = np.array(person_feature_inorder).astype(np.float)
    person_feature_inorder = normalize(person_feature_inorder)
    person_feature_inorder = np.nan_to_num(person_feature_inorder)
    np.save('/home/m21_huangzijun/pythonprojs/sichuan/data_after/static_feature/static_feature.npy',
            person_feature_inorder)


def test():
    arr1 = np.load(
        '/home/m21_huangzijun/pythonprojs/sichuan/data_after/static_feature/static_feature.npy')
    print(arr1.shape)


if __name__ == "__main__":
    test()
