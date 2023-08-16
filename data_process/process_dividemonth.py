import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict

def normalize(mx:np.ndarray):  # 归一化
    mx=mx.astype(np.float16)
    shape=mx.shape
    # mx=mx.reshape((-1,shape[-1]))
    for k in range(mx.shape[-1]):
        mx[:,k]=(mx[:,k]-np.mean(mx[:,k]))/np.std(mx[:,k])
    mx=mx.reshape(shape)
    return mx

data_path='/home/m21_huangzijun/pythonprojs/sichuan'
voc=pd.read_csv(data_path+'/data/train_voc.csv')
user=pd.read_csv(data_path+'/data/train_user.csv')
user_voc=user.merge(right=voc,on='phone_no_m',how='left')
user_voc['start_datetime']=pd.to_datetime(user_voc['start_datetime'])
voc['start_datetime']=pd.to_datetime(voc['start_datetime'])

ids=user['phone_no_m'].tolist()

months=[g for _,g in voc.groupby(pd.Grouper(key='start_datetime',freq='M'))]

fea_dict=defaultdict(list)





for month in months:
    tmp_ids=ids.copy()
    tmp_ids=set(tmp_ids)
    month['call_time']=month.groupby('phone_no_m')['phone_no_m'].transform('count') #打过几次电话
    month['inoutdegree']=month.groupby(['phone_no_m','calltype_id'])['phone_no_m'].transform('nunique')
    in_voc=month.loc[month['calltype_id']==2][['phone_no_m','opposite_no_m']]
    out_voc=month.loc[month['calltype_id']==1][['phone_no_m','opposite_no_m']]
    in_voc['indegree']=in_voc.groupby('phone_no_m')['opposite_no_m'].transform('nunique')
    out_voc['outdegree']=out_voc.groupby('phone_no_m')['opposite_no_m'].transform('nunique')
    in_voc=in_voc[['phone_no_m','indegree']].drop_duplicates()
    out_voc=out_voc[['phone_no_m','outdegree']].drop_duplicates()
    month=month.merge(right=in_voc,on='phone_no_m',how='left')
    month=month.merge(right=out_voc,on='phone_no_m',how='left')
    month.rename(columns={'indegree_x':'indegree'},inplace=True)
    month['indegree']=month['indegree'].fillna(0)
    month['outdegree']=month['outdegree'].fillna(0)
    month['neigh_degree']=month.groupby(['opposite_no_m'])['phone_no_m'].transform('nunique')
    month['mean_duration']=month.groupby('phone_no_m')['call_dur'].transform(np.mean)
    month['var_duration']=month.groupby('phone_no_m')['call_dur'].transform(np.var)
    month['each_person']=month.groupby(['phone_no_m','opposite_no_m'])['opposite_no_m'].transform('count')
    month['energy_dispersion']=month['each_person']/month['call_time']
    del month['each_person']
    persons=[g for _,g in month.groupby('phone_no_m')]
    for person in persons:
        id=person['phone_no_m'].iloc[0]
        indegree=person['indegree'].iloc[0]
        outdegree=person['outdegree'].iloc[0]
        neigh_degree=person['neigh_degree'].iloc[0]
        mean_duration=person['mean_duration'].iloc[0]
        var_duration=person['var_duration'].iloc[0]
        ed=person['energy_dispersion'].iloc[0]
        fea_dict[id].append([indegree,outdegree,neigh_degree,mean_duration,var_duration,ed])
        tmp_ids.remove(id)
    for id in list(tmp_ids):
        fea_dict[id].append([0 for i in range(6)])
    
for i in range(8):
    feas=[]
    ys=[]
    id_label=zip(user['phone_no_m'].tolist(),user['label'].tolist())
    for id,label in id_label:
        fea=fea_dict[id][i]
        feas.append(fea)
        ys.append(label)
    feas=np.array(feas)
    feas=normalize(feas)
    feas=np.nan_to_num(feas)
    ys=np.array(ys)
    np.save('data_process/static_{}'.format(i+1),feas)
    np.save('data_process/static_y_{}'.format(i+1),ys)


pass