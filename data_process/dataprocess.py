# %%
import torch
import numpy as np
import math,csv,gzip,os,io,copy,threading,re
import pandas as pd
import copy
import time as timee
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor,as_completed
from multiprocessing import Pipe,Pool
from scipy import stats
from pathlib import  Path
import networkx as nx

# %%
@contextmanager
def timer(title):
    t0 = timee.time()
    yield
    print("{} - done in {:.0f}s".format(title, timee.time() - t0))






# %%
def get_person_months(name_file,file):
    with timer('read file'):
        df_name=pd.read_csv(name_file)
        df_name_voc=pd.read_csv(file)
        
        df_name_voc['start_datetime']=pd.to_datetime(df_name_voc['start_datetime'])
        #df_voc['day']=df_voc['start_datetime'].dt.day
        df_name_voc['hour']=df_name_voc['start_datetime'].dt.hour
        
    print('read file finished')
    #df_name.set_index('phone_no_m')
    
    pattern='.*arpu.*'
    columns=df_name_voc.columns.copy()
    for column in columns:
        if re.match(pattern=pattern,string=column):
            del(df_name_voc[column])

    
    df_name_voc=df_name_voc.dropna()         #删掉所有空缺值的行
    
    

    #有多少个人
    people=set(df_name_voc['phone_no_m'])
    #people=people[:6]
    
    #按周划分，每个元素是一个月的dataframe
    month_group=[g for _,g in df_name_voc.groupby(pd.Grouper(key='start_datetime',freq='W'))]    
       
    print('divide month finished')
    
    
    '''month group is a list of dataframes'''
    #month_group=month_group[:1]
    #按人划分    
    with timer('group by person'):
        for idx,month in enumerate(month_group):
            month_group[idx]=[g for _,g in month.groupby('phone_no_m')]   #将month_group的每个元素转换成人在本月的列表
    print('divide person finished')
    
    
    
    person_months={}
    with timer('month person to person month'):
        for person in people:
            person_months[person]=list()    #这个人所有月的dataframe
        index=0
        for persons in month_group:    #对所有月
            # if index>=9:
            #     break
            # index+=1
            temp_people=people.copy()
            for person in persons:   #对该月的所有人
                person=person.sort_values(by='start_datetime')
                person=person.reset_index(drop=True)
                who=person.loc[0,'phone_no_m']
                person_months[who].append(person.to_dict(orient='list'))
                temp_people.remove(who)
            for person in temp_people:   #对不在该月的所有人
                person_months[person].append(pd.DataFrame(columns=df_name_voc.columns).to_dict(orient='list'))
                
    person_monthes=[]
    labels=[]
    df_name.set_index(keys='phone_no_m',inplace=True)
    for p,ms in person_months.items(): #对所有人
        person_monthes.append(ms) #ms是一个dict，是这个人在该时间片的数据
        labels.append(df_name.loc[p,'label'])
    return person_monthes,labels

def get_person_info_and_feature(name_file,file,place_code):
    with timer('read file'):
        df_name=pd.read_csv(name_file)     # ID information file
        df_name_voc=pd.read_csv(file)    #ID and voc information file
        
        df_name_voc['start_datetime']=pd.to_datetime(df_name_voc['start_datetime'])
        
        df_name_voc['hour']=df_name_voc['start_datetime'].dt.hour
    
    # drop the arpu columns
    pattern='.*arpu.*'
    columns=df_name_voc.columns.copy()
    for column in columns:
        if re.match(pattern=pattern,string=column):
            del(df_name_voc[column])

    df_name_voc=df_name_voc.dropna()         
    
    #people set
    people=set(df_name_voc['phone_no_m'])
    
    
    
    # group all voc by week
    month_group=[g for _,g in df_name_voc.groupby(pd.Grouper(key='start_datetime',freq='W'))]    
       
    print('divide month finished')
    
    
    # group all voc by person for each week 
    with timer('group by person'):
        for idx,month in enumerate(month_group):
            month_group[idx]=[g for _,g in month.groupby('phone_no_m')]   #将month_group的每个元素转换成人在本月的列表
    print('divide person finished')
    
    
    
    person_months={}
    with timer('month person to person month'):
        for person in people:
            person_months[person]=list()    #这个人所有月的dataframe
        index=0
        for persons in month_group:    #所有周
            # if index>=9:
            #     break
            # index+=1
            temp_people=people.copy()
            for person in persons:   #这周的所有人，这个人的dataframe按时间排序
                person=person.sort_values(by='start_datetime')
                person=person.reset_index(drop=True)
                who=person.loc[0,'phone_no_m']
                person_months[who].append(person.to_dict(orient='list'))
                temp_people.remove(who)
            for person in temp_people:   #不在这周的人，插入一个空dataframe
                person_months[person].append(pd.DataFrame(columns=df_name_voc.columns).to_dict(orient='list'))
                
    person_monthes=[]
    person_id_features=[]
    labels=[]
    df_name.set_index(keys='phone_no_m',inplace=True)
    for p,ms in person_months.items(): #对所有人
        person_monthes.append(ms) # ms is a dict
        city=str(df_name.loc[p,'city_name'])
        county=str(df_name.loc[p,'county_name'])
        place=' '.join([city,county])
        idcard_cnt=df_name.loc[p,'idcard_cnt']
        if place not in place_code:
            place_code[place]=len(place_code)
        person_id_features.append([place_code[place],idcard_cnt])
        
        
        labels.append(df_name.loc[p,'label'])

    return person_monthes,person_id_features,labels,place_code   # perosn_monthes is a list of all persons, a person is a list of all week dataframe



def get_person_info_and_feature_test(name_file,voc_file,place_code):
    with timer('read file'):
        df_name=pd.read_csv(name_file)     #用户信息
        df_voc=pd.read_csv(voc_file)    #通话记录
        df_name_voc=pd.merge(left=df_name,right=df_voc,on='phone_no_m')
        df_name_voc['start_datetime']=pd.to_datetime(df_name_voc['start_datetime'])
        #df_voc['day']=df_voc['start_datetime'].dt.day
        df_name_voc['hour']=df_name_voc['start_datetime'].dt.hour
        
    print('read file finished')
    #df_name.set_index('phone_no_m')
    

    #去除月消费记录
    pattern='.*arpu.*'
    columns=df_name_voc.columns.copy()
    for column in columns:
        if re.match(pattern=pattern,string=column):
            del(df_name_voc[column])

    
    df_name_voc=df_name_voc.dropna()         #删掉所有空缺值的行
    
    

    #有多少个人
    people=set(df_name_voc['phone_no_m'])
    #people=list(people) #转换成列表，使其有序
    
    
    #按周划分，每个元素是一个月的dataframe
    month_group=[g for _,g in df_name_voc.groupby(pd.Grouper(key='start_datetime',freq='W'))]    
       
    print('divide month finished')
    
    
    '''month group is a list of dataframes'''
    #month_group=month_group[:1]
    #按人划分    
    with timer('group by person'):
        for idx,month in enumerate(month_group):
            month_group[idx]=[g for _,g in month.groupby('phone_no_m')]   #将month_group的每个元素转换成人在本月的列表
    print('divide person finished')
    
    
    
    person_months={}
    with timer('month person to person month'):
        for person in people:
            person_months[person]=list()    #这个人所有月的dataframe
        index=0
        for persons in month_group:    #对所有月
            # if index>=9:
            #     break
            # index+=1
            temp_people=people.copy()
            for person in persons:   #对该月的所有人
                person=person.sort_values(by='start_datetime')
                person=person.reset_index(drop=True)
                who=person.loc[0,'phone_no_m']
                person_months[who].append(person.to_dict(orient='list'))
                temp_people.remove(who)
            for person in temp_people:   #对不在该月的所有人
                person_months[person].append(pd.DataFrame(columns=df_name_voc.columns).to_dict(orient='list'))
                
    person_monthes=[]
    person_id_features=[]
    labels=[]
    df_name.set_index(keys='phone_no_m',inplace=True)
    for p,ms in person_months.items(): #对所有人
        person_monthes.append(ms) #ms是一个dict，是这个人在该时间片的数据
        
        city=str(df_name.loc[p,'city_name'])
        county=str(df_name.loc[p,'county_name'])
        place=' '.join([city,county])
        idcard_cnt=df_name.loc[p,'idcard_cnt']
        if place not in place_code:
            place_code[place]=len(place_code)
        person_id_features.append([place_code[place],idcard_cnt])
        
        
        labels.append(df_name.loc[p,'label'])

    return person_monthes,person_id_features,labels,place_code

# %%

from features import time_gap2,time2,connector_duplicate2,area_change2,recall_rate2,energy_dispersion2,mean_voc_time2,n_unique_persons2
# %%
import features as features


# %%
def normalize(mx,is_trainset): #归一化
    max_list=[]
    min_list=[]
    num_feats=mx.shape[-1]
    print(num_feats)
    if is_trainset:  #如果是训练集
        for i in range(num_feats):
            max_list.append(np.max(mx[:,:,i].flatten()))
            min_list.append(np.min(mx[:,:,i].flatten()))
        np.save(file=f'data_after/mymodel/max_value.npy',arr=max_list)
        np.save(file=f'data_after/mymodel/min_value.npy',arr=min_list)
    else:
        max_list=np.load(file=f'data_after/mymodel/max_value.npy')
        min_list=np.load(file=f'data_after/mymodel/min_value.npy')
    
    for i in range(num_feats):
        mx[:,:,i]=(mx[:,:,i]-min_list[i])/(max_list[i]-min_list[i])
    return mx


    


# %%
def main():
    person_weeks,labels=get_person_months('data/0527/train/train_user.csv', 'data/0527/train/train_voc.csv')
    np_feature=[]

    '''
    split persons labels
    '''
    '''cancel comment'''
    with timer('feature '):
        for weeks in person_weeks:
            #weeks=weeks.sort_values()
            weeks_feature=[]
            index=0
            #with timer(f'the {index} person'):
            for idx,week in enumerate(weeks):
                #week
                mean=mean_voc_time2(week)
                var=features.var2(week)
                nup=n_unique_persons2(week)
                ed_mean=energy_dispersion2(week)
                rate=recall_rate2(week)
                change=area_change2(week)
                duplication=0.0
                
                if idx>0:
                    duplication=connector_duplicate2(week,weeks[idx-1])
                call_time=time2(week)
                gap=time_gap2(week)
                #index+=1
                weeks_feature.append([mean,var,nup,ed_mean,rate,change,duplication,call_time,gap])
            np_feature.append(weeks_feature)    
    
    np_feature2=[]
    labels2=[]
    index=0
    for weeks in np_feature:
        for i in range(0,len(weeks)-4):
            np_feature2.append(weeks[i:i+5])
            labels2.append(labels[index])
        index+=1
    np_array=np.array(np_feature2)
    '''cancel comment'''
    tensor_feature=torch.tensor(np_array)
    tensor_feature=normalize(tensor_feature)
    torch.save(tensor_feature, 'feature9.pt')
    tensor_label=torch.tensor(labels2)
    torch.save(tensor_label, 'label9.pt')


# %%

def main_static(file_no):
    person_weeks,labels=get_person_months('data/0527/train/train_user.csv', f'data/test{file_no}.csv')
    feature_all=[]
    with timer('feature '):
        for person in person_weeks:
            #weeks=weeks.sort_values()
            weeks_feature=[]
            index=0
            #with timer(f'the {index} person'):
            mean=[]
            nup=[]
            eds=[]
            rates=[]
            areas=[]
            dups=list()
            call_times=list()
            gaps=list()
            for idx,week in enumerate(person):
                #week
                mean.extend(week['call_dur'])
                nup.extend(week['opposite_no_m'])
                ed_mean=energy_dispersion2(week)
                eds.append(ed_mean)
                rate=recall_rate2(week)
                rates.append(rate)
                areass=features.areas(week)
                areas.extend(areass)
                duplication=0.0
                
                if idx>0:
                    duplication=connector_duplicate2(week,person[idx-1])
                    dups.append(duplication)
                call_times.extend(week['hour'])
                gap=features.time_gap_static(week)
                gaps.extend(gap)
                #index+=1
                #weeks_feature.append([mean,nup,ed_mean,rate,change,duplication,call_time,gap])
            #np_feature.append(weeks_feature)    
            mean=np.mean(mean)
            var=np.var(mean)
            nup=len(set(nup))
            eds=np.mean(eds)
            rates=np.mean(rates)
            areas=len(set(areas))
            dups=np.mean(dups)
            call_times=stats.mode(call_times)[0][0]
            gaps=np.mean(gaps)
            feature_all.append([mean,var,nup,eds,rates,areas,dups,call_times,gaps])
    
    np_feats=np.array(feature_all)
    np_labels=np.array(labels)
    np.save(file=f'data_after/baselinemodel/test_x_{file_no}.npy',arr=np_feats)
    np.save(file=f'data_after/baselinemodel/test_y_{file_no}.npy',arr=np_labels)
    '''cancel comment'''

    pass


# %%
def main_allweeks():
    data_path='~/pythonprojs/sichuan/'
    place_code=dict()
    filepath=Path(data_path+'data_after/mymodel/place_code.npy')
    if filepath.is_file():
        place_code=np.load('data_after/mymodel/place_code.npy',allow_pickle=True).item()
    # 获取用户每周的通话记录dataframe
    # person_weeks,person_feature,labels,place_code=get_person_info_and_feature(data_path+'data/0527/train/train_user.csv', data_path+f'data/train.csv',place_code=place_code)
    # person_weeks_t,person_feature_t,labels_t,place_code=get_person_info_and_feature(data_path+'data/0527/train/train_user.csv',data_path+f'data/test.csv',place_code=place_code)
    
    
    train_slice_feature=dict()
    test_slice_feature=dict()
    
    train_id_feature=dict()
    test_id_feature=dict()
    
    np_feature=[]

    # 建立每周的网络
    all_user=pd.read_csv('data/0527/train/train_user.csv')
    # train_user=all_user.loc[all_user['phone_no_m'] in trai]
    
    train_voc=pd.read_csv(data_path+f'data/train.csv')
    test_voc=pd.read_csv(data_path+f'data/test.csv')
    train_voc['start_datetime']=pd.to_datetime(train_voc['start_datetime'])
    test_voc['start_datetime']=pd.to_datetime(test_voc['start_datetime'])
    train_voc['hour']=train_voc['start_datetime'].dt.hour
    test_voc['hour']=test_voc['start_datetime'].dt.hour
    train_voc=train_voc.merge(all_user[['phone_no_m','label']],how='left',on='phone_no_m')
    test_voc=test_voc.merge(all_user[['phone_no_m','label']],how='left',on='phone_no_m')
    train_user=train_voc['phone_no_m'].tolist()
    test_user=test_voc['phone_no_m'].tolist()
    train_vocs=[g for _,g in train_voc.groupby(pd.Grouper(key='start_datetime',freq='W'))]
    test_vocs=[g for _,g in test_voc.groupby(pd.Grouper(key='start_datetime',freq='W'))]
    train_nets=[nx.DiGraph() for i in range(len(train_vocs))]#训练集网络
    test_nets=[nx.DiGraph() for i in range(len(test_vocs))]  #测试集网络
    # train_nets2=[nx.Graph() for i in range(len(train_vocs))]
    # test_nets2=[nx.Graph() for i in range(len(test_vocs))]
    train_neighbor={}#训练集邻居
    test_neighbor={}
    all_user.set_index(keys='phone_no_m',drop=False,inplace=True)
    for idx,net in enumerate(train_nets):  #每个时间片的图
        net.add_nodes_from(train_user)
        # net2=train_nets2[idx]
        voc=train_vocs[idx]
        for row in voc.itertuples():   #每次通话
            calltype_id=getattr(row,'calltype_id')
            source=getattr(row,'phone_no_m')
            target=getattr(row,'opposite_no_m')
            net.add_edge(source,target) if calltype_id==1 else net.add_edge(target,source)
            if source not in train_neighbor:train_neighbor[source]=[]
            train_neighbor[source].append(target)
    
    for idx,net in enumerate(test_nets):
        net.add_nodes_from(test_user)
        voc=test_vocs[idx]
        for row in voc.itertuples():
            calltype_id=getattr(row,'calltype_id')
            source=getattr(row,'phone_no_m')
            target=getattr(row,'opposite_no_m')
            net.add_edge(source,target) if calltype_id==1 else net.add_edge(target,source)
            if source not in test_neighbor:test_neighbor[source]=[]
            test_neighbor[source].append(target)
    
    
    # 平均通话时长
    # 通话时长方差
    # 精力分散度
    
    train_voc['mean_dur']=train_voc.groupby(['phone_no_m',pd.Grouper(key='start_datetime',freq='W')])['call_dur'].transform('mean') #通话时长平均
    train_voc['var_dur']=train_voc.groupby(['phone_no_m',pd.Grouper(key='start_datetime',freq='W')])['call_dur'].transform('var') # 通话时长方差
    train_voc['sum_call_times']=train_voc.groupby(['phone_no_m'])['phone_no_m'].transform('count')  #
    train_voc['every_one_calltimes']=train_voc.groupby(['phone_no_m','opposite_no_m'])['phone_no_m'].transform('count') 
    train_voc['energy_dispersion']=train_voc['every_one_calltimes']/train_vocs['sum_call_times']  # 精力分散度
    del train_voc['sum_call_times']
    del train_voc['every_one_calltimes']
    
    test_voc['mean_dur']=test_voc.groupby(['phone_no_m',pd.Grouper(key='start_datetime',freq='W')])['call_dur'].transform('mean') #通话时长平均
    test_voc['var_dur']=test_voc.groupby(['phone_no_m',pd.Grouper(key='start_datetime',freq='W')])['call_dur'].transform('var') # 通话时长方差
    test_voc['sum_call_times']=test_voc.groupby(['phone_no_m'])['phone_no_m'].transform('count')  #
    test_voc['every_one_calltimes']=test_voc.groupby(['phone_no_m','opposite_no_m'])['phone_no_m'].transform('count') 
    test_voc['energy_dispersion']=test_voc['every_one_calltimes']/test_vocs['sum_call_times']  # 精力分散度
    del test_voc['sum_call_times']
    del test_voc['every_one_calltimes']

    # 需要算 出度、入度、邻居平均度、聚类系数、回拨率、联系人重复率、通话时刻分布
    # id 属性：
    
    train_user=set(train_user)
    test_user=set(test_user)
    tmp_train_user=train_user.copy()
    tmp_test_user=test_user.copy()
    with timer('train feature '):
        for idx,slice in enumerate(train_vocs): # each slice 
            ps=[g for _,g in slice.groupby('phone_no_m')]
            net=train_nets[idx]
            net:nx.DiGraph
            for p in ps: # each person in this slice
                id=p['phone_no_m'].iloc[0]
                if idx==0: repeat_rate=0
                else:
                    pre_week=train_vocs[idx-1]
                    pre_week=pre_week.loc[pre_week['phone_no_m']==id]
                    repeat_rate=connector_duplicate2(pre_week,p)
                indegree=net.in_degree[id]
                outdegree=net.out_degree[id]
                neighbor_degree=list()
                for neigh in train_neighbor[id]:
                    neighbor_degree.append(net.degree[neigh])
                neighbor_degree=np.mean(neighbor_degree)
                coefficient=nx.clustering(net,id)
                recall_rate=features.recall_rate2(p)
                time_dis=[0 for i in range(24)]
                for time,count in p['hour'].value_counts():
                    time_dis[time]=count
                if idx==0:train_slice_feature[id]=list()
                mean_dur=p['mean_dur'].iloc[0]
                var_dur=p['var_dur'].iloc[0]
                
                train_slice_feature[id].append([indegree,outdegree,neighbor_degree,coefficient,recall_rate,repeat_rate,mean_dur,var_dur]+time_dis)
                tmp_train_user.pop(id)
            for id in list(tmp_train_user): # persons not in this slice
                train_slice_feature[id]=[0 for i in range(32)]
            tmp_train_user
        static_voc=train_voc[['phone_no_m','energy_dispersion']].drop_duplicates(subset='phone_no_m').set_index(keys='phone_no_m')
        for id in list(train_user):
            idcard_cnt=all_user.loc[id,'idcard_cnt']
            ed=static_voc.loc['phone_no_m','energy_dispersion']
            train_id_feature[id]=[idcard_cnt,ed]

    for id,v in train_slice_feature.items():
        np_feature.append(v)
        np_id_feature.append(train_id_feature[id])
        labels.append(all_user.loc['phone_no_m','label'])

    np_feature=np.array(np_feature)
    np_id_feature=np.array(person_feature)
    labels=np.array(labels)
    #train_n=int(len(np_feature)/4)*3
    np_feature[:,[1,2,3,5,6]]=normalize(np_feature[:,[1,2,3,5,6]],is_trainset=True)
    np.save(arr=np_feature,file=data_path+f'data_after/mymodel3/train_x.npy')
    np.save(arr=np_id_feature,file=data_path+'data_after/mymodel3/train_x_id.npy')
    np.save(arr=labels,file=data_path+f'data_after/mymodel3/train_y.npy')

    np_feature=[]    
    
    train_voc['mean_dur']=train_voc.groupby(['phone_no_m',pd.Grouper(key='start_datetime',freq='W')])['call_dur'].transform('mean') #通话时长平均
    train_voc['var_dur']=train_voc.groupby(['phone_no_m',pd.Grouper(key='start_datetime',freq='W')])['call_dur'].transform('var') # 通话时长方差
    train_voc['sum_call_times']=train_voc.groupby(['phone_no_m'])['phone_no_m'].transform('count')  #
    train_voc['every_one_calltimes']=train_voc.groupby(['phone_no_m','opposite_no_m'])['phone_no_m'].transform('count') 
    train_voc['energy_dispersion']=train_voc['every_one_calltimes']/train_vocs['sum_call_times']  # 精力分散度
    del train_voc['sum_call_times']
    del train_voc['every_one_calltimes']
    
    pass



def main_allweeks_testastest():
    place_code=dict()
    filepath=Path('data_after/mymodel/place_code.npy')
    if filepath.is_file():
        place_code=np.load('data_after/mymodel/place_code.npy',allow_pickle=True).item()
    person_weeks,person_feature,labels,place_code=get_person_info_and_feature_test('data/0527/test/test_user1.csv', f'data/0527/test/test_voc.csv',place_code=place_code)

    np_feature=[]

    with timer('train feature '):
        for weeks in person_weeks:
            #weeks=weeks.sort_values()
            weeks_feature=[]
            index=0
            #with timer(f'the {index} person'):
            for idx,week in enumerate(weeks):
                #week
                mean=mean_voc_time2(week)
                var=features.var2(week)
                nup=n_unique_persons2(week)
                ed_mean=energy_dispersion2(week)
                rate=recall_rate2(week)
                change=area_change2(week)
                duplication=0.0
                
                if idx>0:
                    duplication=connector_duplicate2(week,weeks[idx-1])
                call_time=time2(week)
                gap=time_gap2(week)
                #index+=1
                weeks_feature.append([mean,var,nup,ed_mean,rate,change,duplication,call_time,gap])
            np_feature.append(weeks_feature)    
    np_feature=np.array(np_feature)
    np_id_feature=np.array(person_feature)
    labels=np.array(labels)


    np.save(arr=normalize(np_feature,is_trainset=True),file=f'data_after/mymodel/true_test_x.npy')
    np.save(arr=np_id_feature,file='data_after/mymodel/true_test_x_id.npy')
    np.save(arr=labels,file=f'data_after/mymodel/true_test_y.npy')
    pass
# %%
if __name__=="__main__":
    main_allweeks()
    



# %%
