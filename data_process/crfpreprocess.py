import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
import math
import torch.nn.functional as F
import sys
import scipy as sp
from scipy import stats
from scipy.io import loadmat
import time as timee
from contextlib import contextmanager
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

@contextmanager
def timer(title):
    t0 = timee.time()
    yield
    print("{} - done in {:.0f}s".format(title, timee.time() - t0))

def main(call_times,k_common_connector):
    data_path= '/home/m21_huangzijun/pythonprojs/sichuan/'
    id_file = data_path + 'data/0527/train/train_user.csv'


    #分出哪些人是知道，哪些人是不知道
    voc_file = data_path + 'data/train.csv'

    voc_file2=data_path+'data/test.csv'
    voc_file=pd.read_csv(voc_file)
    voc_file2=pd.read_csv(voc_file2)
    train_user=voc_file['phone_no_m'].unique().tolist()
    test_user=voc_file2['phone_no_m'].unique().tolist()

    voc_file = data_path + 'data/0527/train/train_voc.csv'

    #地区编码
    place_code_dict_path=data_path+'data_after/mymodel/place_code.npy'
    place_code_dict_path=Path(place_code_dict_path)
    place_code_dict = dict()    #地区dict
    # if place_code_dict_path.is_file():
    #     place_code_dict = np.load(data_path + 'data_after/mymodel/place_code.npy').item()
    #imei编码
    imei_dict_path=data_path+'data_after/mymodel/imei_code.npy'
    imei_dict_path=Path(imei_dict_path)
    imei_dict=dict()    #imei dict
    # if imei_dict_path.is_file():
    #     imei_dict=np.load(data_path+'data_after/mymodel/imei_code.npy').item()
    #读取数据集
    name=pd.read_csv(id_file)
    name['city_name'].fillna(value=' ')
    name['county_name'].fillna(value=' ')
    name['idcard_cnt'].fillna(0)
    voc=pd.read_csv(voc_file)
    voc['opposite_no_m'].fillna(' ')
    voc['calltype_id'].fillna(1)
    voc['start_datetime'].fillna('1/1/1 00:00')
    voc['call_dur'].fillna(0)
    voc['city_name'].fillna(' ')
    voc['county_name'].fillna(' ')


    voc['start_datetime']=pd.to_datetime(voc['start_datetime'])
    voc['hour']=voc['start_datetime'].dt.hour
    all_user=name['phone_no_m'].unique().tolist()


    #建图
    graph = nx.MultiGraph()

    name=name.to_dict('records')
    #个人信息特征
    id_feature=dict()

    #标签
    label=dict()
    #个人通话时长
    call_dur_dict=dict()

    #出入度
    indegree_dict=dict()
    outdegree_dict=dict()

    recipe_call=dict()

    #呼叫时间分布
    call_time_distribute=dict()

    voc['call_time_onday']=voc['start_datetime'].dt.hour


    #加点，属性特征提取
    for id in name:
        phone_no_m=id['phone_no_m']
        if not isinstance(id['city_name'],str):
            id['city_name']=' '
        if not isinstance(id['county_name'],str):
            id['county_name']=' '
        place=' '.join([id['city_name'],id['county_name']])
        place,place_code_dict=get_place_code(place,place_code_dict)
        if not isinstance(id['idcard_cnt'],(int,float)):
            id['idcard_cnt']=0
        idcard_cnt=id['idcard_cnt']
        id_feature[phone_no_m]=[place,idcard_cnt] #地方编码，名下账户数
        label[phone_no_m]=id['label']

        graph.add_node(phone_no_m)
    voc.sort_values(by='start_datetime')
    voc=voc.to_dict('records')

    # uncare_user_place=dict()    #key是user，value是place id
    #加边，
    for edge in voc:

        phone_no_m=edge['phone_no_m']
        opposite_no_m=edge['opposite_no_m']

        if opposite_no_m not in id_feature:
            if edge['calltype_id']==2:
                city_name=edge['city_name']
                county_name=edge['county_name']
                if not isinstance(city_name,str):
                    city_name=' '
                if not isinstance(county_name,str):
                    county_name=' '
                place,place_code_dict=get_place_code(' '.join([city_name,county_name]),place_code_dict)
                id_feature[opposite_no_m]=[place,0]
            else:
                id_feature[opposite_no_m]=[-1,0]

        # if opposite_no_m not in id_feature and edge['calltype_id']==2:    #如果opposite no 不在信息表中
        #     place=' '.join([edge['city_name'],edge['county_name']])
        #     place,place_code_dict=get_place_code(place,place_code_dict)
        #     uncare_user_place[opposite_no_m]=place
        if phone_no_m not in call_dur_dict:
            call_dur_dict[phone_no_m]=list()
        call_dur_dict[phone_no_m].append(edge['call_dur'])
        if opposite_no_m not in call_dur_dict:
            call_dur_dict[opposite_no_m]=list()
        call_dur_dict[opposite_no_m].append(edge['call_dur'])
        if phone_no_m not in call_time_distribute:
            call_time_distribute[phone_no_m]=[0 for i in range(24)]
        call_time_distribute[phone_no_m][edge['call_time_onday']]+=1
        if opposite_no_m not in call_time_distribute:
            call_time_distribute[opposite_no_m]=[0 for i in range(24)]
        call_time_distribute[opposite_no_m][edge['call_time_onday']]+=1
        #回拨率还需要讨论
        graph.add_edge(phone_no_m,opposite_no_m)


        if edge['calltype_id']==1:
            if opposite_no_m not in indegree_dict:
                indegree_dict[opposite_no_m]=0
            indegree_dict[opposite_no_m]+=1
            if phone_no_m not in outdegree_dict:
                outdegree_dict[phone_no_m]=0
            outdegree_dict[phone_no_m]+=1

        elif edge['calltype_id']==2:
            if phone_no_m not in indegree_dict:
                indegree_dict[phone_no_m]=0
            if opposite_no_m not in outdegree_dict:
                outdegree_dict[opposite_no_m]=0
            indegree_dict[phone_no_m]+=1
            outdegree_dict[opposite_no_m]+=1
        graph.add_edge(phone_no_m,opposite_no_m)

    k_common_neighbors=dict()

    #共同好友图
    for node in all_user:   #all_user是知道信息的user
        if node not in graph:continue
        for node2 in all_user:
            if node2==node:
                continue
            if node2 not in graph:
                continue
            connector1=[k for k,v in graph[node].items()]
            connector2=[k for k,v in graph[node2].items()]
            if len(set(connector1)&set(connector2))>=k_common_connector:
                if node not in k_common_neighbors:k_common_neighbors[node]=list()
                k_common_neighbors[node].append(node2)

    graph_k_common=nx.Graph()
    for k,v in k_common_neighbors:
        for neighbor in v:
            graph_k_common.add_edge(k,neighbor)

    #属性相似度图
    graph_attrib_similar=nx.Graph()
    for node in all_user:
        neighbor_similar=dict()
        
        for node2 in all_user:
            if node2==node:continue
            
    

    #邻居特征提取
    for node in all_user:
        neighbor_place=[]
        neighbor_place_frequen=dict()
        neighbors_degree=[]

        for neighbor in graph.neighbors(node):
            # if neighbor not in id_feature:
            #     neighbor_place.append(lambda x:uncare_user_place[x] if x in uncare_user_place else -1)

            neighbor_place.append(id_feature[neighbor][0])     #有bug，找不到某个neighbor，有可能，因为不保证neighbor信息也在其中
            neighbors_degree.append(graph.degree(neighbor))
        for place in neighbor_place:
            if place not in neighbor_place_frequen:
                neighbor_place_frequen[place]=1
            else:
                neighbor_place_frequen[place]+=1
        all_place=0
        for k,v in neighbor_place_frequen.items():
            all_place+=v
        for k,v in neighbor_place_frequen.items():
            neighbor_place_frequen[k]=v/all_place
        entropy=0
        for k,v in neighbor_place_frequen.items():
            entropy+=-v*math.log(v)
        id_feature[node].append(entropy)      #所有邻居地点熵
        id_feature[node].append(indegree_dict[node] if node in indegree_dict else 0)  #入度
        id_feature[node].append(outdegree_dict[node] if node in outdegree_dict else 0)  #出度
        id_feature[node].append(np.mean(neighbors_degree))  #邻居度平均
        #聚集系数


        id_feature[node].append(np.mean(call_dur_dict[node]) if node in call_dur_dict else 0)  #平均通话时长
        neighbor_mean_durs=[]
        for neighbor in graph.neighbors(node):
            neighbor_mean_durs.append(np.mean(call_dur_dict[neighbor]) if neighbor in call_dur_dict else 0)
        id_feature[node].append(np.mean(neighbor_mean_durs))  # 邻居平均每次通话时长


        id_feature[node].append(np.var(call_dur_dict[node]) if node in call_dur_dict else 0)  #每次通话时长的方差

        id_feature[node].append(len(set(graph.neighbors(node))))#能量分散

        id_feature[node].append(call_time_distribute[node] if node in call_time_distribute else [0 for i in range(24)])    #拨打时间分布


    after_file_path=data_path+'data_after/baselinepapermodel/sichuan.dat'
    file=open(after_file_path,'w')

    #属性信息输出
    for user in all_user:
        wristr=''
        if user in test_user:
            wristr+='?'+str(label[user])+'\t'
        else:
            wristr+='+'+str(label[user])+'\t'
        feats=id_feature[user]
        wristr+='place_code_'+str(feats[0])+':1\t'
        wristr+='idcard_cnt_'+str(feats[1])+':1\t'
        entropy1=feats[2]
        entropy2=int(entropy1)
        if entropy1==0:

            wristr+='entropy_'+str(0)+':1\t'
        else:
            wristr += 'entropy_' + str(entropy1) + ':1\t'
        wristr+='indegree_'+str(feats[3])+':1\t'
        wristr+='outdegree_'+str(feats[4])+':1\t'
        wristr+='neighbor_mean_degree_'+str(feats[5])+':1\t'
        wristr+='mean_call_dur_'+str(feats[6])+':1\t'
        wristr+='mean_call_dur_neighbor_'+str(feats[7])+':1\t'
        wristr+='var_call_dur_'+str(feats[8])+':1\t'
        wristr+='disperation_'+str(feats[9])+':1\t'
        for idx,call_time in enumerate(feats[10]):
            wristr+=str(idx)+'_'+str(call_time)+':1\t'
        wristr+='\n'
        file.write(wristr)


    #边信息输出
    for idx,user in enumerate(graph.nodes):

        for neighbor in graph[user]:
            if user not in all_user or neighbor not in all_user:
                continue
            wristr='#edge'
            if all_user.index(neighbor)<=all_user.index(user):
                continue
            if graph.number_of_edges(user,neighbor)>=call_times:
                wristr=' '.join([wristr,str(all_user.index(user)),str(all_user.index(neighbor))])
                wristr+='\n'
                file.write(wristr)









    pass
    #print(nx.degree(graph))

def test():
    data_path = '/home/m21_huangzijun/pythonprojs/sichuan/'
    id_file = data_path + 'data/0527/train/train_user.csv'
    user=pd.read_csv(id_file)
    hey =user['phone_no_m'].unique().tolist()
    print(hey[0])

def main2():
    data_path='/home/m21_huangzijun/pythonprojs/sichuan/data/0527/test'
    user_info=pd.read_csv(data_path+'/test_user1.csv')
    user_info=user_info.set_index('phone_no_m',drop=False)
    voc=pd.read_csv(data_path+'/test_voc.csv')
    voc['start_datetime']=pd.to_datetime(voc['start_datetime'])
    voc['hour']=voc['start_datetime'].dt.hour

    voc_group={k:v for k,v in voc.groupby('phone_no_m')}
    
    train_ids=np.load('/home/m21_huangzijun/pythonprojs/sichuan/data_after/mymodel5_2_onemonth'+'/train_ids.npy')
    test_ids=np.load('/home/m21_huangzijun/pythonprojs/sichuan/data_after/mymodel5_2_onemonth'+'/test_ids.npy')


    ids=np.concatenate([train_ids,test_ids])

    with timer('degree'):    
        calling=voc[['phone_no_m','opposite_no_m']].drop_duplicates()
        calling['degree']=calling.groupby('phone_no_m')['opposite_no_m'].transform('count')
        degree=calling[['phone_no_m','degree']].drop_duplicates().set_index('phone_no_m')

    called=voc.loc[voc['calltype_id']==2].drop_duplicates(subset=['phone_no_m','opposite_no_m'])
    called['indegree']=called.groupby('phone_no_m')['opposite_no_m'].transform('count')
    called=called[['phone_no_m','indegree']]
    # voc=voc.merge(called,how='left',on='phone_no_m')
    indegree=called[['phone_no_m','indegree']].drop_duplicates().set_index('phone_no_m',drop=False)

    called=voc.loc[voc['calltype_id']==1].drop_duplicates(subset=['phone_no_m','opposite_no_m'])
    called['outdegree']=called.groupby('phone_no_m')['opposite_no_m'].transform('count')
    called=called[['phone_no_m','outdegree']]
    # voc=voc.merge(called,how='left',on='phone_no_m')
    outdegree=called[['phone_no_m','outdegree']].drop_duplicates().set_index('phone_no_m',drop=False)

    voc['mean_duration']=voc.groupby('phone_no_m')['call_dur'].transform(np.mean)
    voc['var_duration']=voc.groupby('phone_no_m')['call_dur'].transform(np.var)
    mean_duration=voc[['phone_no_m','mean_duration']].drop_duplicates().set_index('phone_no_m')
    var_duration=voc[['phone_no_m','var_duration']].drop_duplicates().set_index('phone_no_m')

    voc['calltimes']=voc.groupby('phone_no_m')['opposite_no_m'].transform('count')
    voc['everyone_times']=voc.groupby(['phone_no_m','opposite_no_m'])['opposite_no_m'].transform('count')
    voc['energy_eachone']=voc['everyone_times']/voc['calltimes']
    voc['energy_dispersion']=voc.groupby('phone_no_m')['energy_eachone'].transform(np.mean)
    del voc['calltimes']
    del voc['everyone_times']
    del voc['energy_eachone']
    energy_dispersion=voc[['phone_no_m','energy_dispersion']].drop_duplicates().set_index('phone_no_m')
    
    with timer('graph'):
    #建图
        nodeingraph=set()
        graph=nx.Graph()
        rs=np.array(voc[['phone_no_m','opposite_no_m']])
        for r in rs:
            source_node=r[0]
            target_node=r[1]
            # calltype=r['calltype_id']
            graph.add_edge(source_node,target_node)
            nodeingraph.add(source_node)
            # nodeingraph.add(target_node)

    #   年龄和性别、邻居的年龄分布、邻居的性别分布、邻居地点的熵、
    # 入度、出度、邻居度、聚类系数、平均通话时长、邻居平均通话时长、通话时长方差、精力分散度、通话时间分布
    fs=[]
    count=0
    for id in train_ids:
        if id in nodeingraph:
            idcard_cnt=user_info.loc[id,'idcard_cnt']
            try:
                in_degree=indegree.loc[id,'indegree']
            except KeyError:
                in_degree=0
            try:
                out_degree=outdegree.loc[id,'outdegree']
            except KeyError:
                out_degree=0
            neigh_degrees=[]
            neighs=graph.neighbors(id)
            for neigh in neighs:
                neigh_degrees.append(graph.degree[neigh])
            neigh_degree=np.mean(neigh_degrees)
            # clus=nx.clustering(graph,id)
            mean_dur=mean_duration.loc[id,'mean_duration']
            var_dur=var_duration.loc[id,'var_duration']
            ed=energy_dispersion.loc[id,'energy_dispersion']
            time_dis=[0 for i in range(24)]
            the_voc=voc_group[id]
            time_count=the_voc['hour'].value_counts(normalize=True)
            for i in range(24):
                time_dis[i]=time_count[i] if i in time_count else 0
            f=[idcard_cnt,in_degree,out_degree,neigh_degree,mean_dur,var_dur,ed]+time_dis
        else:
            f=[0 for i in range(31)]
        fs.append(f)
        print(count)
        count+=1
    for id in test_ids:
        if id in nodeingraph:
            idcard_cnt=user_info.loc[id,'idcard_cnt']
            try:
                in_degree=indegree.loc[id,'indegree']
            except KeyError:
                in_degree=0
            try:
                out_degree=outdegree.loc[id,'outdegree']
            except KeyError:
                out_degree=0
            neigh_degrees=[]
            neighs=graph.neighbors(id)
            for neigh in neighs:
                neigh_degrees.append(graph.degree[neigh])
            neigh_degree=np.mean(neigh_degrees)
            # clus=nx.clustering(graph,id)
            mean_dur=mean_duration.loc[id,'mean_duration']
            var_dur=var_duration.loc[id,'var_duration']
            ed=energy_dispersion.loc[id,'energy_dispersion']
            time_dis=[0 for i in range(24)]
            the_voc=voc_group[id]
            time_count=the_voc['hour'].value_counts(normalize=True)
            for i in range(24):
                time_dis[i]=time_count[i] if i in time_count else 0
            f=[idcard_cnt,in_degree,out_degree,neigh_degree,mean_dur,var_dur,ed]+time_dis
        else:
            f=[0 for i in range(31)]
        fs.append(f)
        print(count)
        count+=1
    
    edges=[]

    ids=ids.tolist()
    for r in rs:
        s=r[0]
        t=r[1]
        if (s not in ids) or (t not in ids):
            continue
        s=ids.index(s)
        t=ids.index(t)
        edges.append((s,t))
    
    write_path='/home/m21_huangzijun/pythonprojs/sichuan/data_after/baselinepapermodel'
    np.save(write_path+'/sichuan3_f_onemonth.npy',fs)
    np.save(write_path+'/sichuan3_e_onemonth.npy',edges)
    with open (write_path+'/sichuan3_onemonth.dat','w',encoding='utf-8') as w:
        
        for i in range(len(fs)):
            if i<len(train_ids):line='+'
            else:line='?'
            label=user_info.loc[ids[i],'label']
            line+=str(label)+'\t'
            count=0
            for each_f in fs[i]:
                
                line+='fea_{}_{}:1\t'.format(count,each_f)
                count+=1
            w.write(line+'\n')
            

        for i in range(len(edges)):
            w.write('#edge\t{}\t{}\n'.format(edges[i][0],edges[i][1]))





def test():
    fs=np.load('/home/m21_huangzijun/pythonprojs/sichuan/data_after/baselinepapermodel/sichuan3_f.npy')
    es=np.load('/home/m21_huangzijun/pythonprojs/sichuan/data_after/baselinepapermodel/sichuan3_e.npy')
    print(fs.shape)

def Amazon():
    datafile=loadmat('/home/m21_huangzijun/pythonprojs/sichuan/Amazon.mat')
    feats=datafile['features']
    feats=feats.todense().A
    homo=datafile['homo']
    upu=datafile['net_upu']
    usu=datafile['net_usu']
    uvu=datafile['net_uvu']
    label=datafile['label']
    label=label.flatten()
    homo_edge=homo.nonzero()
    upu_edge=upu.nonzero()
    usu_edge=usu.nonzero()
    uvu_edge=uvu.nonzero()
    indices=np.arange(feats.shape[0])
    [x_train,x_test,y_train,y_test,indices_train,indices_test]=train_test_split(feats,label,indices,test_size=0.4)
    indices_train=set(indices_train)
    indices_test=set(indices_test)
    write_path='/home/m21_huangzijun/pythonprojs/sichuan/data_after/baselinepapermodel/amazon.dat'
    allgraph=nx.Graph()
    for u,v in zip(homo_edge[0],homo_edge[1]):
        allgraph.add_edge(u,v)
    for u,v in zip(upu_edge[0],upu_edge[1]):
        allgraph.add_edge(u,v)
    for u,v in zip(usu_edge[0],usu_edge[1]):
        allgraph.add_edge(u,v)
    for u,v in zip(uvu_edge[0],uvu_edge[1]):
        allgraph.add_edge(u,v)
    with open(write_path,'w',encoding='utf-8') as w:
        for i in range(len(feats)):
            if i in indices_train:
                w.write('+'+'{}\t'.format(int(label[i])))
            else:
                w.write('?'+'{}\t'.format(int(label[i])))
            for j,f in enumerate(feats[i]):
                w.write('feat_{}_{}:1\t'.format(j,f))
            w.write('\n')
        for u,v in allgraph.edges:
            w.write('#edge {} {}\n'.format(u,v))
        # for u,v in zip(homo_edge[0],homo_edge[1]):
        #     w.write('#edge {} {}\n'.format(u,v))
        # for u,v in zip(homo_edge[0],homo_edge[1]):
        #     w.write('#edge {} {}\n'.format(u,v))
        
    pass

def YelpChi():
    datafile=loadmat('/home/m21_huangzijun/pythonprojs/sichuan/YelpChi.mat')
    feats=datafile['features']
    feats=feats.todense().A
    homo=datafile['homo']
    rur=datafile['net_rur']
    rtr=datafile['net_rtr']
    rsr=datafile['net_rsr']
    label=datafile['label']
    label=label.flatten()
    homo_edge=homo.nonzero()
    upu_edge=rur.nonzero()
    usu_edge=rtr.nonzero()
    uvu_edge=rsr.nonzero()
    indices=np.arange(feats.shape[0])
    [x_train,x_test,y_train,y_test,indices_train,indices_test]=train_test_split(feats,label,indices,test_size=0.4)
    indices_train=set(indices_train)
    indices_test=set(indices_test)
    write_path='/home/m21_huangzijun/pythonprojs/sichuan/data_after/baselinepapermodel/YelpChi.dat'
    allgraph=nx.Graph()
    for u,v in zip(homo_edge[0],homo_edge[1]):
        allgraph.add_edge(u,v)
    for u,v in zip(upu_edge[0],upu_edge[1]):
        allgraph.add_edge(u,v)
    for u,v in zip(usu_edge[0],usu_edge[1]):
        allgraph.add_edge(u,v)
    for u,v in zip(uvu_edge[0],uvu_edge[1]):
        allgraph.add_edge(u,v)
    with open(write_path,'w',encoding='utf-8') as w:
        for i in range(len(feats)):
            if i in indices_train:
                w.write('+'+'{}\t'.format(int(label[i])))
            else:
                w.write('?'+'{}\t'.format(int(label[i])))
            for j,f in enumerate(feats[i]):
                w.write('feat_{}_{}:1\t'.format(j,f))
            w.write('\n')
        for u,v in allgraph.edges:
            w.write('#edge {} {}\n'.format(u,v))
        # for u,v in zip(homo_edge[0],homo_edge[1]):
        #     w.write('#edge {} {}\n'.format(u,v))
        # for u,v in zip(homo_edge[0],homo_edge[1]):
        #     w.write('#edge {} {}\n'.format(u,v))
        
    pass


if __name__ == '__main__':
    # main2()
    YelpChi()
    # not_common()