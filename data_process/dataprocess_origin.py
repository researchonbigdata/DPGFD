import pandas as pd
from contextlib import contextmanager
import time as timee
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


import torch
from torch.nn import LSTM

def get_code(origin,mapping_dict):
    if origin in mapping_dict:
        return mapping_dict[origin],mapping_dict
    else:
        mapping_dict[origin]=len(mapping_dict)
        return mapping_dict[origin],mapping_dict

@contextmanager
def timer(title):
    t0 = timee.time()
    yield
    print("{} - done in {:.0f}s".format(title, timee.time() - t0))

def main():
    data_path='/home/m21_huangzijun/pythonprojs/sichuan/'
    id_file=data_path+'data/train_user0.csv'
    voc_file=data_path+'data/0527/train/train_voc.csv'
    name=pd.read_csv(id_file)
    voc=pd.read_csv(voc_file)

    #数据类型转换，数据清洗
    voc['location']=voc['city_name']+' '+voc['county_name']
    voc['start_datetime']=pd.to_datetime(voc['start_datetime'])

    n_target_no=voc['opposite_no_m'].nunique()
    n_location=voc['location'].nunique()
    
    # 编码
    encoder=LabelEncoder.fit(voc['opposite_no_m'])
    voc['opposite_no_m']=encoder.transform(voc['opposite_no_m'])

    encoder=LabelEncoder.fit(voc['location'])
    voc['location']=encoder.transform(voc['location'])

    # opposite_path_str=data_path+'data_after/mymodel/opposite_code.npy' #对方账户编码
    # location_path_str=data_path+'data_after/mymodel/place_code.npy'  #地点编码
    # imei_path_str=data_path+'data_after/mymodel/imei_code.npy'#imei编码
    # opposite_path=Path(opposite_path_str)
    # opposite_dict={}

    # location_path=Path(location_path_str)
    # location_dict={}
    
    # imei_m_path=Path(imei_path_str)
    # imei_m_dict={}
    
    # label={}
    # feats={}
    # if opposite_path.is_file():
    #     opposite_dict=np.load(opposite_path_str,allow_pickle=True).item()
    # if location_path.is_file():
    #     location_dict=np.load(location_path_str,allow_pickle=True).item()
    # if imei_m_path.is_file():
    #     imei_m_dict=np.load(imei_path_str,allow_pickle=True).item()


    voc=voc.merge(name[['phone_no_m','label']],how='left',on='phone_no_m')
    name_voc=pd.merge(name,voc,on='phone_no_m',suffixes=['_L','_R'])
    name_voc['start_datetime']=pd.to_datetime(name_voc['start_datetime'])
    name_voc.sort_values(by='start_datetime',inplace=True)
    name_voc=name_voc.to_dict(orient='records')
    for record in name_voc:
        #print(record['imei_m'])
        if record['opposite_no_m'] not in opposite_dict:
            opposite_dict[record['opposite_no_m']]=len(opposite_dict)
        start_datetime=record['start_datetime'].to_pydatetime().timestamp()
        location=' '.join([str(record['city_name_R']),str(record['county_name_R'])])
        if location not in location_dict:
            location_dict[location]=len(location_dict)
        if record['imei_m'] not in imei_m_dict:
            imei_m_dict[record['imei_m']]=len(imei_m_dict)
        if record['phone_no_m'] not in feats:
            feats[record['phone_no_m']]=list()
        feats[record['phone_no_m']].append([opposite_dict[record['opposite_no_m']],record['calltype_id'],location_dict[location],start_datetime,record['call_dur'],imei_m_dict[record['imei_m']]])
        if record['phone_no_m'] not in label:
            label[record['phone_no_m']]=record['label']
    n_people=len(label)
    static_feats={}
    name=name.to_dict('records')
    for record in name:
        location=' '.join([str(record['city_name']),str(record['county_name'])])
        if location not in location_dict:
            location_dict[location]=len(location_dict)
        if record['phone_no_m'] not in static_feats:
            static_feats[record['phone_no_m']]=list()
        static_feats[record['phone_no_m']]=[location_dict[location],record['idcard_cnt']]
    train_n=int(n_people/10*7)
    train_x=[]
    train_x_static=[]
    train_y=[]
    test_x=[]
    test_x_static=[]
    test_y=[]
    people=[k for k,v in label.items()]
    for person in people:
        train_x.append(feats[person])
        train_x_static.append(static_feats[person])
        train_y.append(label[person])
    
    np.save(file=data_path+'data_after/mymodel/train_x_origin.npy',arr=train_x)
    np.save(file=data_path+'data_after/mymodel/train_x_static_origin.npy',arr=train_x_static)
    np.save(file=data_path+'data_after/mymodel/train_y_origin.npy',arr=train_y)
    np.save(file=data_path+'data_after/mymodel/test_x_origin.npy',arr=test_x)
    np.save(file=data_path+'data_after/mymodel/test_x_static_origin.npy',arr=test_x_static)
    np.save(file=data_path+'data_after/mymodel/test_y_origin.npy',arr=test_y)
    test_x = []
    test_x_static = []
    test_y = []
    '''
    保存编码字典
    '''
    np.save(file=opposite_path_str,arr=opposite_dict)
    np.save(file=location_path_str,arr=location_dict)
    np.save(file=imei_path_str,arr=imei_m_dict)

    pass

def main_2():
    model=torch.load('lstmae.pt')
    #读字典
    data_path='/home/m21_huangzijun/pythonprojs/sichuan/'
    opposite_path_str=data_path+'data_after/mymodel/opposite_code.npy' #对方账户编码
    location_path_str=data_path+'data_after/mymodel/place_code.npy'  #地点编码
    imei_path_str=data_path+'data_after/mymodel/imei_code.npy'#imei编码
    opposite_path=Path(opposite_path_str)
    opposite_dict={}

    location_path=Path(location_path_str)
    location_dict={}
    
    imei_m_path=Path(imei_path_str)
    imei_m_dict={}

    if opposite_path.is_file():
        opposite_dict=np.load(opposite_path_str,allow_pickle=True).item()
    if location_path.is_file():
        location_dict=np.load(location_path_str,allow_pickle=True).item()
    if imei_m_path.is_file():
        imei_m_dict=np.load(imei_path_str,allow_pickle=True).item()

    #读数据
    id_file=data_path+'data/train_user0.csv'
    voc_file=data_path+'data/0527/train/train_voc.csv'
    name=pd.read_csv(id_file)
    voc=pd.read_csv(voc_file)
    #清洗数据
    voc['opposite_no_m'].fillna(' ')
    voc['calltype_id'].fillna(-1)
    voc['city_name'].fillna(' ')
    voc['county_name'].fillna(' ')
    voc['start_datetime'].fillna('0/0/0 0:0')
    voc['call_dur'].fillna(0)
    voc['imei_m'].fillna(' ')

    label=name[['phone_no_m','label']].to_dict(orient='records')
    people=[record['phone_no_m'] for record in label]
    voc['start_datetime']=pd.to_datetime(voc['start_datetime'])
    vocs=[week_vocs for i,week_vocs in voc.groupby(pd.Grouper(key='start_datetime',freq='W'))]
    people_feats=dict()
    for week_vocs in vocs:   #每一周的pandas表
        week_persons=[v for k,v in week_vocs.groupby('phone_no_m')]
        for person in week_persons:       #每个人的pandas表
            person.sort_values('start_datetime',inplace=True)   #通话记录按时间排序
            person_list=person.to_dict(orient="records")
            who=person_list[0]['phone_no_m']
            if who not in people_feats:
                people_feats[who]=list()
            week_feature=[]
            for voc_rec in person_list: #每条通话记录
                phone_no_m=voc_rec['phone_no_m']
                opposite_no_m,opposite_dict=get_code(voc_rec['opposite_no_m'],opposite_dict)
                calltype=voc_rec['calltype_id']
                location,location_dict=get_code(' '.join([voc_rec['city_name'],voc_rec['county_name']]),location_dict)
                start_datetime=voc_rec['start_datetime'].to_pydatetime().timestamp()
                call_dur=voc_rec['call_dur']
                imei,imei_m_dict=get_code(voc_rec['imei_m'],imei_m_dict)
                week_feature.append([opposite_no_m,calltype,location,start_datetime,call_dur,imei])
            week_feature=torch.tensor(week_feature,dtype=torch.float).to(1)
            x,hidden=model(week_feature)
            people_feats[who].append()

def main3():
    data_path='/home/m21_huangzijun/pythonprojs/sichuan/'
    id_file=data_path+'data/0527/train/train_user.csv'
    voc_file=data_path+'data/0527/train/train_voc.csv'
    train_id=data_path+'data/train_user0.csv'
    name=pd.read_csv(id_file)
    voc=pd.read_csv(voc_file)
    train_name=pd.read_csv(train_id)

    #数据类型转换，数据清洗
    voc['city_name'].fillna(' ',inplace=True)
    voc['county_name'].fillna(' ',inplace=True)
    voc['location']=voc['city_name']+' '+voc['county_name']
    voc['start_datetime'].fillna('')
    voc['start_datetime']=pd.to_datetime(voc['start_datetime'])

    n_target_no=voc['opposite_no_m'].nunique()
    n_location=voc['location'].nunique()
    
    # 编码
    encoder=LabelEncoder().fit(voc['opposite_no_m'])
    voc['opposite_no_m']=encoder.transform(voc['opposite_no_m'])

    encoder=LabelEncoder().fit(voc['location'])
    voc['location']=encoder.transform(voc['location'])

    voc=voc.merge(name[['phone_no_m','label']],how='left',on='phone_no_m')

    train_voc=voc[voc['phone_no_m'].isin(train_id['phone_no_m'])]
    test_voc=voc[~(voc['phone_no_m'].isin(train_id['phone_no_m']))]


    

if __name__=='__main__':
    main3()