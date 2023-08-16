import argparse
import os

parser=argparse.ArgumentParser()
parser.add_argument('--num_features',type=int,default=31)
parser.add_argument('--hidden_size',type=int,default=64)
parser.add_argument('--id_feature_size',type=int,default=2)
parser.add_argument('--device',type=str,default='2')
parser.add_argument('--lr',type=list,default=[0.1,0.0003,0.0003,0.0003])
parser.add_argument('--dropout',type=float,default=0.5)
parser.add_argument('--eps',type=float,default=0.3)
parser.add_argument('--epoches',type=int,default=100)
parser.add_argument('--dataset', default='Sichuan')
parser.add_argument('--model', default='GraphSage')
parser.add_argument('--train_ratio',default=0.4)
parser.add_argument('--seed',default=2)

arg=parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=arg.device


import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset,DataLoader
import numpy as np

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from collections import Counter
import math
import dgl
from dgl.nn import GraphConv
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GraphSAGE


class MyGraphSage(nn.Module):
    def __init__(self,arg) -> None:
        super(MyGraphSage,self).__init__()
        self.graphsage = GraphSAGE(in_channels=arg.num_features,hidden_channels=32,out_channels=2,num_layers=2)
    def forward(self,x,edge_index):
        pre = self.graphsage(x,edge_index)
        return pre

class MyGCN(nn.Module):
    def __init__(self,arg) -> None:
        super().__init__()
        self.conv1=GraphConv(arg.num_features,2)

    def forward(self,x,g):
        pre=self.conv1(g,x)
        
        return pre

def twodnormalize(x):
    for i in range(x.shape[-1]):
        x[:,i]=(x[:,i]-np.mean(x[:,i]))/np.std(x[:,i])
    return x
def train():
    if arg.dataset=="Sichuan":
        prefix='/data01/social_network_group/m21_huangzijun/sichuan/data_after/mymodel5_2'
        x=np.load(f'/data01/social_network_group/m21_huangzijun/sichuan/data_after/baselinepapermodel/sichuan3_f.npy')
    elif arg.dataset=='Sichuan_one':
        prefix='/data01/social_network_group/m21_huangzijun/sichuan/data_after/mymodel5_2_onemonth'
        x=np.load(f'/data01/social_network_group/m21_huangzijun/sichuan/data_after/baselinepapermodel/sichuan3_f_onemonth.npy')
    
    x=np.nan_to_num(x)
    x=twodnormalize(x)
    

    x=torch.from_numpy(x).type(torch.float32).cuda()
    
    

    label=np.load(prefix+f'/y_train.npy')
    label2=np.load(prefix+f'/y_test.npy')

    label=torch.from_numpy(label).type(torch.long).cuda()
    label2=torch.from_numpy(label2).type(torch.long).cuda()

    label=torch.concat([label,label2],dim=0)
    
    train_id=np.load(prefix+'/train_ids.npy')
    test_id=np.load(prefix+'/test_ids.npy')

    

    num_node=len(train_id)+len(test_id)
    index = [i for i in range(num_node)]

    idx_train, idx_rest, y_train, y_rest = train_test_split(index, label,train_size=arg.train_ratio, random_state=arg.seed,shuffle=True)
    idx_val, idx_test, y_val, y_test= train_test_split(idx_rest,y_rest,test_size=0.66,random_state=arg.seed, shuffle=True)

    source=np.load(prefix+'/source.npy')
    target=np.load(prefix+'/target.npy')
    tmps=source.copy()
    tmpt=target.copy()
    source=np.concatenate([source,tmpt])
    target=np.concatenate([target,tmps])
    edge_index=np.stack([source,target],axis=0)
    edge_index=torch.from_numpy(edge_index).cuda()
    
    device=torch.device('cuda:'+arg.device)
    

    model=MyGraphSage(arg).cuda()
    loss=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.03
    )
    losses=list()
    xlabel=list()

    best_auc = 0

    auc_test=0
    recall_test=0
    f1_test=0

    for epoch in range(arg.epoches):
        pre=model(x,edge_index)

        optimizer.zero_grad()

        loss_v=loss(pre[idx_train],label[idx_train])
        loss_v.backward()
        losses.append(loss_v.item())
        xlabel.append(epoch)
        optimizer.step()

        ## validation
        if epoch % 5==0 and epoch!=0:
            pro_val = pre[idx_val]
            pro_val = pro_val.detach().clone().cpu().numpy()
            pre_val = np.argmax(pro_val, axis=-1)
            pos_val = pro_val[:,-1]
            label_val = label[idx_val].detach().clone().cpu().numpy()

            auc_val = roc_auc_score(label_val,pos_val)

            if auc_val>best_auc:
                best_auc = auc_val

                pro_test = pre[idx_test]
                pro_test = pro_test.detach().clone().cpu().numpy()
                pre_test = np.argmax(pro_test, axis=-1)
                pos_test = pro_test[:,-1]
                label_test = label[idx_test].detach().clone().cpu().numpy()

                auc_test=roc_auc_score(label_test,pos_test)
                recall_test = recall_score(label_test,pre_test)
                f1_test = f1_score(label_test,pre_test)

    pre=model(x,edge_index)
    pro_test = pre[idx_test]
    pro_test = pro_test.detach().clone().cpu().numpy()
    pre_test = np.argmax(pro_test, axis=-1)
    pos_test = pro_test[:,-1]
    label_test = label[idx_test].detach().clone().cpu().numpy()

    auc_test=roc_auc_score(label_test,pos_test)
    recall_test = recall_score(label_test,pre_test)
    f1_test = f1_score(label_test,pre_test)
    plt.cla()
    plt.plot(xlabel,losses,)
    plt.savefig('loss_picture/{}_{}.png'.format(arg.model, arg.dataset))
    print('test:',
          'auc:%.4f'%auc_test,
          'recall:%.4f'%recall_test,
          'f1:%.4f'%f1_test)
    

train()
