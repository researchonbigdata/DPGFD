from typing import Any, Dict, List, Optional, Union
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn.aggr import Aggregation
from model_source_adap.layers import Adagnn_without_weight, Adagnn_with_weight
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import dgl
from dgl.nn import GraphConv
import dgl.function as fn
import math
import numpy as np

class AdaGNN(nn.Module):
    def __init__(self, diag_dimension, nfeat, nhid, nlayer, nclass, dropout,edge_index,labels):
        super(AdaGNN, self).__init__()

        # 1 with weight 2 without weight 1 with weight

        self.should_train_1 = Adagnn_with_weight(diag_dimension, nfeat, nhid)
        assert nlayer - 2 >= 0
        self.hidden_layers = nn.ModuleList([
            Adagnn_without_weight(nfeat, nhid, nhid, bias=False)
            for i in range(nlayer - 2)
        ]) 
        self.should_train_2 = Adagnn_with_weight(diag_dimension, nhid, nclass)
        self.dropout = dropout

    def forward(self, x, l_sym):


        x = F.relu(self.should_train_1(x, l_sym))  # .relu
        x = F.dropout(x, self.dropout, training=self.training)

        for i, layer in enumerate(self.hidden_layers):
            x = layer(x, l_sym)
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.should_train_2(x, l_sym)  # + res1
        return F.log_softmax(x, dim=1)
    def getsigma(self):
        return self.should_train_2.getsigma()


class FAdaGNN(MessagePassing):
    def __init__(self, diag_dimension, nfeat, nhid, nlayer, nclass, dropout,edge_index,labels):
        super(FAdaGNN, self).__init__()

        self.edge_index = edge_index
        self.labels = labels
        self.gate = nn.Linear(2*nhid,1)
        self.row, self.col = edge_index
        self.norm_degree = degree(self.row, num_nodes=labels.shape[0],).clamp(min=1)
        self.norm_degree = torch.pow(self.norm_degree, -0.5)

        self.should_train_1 = Adagnn_with_weight(diag_dimension, nfeat, nhid)
        assert nlayer - 2 >= 0
        self.hidden_layers = nn.ModuleList([
            Adagnn_without_weight(nfeat, nhid, nhid, bias=False)
            for i in range(nlayer - 2)
        ]) 
        self.should_train_2 = Adagnn_with_weight(diag_dimension, nhid, nclass)
        self.dropout = dropout
        self.dropout_nn = nn.Dropout(dropout)

    def forward(self, x, l_sym):
        h2=torch.cat((x[self.row],x[self.col]),dim=1)
        g=torch.tanh(self.gate(h2)).squeeze()

        norm = g*self.norm_degree[self.row]*self.norm_degree[self.col]
        norm = self.dropout_nn(norm)
        x = self.propagate(self.edge_index,size=(x.size(0),x.size(0)),x=x, norm=norm)


        x = F.relu(self.should_train_1(x, l_sym))  # .relu
        x = F.dropout(x, self.dropout, training=self.training)

        for i, layer in enumerate(self.hidden_layers):
            x = layer(x, l_sym)
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.should_train_2(x, l_sym)  # + res1
        return F.log_softmax(x, dim=1)
    def getsigma(self):
        return self.should_train_2.getsigma()


class SeveralLevelAttnLstm(nn.Module):
    def __init__(self, arg) -> None:
        super().__init__()
        # self.device=torch.device(arg.device)
        self.embedding = Parameter(torch.rand(
            size=(1, arg.hidden),), requires_grad=True)
        self.feature_attn = Parameter(torch.rand(
            size=(arg.num_features,),), requires_grad=True)
        self.weekLstm = nn.LSTM(
            input_size=arg.hidden, hidden_size=arg.hidden, batch_first=True)
        self.twoweekLstm = nn.LSTM(
            input_size=arg.hidden, hidden_size=arg.hidden, batch_first=True)
        self.threeweekLstm = nn.LSTM(
            input_size=arg.hidden, hidden_size=arg.hidden, batch_first=True)
        self.monthLstm = nn.LSTM(
            input_size=arg.hidden, hidden_size=arg.hidden, batch_first=True)
        self.weekmonthattn = Parameter(
            torch.rand(size=(4,)), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)
        self.clf = nn.Linear(arg.hidden*4+arg.id_feature_size, 2)
        self.clf2 = nn.Linear(arg.hidden*4+arg.id_feature_size,arg.hidden)
        # self.fagcn=FAGCN()

    def forward(self, input_week, input_twoweek, input_threeweek, input_month, id_feature=None, g=None,): # return prediction and embedding
        input_week = torch.unsqueeze(input_week, dim=-1)
        input_twoweek = torch.unsqueeze(input_twoweek, dim=-1)
        input_threeweek = torch.unsqueeze(input_threeweek, dim=-1)
        input_month = torch.unsqueeze(input_month, dim=-1)
        input_week = torch.matmul(input_week, self.embedding)
        input_twoweek = input_twoweek@self.embedding
        input_threeweek = input_threeweek@self.embedding
        input_month = input_month@self.embedding
        input_week = torch.permute(input_week, dims=(
            0, 1, 3, 2))@self.softmax(self.feature_attn)
        input_twoweek = torch.permute(input_twoweek, dims=(
            0, 1, 3, 2))@self.softmax(self.feature_attn)
        input_threeweek = torch.permute(input_threeweek, dims=(
            0, 1, 3, 2))@self.softmax(self.feature_attn)
        input_month = torch.permute(input_month, dims=(
            0, 1, 3, 2))@self.softmax(self.feature_attn)
        output, (final_week, _) = self.weekLstm(input_week)
        output2, (final_twoweek, _) = self.weekLstm(input_twoweek)
        output3, (final_threeweek, _) = self.weekLstm(input_threeweek)
        output4, (final_month, _) = self.weekLstm(input_month)
        final_week = final_week[0, :, :]
        final_twoweek = final_twoweek[0, :, :]
        final_threeweek = final_threeweek[0, :, :]
        final_month = final_month[0, :, :]
        # final=torch.stack([final_week,final_twoweek,final_threeweek,final_month],dim=-1)@self.weekmonthattn
        final = torch.concat(
            [final_week, final_twoweek, final_threeweek, final_month, id_feature], dim=-1)
        final2 = self.clf(final)
        final2 = F.log_softmax(final2)
        # final = self.clf2(final)
        return final2, final   

        # return final

    def get_fea_attn(self):
        return self.softmax(self.feature_attn)

    def get_timeattn(self):
        return self.softmax(self.weekmonthattn)

class LSTMAdap(nn.Module):
    def __init__(self,args, diag_dimension, nhid, nlayer, nclass, dropout,edge_index,labels):
        super(LSTMAdap, self).__init__()

        self.lstm=SeveralLevelAttnLstm(args)

        self.adap=FAdaGNN(diag_dimension=diag_dimension,nfeat=args.hidden,nhid=nhid,nlayer=nlayer,nclass=nclass,dropout=dropout,edge_index=edge_index,labels=labels)

        self.gate=nn.Linear(args.hidden,nclass)


    def forward(self, oneweek, twoweek ,threeweek, month, id_feature, l_sym):

        pre1,hid=self.lstm(oneweek, twoweek ,threeweek, month, id_feature)

        pre=self.adap(hid,l_sym)
        
        

        return pre,pre1
    def getsigma_and_emb(self,oneweek, twoweek ,threeweek, month, id_feature):
        sigma = self.adap.getsigma()
        _, emb = self.lstm(oneweek,twoweek,threeweek,month,id_feature)
        return sigma, emb
    
class FAdaLayer(nn.Module):
    def __init__(self,arg, g, in_dim, dropout_rate, num_layer):
        super(FAdaLayer, self).__init__()
        self.lstm=SeveralLevelAttnLstm(arg=arg)
        self.g = g
        self.dropout = nn.Dropout(dropout_rate)
        self.num_layer = num_layer
        self.gate_lstm= nn.Linear(in_dim,arg.hidden)
        self.gate = nn.Linear(2*arg.hidden, arg.hidden)
        self.F=arg.hidden
        self.learnable_diag_1 = Parameter(torch.FloatTensor(32))  # in_features
        nn.init.xavier_normal_(self.gate_lstm.weight, gain=1.414)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)
        # nn.init.xavier_normal_(self.learnable_diag_1, gain=1.414)
        nn.init.normal_(self.learnable_diag_1)
        self.clf=nn.Linear(arg.hidden,2)

    def edge_applying(self, edges):
        h2 = torch.cat([edges.dst['h'],edges.src['h']], dim=1)  # two edges
        a = torch.tanh(self.gate(h2)/np.sqrt(2*self.F)) # first 
        e = edges.dst['d'] * edges.src['d']
        return {'e': e, "a":a}
    
    def message_func(self,edges):
        tmp = torch.mul(edges.src['h'],edges.data['a'])
        tmp = tmp * edges.data['e'].unsqueeze(1)
        return {'_':tmp}

    def forward(self,x):
        x1,x2,x3,x4,id_x=x
        pre2,h=self.lstm(x1,x2,x3,x4,id_x)
        h=self.gate_lstm(h)
        for i in range(self.num_layer):
            
            self.g.ndata['h'] = h
            self.g.apply_edges(self.edge_applying)
            
            self.g.update_all(self.message_func, fn.sum('_', 'z'))
        z=self.clf(self.g.ndata['z'])
        z=F.log_softmax(z,dim=-1)
        return z,pre2,self.g.edata['a']
    
class FAdaLayer_nosequence(nn.Module):
    def __init__(self,arg, g, in_dim, dropout_rate, num_layer):
        super(FAdaLayer_nosequence, self).__init__()
        
        self.g = g
        self.dropout = nn.Dropout(dropout_rate)
        self.num_layer = num_layer
        self.input_gate=nn.Linear(in_dim,arg.hidden)
        self.gate = nn.Linear(2*arg.hidden, arg.hidden)
        self.F=arg.hidden
        nn.init.xavier_normal_(self.input_gate.weight, gain=1.414)        
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)        
        self.clf=nn.Linear(arg.hidden,2)

    def edge_applying(self, edges):
        h2 = torch.cat([edges.dst['h'],edges.src['h']], dim=1)  # two edges
        a = torch.tanh(self.gate(h2)/np.sqrt(2*self.F)) # first 
        e = edges.dst['d'] * edges.src['d']
        return {'e': e, "a":a}
    
    def message_func(self,edges):
        tmp = torch.mul(edges.src['h'],edges.data['a'])
        tmp = tmp * edges.data['e'].unsqueeze(1)
        return {'_':tmp}

    def forward(self,x):
        h=self.input_gate(x)
        for i in range(self.num_layer):
            
            self.g.ndata['h'] = h
            self.g.apply_edges(self.edge_applying)
            
            self.g.update_all(self.message_func, fn.sum('_', 'z'))
        z=self.clf(self.g.ndata['z'])
        z=F.log_softmax(z,dim=-1)
        return z,self.g.edata['a']
    
class FAdaLayer_noAF(nn.Module):
    def __init__(self,arg, g, in_dim, dropout_rate, num_layer,nclass):
        super(FAdaLayer_noAF, self).__init__()
        self.lstm=SeveralLevelAttnLstm(arg=arg)
        self.g = g
        self.dropout = nn.Dropout(dropout_rate)
        self.num_layer = num_layer
        self.gate_lstm= nn.Linear(in_dim,arg.hidden)
        self.gcn=GraphConv(arg.hidden,nclass,)
        self.gate = nn.Linear(2*arg.hidden, arg.hidden)
        self.F=arg.hidden
        self.learnable_diag_1 = Parameter(torch.FloatTensor(32))  # in_features
        nn.init.xavier_normal_(self.gate_lstm.weight, gain=1.414)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)
        nn.init.normal_(self.learnable_diag_1)
        self.clf=nn.Linear(arg.hidden,2)

    def forward(self,x):
        x1,x2,x3,x4,id_x=x
        pre2,h=self.lstm(x1,x2,x3,x4,id_x)
        h=self.gate_lstm(h)
        z=self.gcn(self.g,h)
        z=F.log_softmax(z,dim=-1)
        return z,pre2


        


