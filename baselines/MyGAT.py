import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
from dgl.nn.pytorch import GATConv
import torch.nn.functional as F

class MyGAT(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        self.mygat1=GATConv(args.in_feats,args.hidden_size,args.num_heads)
        self.clf=nn.Linear(args.hidden_size*args.num_heads,args.n_classes)
        self.dropout=nn.Dropout(0.5)
    def forward(self,g,X):
        h=self.mygat1(g,X).flatten(1)
        h=F.relu(h)
        h=self.dropout(h)
        h=self.clf(h)
        h=F.sigmoid(h)
        return h
    

