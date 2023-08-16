from model_source_adap.models import AdaGNN,LSTMAdap
from model_source_adap.pre_train_2_layer import pre_train,pre_train2
import argparse
import os
import time
from sklearn.metrics import roc_auc_score, recall_score, f1_score

parser=argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--layers', type=int, default=8,
                    help='Layer number of AdaGNN model.')
parser.add_argument('--mode', type=str, default='s',
                    help='Regularization of adjacency matrix in {"r", "s"}.')

parser.add_argument('--dataset', type=str, default='Sichuan',
                    help='dataset from {"BlogCatalog", "Flickr", "ACM", "cora", "citeseer", "pubmed", "Sichuan", "Amazon"}.')

parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epoches', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,  # 0.0001
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=9e-6,  # 9e-12
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--l1norm', type=float, default=1e-6,  # 1e-6
                    help='L1 loss on Phi in each layer.')
parser.add_argument('--num_features',default=32)
parser.add_argument('--hidden', type=int, default=4,  # 16
                    help='Number of hidden units.')
parser.add_argument('--id_feature_size',default=2)
parser.add_argument('--dropout', type=float, default=0.1,  # 0.2
                    help='Dropout rate (1 - keep probability).')

parser.add_argument('--gamma',type=float,default=0.5)

parser.add_argument('--loss_gamma',type=float,default=0.6)
parser.add_argument('--patience', type=int, default=6)

parser.add_argument('--train_p',default=0.4)
parser.add_argument('--val_p',default=0.2)

parser.add_argument('--train',default=True)
parser.add_argument('--test',default=True)
parser.add_argument('--analysis',default=False)


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']='0'


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model_source_adap.utils import *
from model_source_adap.models import AdaGNN
from model_source_adap.pre_train_2_layer import pre_train
import sys
import numpy as np

args.cuda = not args.no_cuda and torch.cuda.is_available()

print(sys.argv[0])

np.random.seed(args.seed)
torch.manual_seed(args.seed)


if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.dataset == "Amazon":
    load_Amazon(args,)
    adj, oneweek, twoweek ,threeweek, month, id_feature, labels, idx_train, idx_val, idx_test = load_Sichuanmini_static(args)

    model = LSTMAdap(args=args,diag_dimension=labels.shape[0],nhid=args.hidden,nlayer=args.layers,nclass=labels.max().item()+1,dropout=args.dropout)

    optimier=optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model=model.cuda()
        oneweek=oneweek.cuda()
        twoweek=twoweek.cuda()
        threeweek=threeweek.cuda()
        month=month.cuda()
        adj=adj.cuda()
        id_feature=id_feature.cuda()
        labels=labels.cuda()


    # if args.layers>2:
    #     pre_train2(args.cuda, args.dataset, args.hidden, adj, oneweek, twoweek, threeweek, month, id_feature, labels, idx_train, idx_val, gamma)

    stop_count = 0
    val_loss_final = 0
    last_loss = 1000


    def train(epoch,optimizer,oneweek, twoweek, threeweek, month, id_feature,adj ,patience):
        global val_loss_final
        global stop_count
        global last_loss

        t = time.time()
        the_l1 = 0

        for k, v in model.named_parameters():
            if 'learnable_diag' in k:
                the_l1 += torch.sum(abs(v))

        model.train()
        optimizer.zero_grad()
        output,output2 = model(oneweek, twoweek, threeweek, month, id_feature, adj)

        loss_indiv=F.nll_loss(output2[idx_train],labels[idx_train])
        loss_train = F.nll_loss(output[idx_train], labels[idx_train]) + args.l1norm * the_l1
        loss_train = args.loss_gamma*loss_train + (1-args.loss_gamma) * loss_indiv
        acc_train = accuracy(output[idx_train], labels[idx_train])

        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            model.eval()
            output,output2 = model(oneweek, twoweek, threeweek, month, id_feature, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val]) + args.l1norm * the_l1
        acc_val = accuracy(output[idx_val], labels[idx_val])

        if loss_val.item() > last_loss:
            stop_count += 1
        else:
            stop_count = 0
        last_loss = loss_val.item()

        if epoch == 0:
            val_loss_final = loss_val.item()
        elif loss_val.item() < val_loss_final:
            val_loss_final = loss_val.item()
            torch.save(model.state_dict(), 'model_store/'+'lstm_adap-'+args.dataset + '-' + str(args.layers) + '.pkl')

        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),
            'time: {:.4f}s'.format(time.time() - t))

        if stop_count >= patience:  # 6
            print("Early stop  ! ")

    def test():
        global labels
        try:
            model.load_state_dict(torch.load('model_store/'+'lstm_adap-'+args.dataset + '-' + str(args.layers) + '.pkl'),
                                strict=True)
        except FileNotFoundError:
            model.load_state_dict(torch.load('model_store/'+'lstm_adap-'+args.dataset + '-' + str(2) + '.pkl'),
                                strict=False)
        model.eval()
        output,output2 = model(oneweek,twoweek,threeweek,month,id_feature, adj)
        if args.dataset != 'citeseer':
            loss_test = F.nll_loss(output[idx_test], labels[idx_test])
            acc_test = accuracy(output[idx_test], labels[idx_test])
            output=output.detach().cpu().numpy()
            labels=labels.cpu().numpy()
            y_pre=np.argmax(output,axis=1)
            # labels=np.argmax(labels,axis=1)
            y_score=output[:,-1]
            auc_test = roc_auc_score(labels[idx_test],y_score[idx_test],average='macro')
            recall_test= recall_score(labels[idx_test],y_pre[idx_test],average='macro')
            f1_test = f1_score(labels[idx_test], y_pre[idx_test], average='macro')
        if args.dataset == 'citeseer':
            loss_test = F.nll_loss(output[idx_test], labels[-1000:])
            acc_test = accuracy(output[idx_test], labels[-1000:])
        print("Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()),"auc={:.4f}".format(auc_test),
            "recall={:.4f}".format(recall_test),
            "f1_score={:.4f}".format(f1_test))
    def analysis():
        global labels
        try:
            model.load_state_dict(torch.load('model_store/'+'lstm_adap-'+args.dataset + '-' + str(args.layers) + '.pkl'),
                                strict=True)
        except FileNotFoundError:
            model.load_state_dict(torch.load('model_store/'+'lstm_adap-'+args.dataset + '-' + str(2) + '.pkl'),
                                strict=False)
        model.eval()
        
        if args.dataset != 'citeseer':
            sigma, emb=model.getsigma_and_emb(oneweek,twoweek,threeweek,month,id_feature)
            
        if args.dataset == 'citeseer':
            pass
    
    if args.train:
        for epoch in range(args.epoches):
            train(epoch,optimier,oneweek, twoweek, threeweek, month, id_feature,adj ,args.patience)
        print("Optimization Finished!")
    
    if args.test:
        test()
    if args.analysis:
        analysis()  

if args.dataset == "Sichuan_1":
    edge_index,features, labels, idx_train, idx_val, idx_test = load_Sichuan_1(args)

    model = LSTMAdap(args=args,diag_dimension=labels.shape[0],nhid=args.hidden,nlayer=args.layers,nclass=labels.max().item()+1,dropout=args.dropout)

    

    if args.cuda:
        model=model.cuda()
        oneweek=oneweek.cuda()
        twoweek=twoweek.cuda()
        threeweek=threeweek.cuda()
        month=month.cuda()
        adj=adj.cuda()
        id_feature=id_feature.cuda()
        labels=labels.cuda()

    optimier=optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # if args.layers>2:
    #     pre_train2(args.cuda, args.dataset, args.hidden, adj, oneweek, twoweek, threeweek, month, id_feature, labels, idx_train, idx_val, gamma)

    stop_count = 0
    val_loss_final = 0
    last_loss = 1000


    def train(epoch,optimizer,oneweek, twoweek, threeweek, month, id_feature,adj ,patience):
        global val_loss_final
        global stop_count
        global last_loss

        t = time.time()
        the_l1 = 0

        for k, v in model.named_parameters():
            if 'learnable_diag' in k:
                the_l1 += torch.sum(abs(v))

        model.train()
        optimizer.zero_grad()
        output,output2 = model(oneweek, twoweek, threeweek, month, id_feature, adj)

        loss_indiv=F.nll_loss(output2[idx_train],labels[idx_train])
        loss_train = F.nll_loss(output[idx_train], labels[idx_train]) + args.l1norm * the_l1
        loss_train = args.loss_gamma*loss_train + (1-args.loss_gamma) * loss_indiv
        acc_train = accuracy(output[idx_train], labels[idx_train])

        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            model.eval()
            output,output2 = model(oneweek, twoweek, threeweek, month, id_feature, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val]) + args.l1norm * the_l1
        acc_val = accuracy(output[idx_val], labels[idx_val])

        if loss_val.item() > last_loss:
            stop_count += 1
        else:
            stop_count = 0
        last_loss = loss_val.item()

        if epoch == 0:
            val_loss_final = loss_val.item()
        elif loss_val.item() < val_loss_final:
            val_loss_final = loss_val.item()
            torch.save(model.state_dict(), 'model_store/'+'lstm_adap-'+args.dataset + '-' + str(args.layers) + '.pkl')

        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),
            'time: {:.4f}s'.format(time.time() - t))

        if stop_count >= patience:  # 6
            print("Early stop  ! ")

    def test():
        global labels
        try:
            model.load_state_dict(torch.load('model_store/'+'lstm_adap-'+args.dataset + '-' + str(args.layers) + '.pkl'),
                                strict=True)
        except FileNotFoundError:
            model.load_state_dict(torch.load('model_store/'+'lstm_adap-'+args.dataset + '-' + str(2) + '.pkl'),
                                strict=False)
        model.eval()
        output,output2 = model(oneweek,twoweek,threeweek,month,id_feature, adj)
        if args.dataset != 'citeseer':
            loss_test = F.nll_loss(output[idx_test], labels[idx_test])
            acc_test = accuracy(output[idx_test], labels[idx_test])
            output=output.detach().cpu().numpy()
            labels=labels.cpu().numpy()
            y_pre=np.argmax(output,axis=1)
            # labels=np.argmax(labels,axis=1)
            y_score=output[:,-1]
            auc_test = roc_auc_score(labels[idx_test],y_score[idx_test],average='macro')
            recall_test= recall_score(labels[idx_test],y_pre[idx_test],average='macro')
            f1_test = f1_score(labels[idx_test], y_pre[idx_test], average='macro')
        if args.dataset == 'citeseer':
            loss_test = F.nll_loss(output[idx_test], labels[-1000:])
            acc_test = accuracy(output[idx_test], labels[-1000:])
        print("Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()),"auc={:.4f}".format(auc_test),
            "recall={:.4f}".format(recall_test),
            "f1_score={:.4f}".format(f1_test))
    def analysis():
        global labels
        try:
            model.load_state_dict(torch.load('model_store/'+'lstm_adap-'+args.dataset + '-' + str(args.layers) + '.pkl'),
                                strict=True)
        except FileNotFoundError:
            model.load_state_dict(torch.load('model_store/'+'lstm_adap-'+args.dataset + '-' + str(2) + '.pkl'),
                                strict=False)
        model.eval()
        
        if args.dataset != 'citeseer':
            sigma, emb=model.getsigma_and_emb(oneweek,twoweek,threeweek,month,id_feature)
            
        if args.dataset == 'citeseer':
            pass
    
    if args.train:
        for epoch in range(args.epoches):
            train(epoch,optimier,oneweek, twoweek, threeweek, month, id_feature,adj ,args.patience)
        print("Optimization Finished!")
    
    if args.test:
        test()
    if args.analysis:
        analysis()    

if args.dataset == 'Sichuan':
    # 加载数据集
    adj, edge_index, oneweek, twoweek ,threeweek, month, id_feature, labels, idx_train, idx_val, idx_test = load_Sichuan(args)

    model = LSTMAdap(args=args,diag_dimension=labels.shape[0],nhid=args.hidden,nlayer=args.layers,nclass=labels.max().item()+1,dropout=args.dropout,edge_index=edge_index,labels=labels)

    optimier=optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        
        
        oneweek=oneweek.cuda()
        twoweek=twoweek.cuda()
        threeweek=threeweek.cuda()
        month=month.cuda()
        adj=adj.cuda()
        id_feature=id_feature.cuda()
        labels=labels.cuda()
        edge_index = edge_index.cuda()
        model = model = LSTMAdap(args=args,diag_dimension=labels.shape[0],nhid=args.hidden,nlayer=args.layers,nclass=labels.max().item()+1,dropout=args.dropout,edge_index=edge_index,labels=labels)
        model=model.cuda()
    # if args.layers>2:
    #     pre_train2(args.cuda, args.dataset, args.hidden, adj, oneweek, twoweek, threeweek, month, id_feature, labels, idx_train, idx_val, gamma)

    stop_count = 0
    val_loss_final = 0
    last_loss = 1000


    def train(epoch,optimizer,oneweek, twoweek, threeweek, month, id_feature,adj ,patience):
        global val_loss_final
        global stop_count
        global last_loss

        t = time.time()
        the_l1 = 0

        for k, v in model.named_parameters():
            if 'learnable_diag' in k:
                the_l1 += torch.sum(abs(v))

        model.train()
        optimizer.zero_grad()
        output,output2 = model(oneweek, twoweek, threeweek, month, id_feature, adj)

        loss_indiv=F.nll_loss(output2[idx_train],labels[idx_train])
        loss_train = F.nll_loss(output[idx_train], labels[idx_train]) + args.l1norm * the_l1
        loss_train = args.loss_gamma*loss_train + (1-args.loss_gamma) * loss_indiv
        acc_train = accuracy(output[idx_train], labels[idx_train])

        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            model.eval()
            output,output2 = model(oneweek, twoweek, threeweek, month, id_feature, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val]) + args.l1norm * the_l1
        acc_val = accuracy(output[idx_val], labels[idx_val])

        if loss_val.item() > last_loss:
            stop_count += 1
        else:
            stop_count = 0
        last_loss = loss_val.item()

        if epoch == 0:
            val_loss_final = loss_val.item()
        elif loss_val.item() < val_loss_final:
            val_loss_final = loss_val.item()
            torch.save(model.state_dict(), 'model_store/'+'lstm_adap-'+args.dataset + '-' + str(args.layers) + '.pkl')

        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),
            'time: {:.4f}s'.format(time.time() - t))

        if stop_count >= patience:  # 6
            print("Early stop  ! ")

    def test():
        global labels
        try:
            model.load_state_dict(torch.load('model_store/'+'lstm_adap-'+args.dataset + '-' + str(args.layers) + '.pkl'),
                                strict=True)
        except FileNotFoundError:
            model.load_state_dict(torch.load('model_store/'+'lstm_adap-'+args.dataset + '-' + str(2) + '.pkl'),
                                strict=False)
        model.eval()
        output,output2 = model(oneweek,twoweek,threeweek,month,id_feature, adj)
        if args.dataset != 'citeseer':
            loss_test = F.nll_loss(output[idx_test], labels[idx_test])
            acc_test = accuracy(output[idx_test], labels[idx_test])
            output=output.detach().cpu().numpy()
            labels=labels.cpu().numpy()
            y_pre=np.argmax(output,axis=1)
            # labels=np.argmax(labels,axis=1)
            y_score=output[:,-1]
            auc_test = roc_auc_score(labels[idx_test],y_score[idx_test],average='macro')
            recall_test= recall_score(labels[idx_test],y_pre[idx_test],average='macro')
            f1_test = f1_score(labels[idx_test], y_pre[idx_test], average='macro')
        if args.dataset == 'citeseer':
            loss_test = F.nll_loss(output[idx_test], labels[-1000:])
            acc_test = accuracy(output[idx_test], labels[-1000:])
        print("Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()),"auc={:.4f}".format(auc_test),
            "recall={:.4f}".format(recall_test),
            "f1_score={:.4f}".format(f1_test))
    def analysis():
        global labels
        try:
            model.load_state_dict(torch.load('model_store/'+'lstm_adap-'+args.dataset + '-' + str(args.layers) + '.pkl'),
                                strict=True)
        except FileNotFoundError:
            model.load_state_dict(torch.load('model_store/'+'lstm_adap-'+args.dataset + '-' + str(2) + '.pkl'),
                                strict=False)
        model.eval()
        
        if args.dataset != 'citeseer':
            sigma, emb=model.getsigma_and_emb(oneweek,twoweek,threeweek,month,id_feature)
            
        if args.dataset == 'citeseer':
            pass
    
    if args.train:
        for epoch in range(args.epoches):
            train(epoch,optimier,oneweek, twoweek, threeweek, month, id_feature,adj ,args.patience)
        print("Optimization Finished!")
    
    if args.test:
        test()
    if args.analysis:
        analysis()

    





