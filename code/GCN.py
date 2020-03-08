# -*- coding: utf-8 -*-
"""
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from graphconvolution import GraphConvolution,GraphAttention

class GCN(nn.Module):
    
    def __init__(self, nfeat, nhid, nclass, dropout, init):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, init=init)
        self.gc2 = GraphConvolution(nhid, nclass, init=init)
        self.dropout = dropout

    def bottleneck(self, path1, path2, path3, adj, in_x):
        return F.relu(path3(F.relu(path2(F.relu(path1(in_x, adj)), adj)), adj))

    def forward(self, x, adj):
        x = F.dropout(F.relu(self.gc1(x, adj)), self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return F.log_softmax(x, dim=1)
    

class GAT(nn.Module):
    
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttention(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttention(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
    

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train(model,optimizer,features,adj,labels,epoch):
    
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    train_size = 1500
    test_size = 494
    idx_train = range(train_size)
    loss_train = F.nll_loss(output[idx_train],labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

  
    model.eval()
    output = model(features, adj)
    idx_val = range(train_size,train_size+test_size)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
