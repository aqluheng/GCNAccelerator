import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, exportMat=False, dataset=""):
        if exportMat:
            matA, matX = adj.cpu().clone(), x.cpu().clone()
        x = F.relu(self.gc1(x, adj))
        if exportMat:
            matH = x.cpu().clone()
            torch.save({"matA":matA,"matX":matX,"matH":matH},"exportMat/%s.pt"%dataset)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
