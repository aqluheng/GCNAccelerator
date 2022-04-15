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
        Z0 = self.gc1(x, adj)
        H0 = F.relu(Z0)
        dropH0 = F.dropout(H0, self.dropout, training=self.training)
        Z1 = self.gc2(dropH0, adj)
        H1 = F.log_softmax(Z1, dim=1)

        if exportMat:
            matA, matX = adj.cpu(), x.cpu()
            matZ0, matH0 = Z0.cpu(), H0.cpu()
            matZ1, matH1 = Z1.cpu(), H1.cpu()
            matDropH0 = dropH0.cpu()
            # print(torch.allclose(Z1, torch.mm(adj,torch.mm(dropH0,self.gc2.weight)+self.gc2.bias)))
            tmpDict = {"A": matA, "X": matX, "Z0": matZ0, "H0": matH0, "Z1": matZ1, "H1": matH1, "dropH0": matDropH0}
            torch.save(tmpDict, "exportMat/%s.pt" % dataset)

        return H1
