from numpy import mod
import torch
from models import GCN
from utils import load_data, accuracy
import torch.nn.functional as F

adj, features, labels, idx_train, idx_val, idx_test = load_data()
model = GCN(nfeat=features.shape[1],
            nhid=16,
            nclass=labels.max().item() + 1,
            dropout=0.5)

model.load_state_dict(torch.load("/home/luheng/GCNtraining/modelCollection/GCN/pygcn/myModel.pt"))
model.eval()
output = model(features, adj)

print(features)
loss_test = F.nll_loss(output[idx_test], labels[idx_test])
acc_test = accuracy(output[idx_test], labels[idx_test])
print("Test set results:",
        "loss= {:.4f}".format(loss_test.item()),
        "accuracy= {:.4f}".format(acc_test.item()))