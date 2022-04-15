from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import dataset

from utils import load_data, accuracy
from models import GCN

exportMat = False

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument("--dataset",type=str,default="Cora",help="Set dataset")


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.dataset in ["Cora","CiteSeer","PubMed"]:
    hiddenFeature = 16
elif args.dataset in ["Nell","Reddit"]:
    hiddenFeature = 64


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
print(len(idx_train), len(idx_val), len(idx_test))
# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=hiddenFeature,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    
    output = model(features, adj)
    lastTime = time.time()
    output = model(features, adj)
    print("Inference Time:", time.time()- lastTime)
    # output = model(features, adj, exportMat=exportMat,dataset=args.dataset)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train], reduction="sum")
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()

    if exportMat:
        gradMap = {}
        for name, parms in model.named_parameters():	
            print('-->name:', name, '-->grad_requirs:',parms.requires_grad, ' -->grad_value:',parms.grad.shape)
            gradMap[name+"_grad"] = parms.grad.cpu()
            gradMap[name] = parms.cpu()
        gradMap["Y"] = labels[idx_train]
        torch.save(gradMap,"exportMat/%s_grad.pt"%args.dataset)

    exit(0)
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj, exportMat=True,dataset=args.dataset)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# torch.save(model.state_dict(),"./myModel.pt")

# Testing
test()
