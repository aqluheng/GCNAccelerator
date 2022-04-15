import torch
from torch import Tensor
from models import GraphSage
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from utils import accuracy
import time


dataset = Planetoid(root='/home/luheng/GCNTraining/datasets/Cora', name='Cora')
model = GraphSage(dataset.num_features, 16, dataset.num_classes)


data = dataset[0]
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    t = time.time()
    model.train()
    pred = model(data.x, data.edge_index)
    exit()
    loss_train = F.nll_loss(pred[data.train_mask], data.y[data.train_mask], reduction="sum")
    acc_train = accuracy(pred[data.train_mask], data.y[data.train_mask])

    # Backpropagation
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    model.eval()
    output = model(data.x, data.edge_index)

    loss_val = F.nll_loss(pred[data.val_mask], data.y[data.val_mask])
    acc_val = accuracy(pred[data.val_mask], data.y[data.val_mask])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))