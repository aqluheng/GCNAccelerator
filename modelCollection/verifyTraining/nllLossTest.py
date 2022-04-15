import torch
import torch.nn.functional as F

batchSize = 10
classNum = 7

Z = torch.rand((batchSize, classNum), requires_grad=True)
labels = torch.randint(0, classNum, (batchSize,))
output = F.log_softmax(Z, dim=1)
output.retain_grad()

loss = F.nll_loss(output, labels, reduction="sum")
loss.backward(retain_graph=True)


labels_onehot = torch.zeros(batchSize, classNum).scatter_(1, labels.view((-1,1)), 1)
myGrad = F.softmax(Z, dim=1) - labels_onehot

print(torch.allclose(myGrad,Z.grad))