import torch
import torch.nn.functional as F

batchSize = 10
classNum = 7

labels = torch.randint(0, classNum, (batchSize,))

A = torch.rand((batchSize, batchSize))
X = torch.rand((batchSize, batchSize))
W = torch.rand((batchSize, classNum), requires_grad=True)
Y = torch.mm(A, torch.mm(X, W))
O = F.log_softmax(Y, dim=1)
Y.retain_grad()
O.retain_grad()


loss = F.nll_loss(O, labels, reduction="sum")
loss.backward(retain_graph=True)

myMMgrad = torch.mm(X.T, torch.mm(A.T, Y.grad))
print(torch.allclose(myMMgrad, W.grad))
