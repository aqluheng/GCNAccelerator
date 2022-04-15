from cProfile import label
import torch
import torch.nn.functional as F

batchSize = 2
classNum = 3

Z = torch.rand((batchSize, classNum), requires_grad=True)
ReluZ = F.relu(Z-0.5)

labels = torch.randint(0, classNum, (batchSize,))
output = F.log_softmax(ReluZ, dim=1)
ReluZ.retain_grad()
output.retain_grad()

loss = F.nll_loss(output, labels, reduction="sum")
loss.backward(retain_graph=True)

print(torch.allclose(Z.grad,ReluZ.grad*(ReluZ>0)))
print(ReluZ.grad*(ReluZ>0))
print(Z.grad)