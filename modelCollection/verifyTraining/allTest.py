import torch
import torch.nn.functional as F
from time import time
# import torch.sparse. as spmm

datasets = ["Cora", "CiteSeer", "PubMed", "Nell", "Reddit"]
# 选择数据集

dataset = datasets[0]
isCuda = False
repeatCount = 1


def printDensity(Mat, name):
    matLen = 1
    for i in Mat.shape:
        matLen = matLen * i
    rowLen = (Mat != 0).sum(dim=1)
    zeroRate = (rowLen == 0).sum() / rowLen.shape[0]
    nonzeroRate = rowLen.sum() / Mat.shape[1] / (rowLen != 0).sum()
    print(name, (Mat != 0).sum()/matLen, Mat.shape, "全零行比例:", zeroRate, "非零行密度:", nonzeroRate)


gradDict = torch.load("%s_grad.pt" % dataset)
forwardDict = torch.load("%s.pt" % dataset)
gtW0grad = gradDict["gc1.weight_grad"]
gtW1grad = gradDict["gc2.weight_grad"]
gtB0grad = gradDict["gc1.bias_grad"]
gtB1grad = gradDict["gc2.bias_grad"]
# 最终目标
gtW0 = gradDict["gc1.weight"]
gtW1 = gradDict["gc2.weight"]
gtB0 = gradDict["gc1.bias"]
gtB1 = gradDict["gc2.bias"]
spA = forwardDict["A"]
gtA = forwardDict["A"].to_dense()
gtX = forwardDict["X"]
gtlabels = gradDict["Y"].cpu()
# 初始权重
gtZ0 = forwardDict["Z0"]
gtH0 = forwardDict["H0"]
gtZ1 = forwardDict["Z1"]
gtH1 = forwardDict["H1"]
dropH0 = forwardDict["dropH0"]
# 中间变量
print(gtX.shape[0], gtX.shape[1], gtH0.shape[1], gtlabels.max().data+1)
if isCuda:
    spA = spA.cuda()
    gtX = gtX.cuda()
    gtB0 = gtB0.cuda()
    gtW0 = gtW0.cuda()
    gtB1 = gtB1.cuda()
    gtW1 = gtW1.cuda()
    gtH1 = gtH1.cuda()
    gtH0 = gtH0.cuda()
    gtZ0 = gtZ0.cuda()
    gtZ1 = gtZ1.cuda()
    gtW0grad = gtW0grad.cuda()
    gtW1grad = gtW1grad.cuda()
    gtB0grad = gtB0grad.cuda()
    gtB1grad = gtB1grad.cuda()
    dropH0 = dropH0.cuda()


def transSparse(X_dense):
    idx = torch.nonzero(X_dense)
    data = X_dense[idx[:, 0], idx[:, 1]]
    return torch.sparse.FloatTensor(idx.T, data, X_dense.shape)
    # return torch.sparse.FloatTensor(idx,data,torch.Size([2,data.shape[0]]))


with torch.no_grad():
    spX = transSparse(gtX)

    tChainMM0 = time()
    for i in range(repeatCount):
        myZ0 = torch.spmm(spA, torch.spmm(gtX, gtW0))
    tChainMM0 = time() - tChainMM0
    # 第一个矩阵连乘的时间

    myH0 = F.relu(myZ0 + gtB0)

    # spdropH0 = transSparse(dropH0)
    spdropH0 = transSparse(dropH0)
    tChainMM1 = time()
    for i in range(repeatCount):
        myZ1 = torch.spmm(spA, torch.spmm(spdropH0, gtW1))
    tChainMM1 = time() - tChainMM1
    # 第二个矩阵连乘的时间

    for i in range(repeatCount):
        myH1 = F.log_softmax(myZ1 + gtB1, dim=1)

    gtZ1 = gtZ1.cpu()
    labels_onehot = torch.zeros(gtZ1.shape[0], gtlabels.max()+1).scatter_(1, gtlabels.view((-1, 1)), 1)
    tSoftmax = time()
    for i in range(repeatCount):
        Z1grad = F.softmax(gtZ1, dim=1) - labels_onehot
    tSoftmax = time() - tSoftmax

    trainIdx = {"Cora": 140, "CiteSeer": 120, "PubMed": 60, "Nell": 105}
    if dataset == "Reddit":
        idx_train = torch.load("RedditIdx.pt")["idx_train"]
        tmpZ1grad = Z1grad.clone()
        Z1grad[:] = 0
        Z1grad[idx_train] = tmpZ1grad[idx_train]
    else:
        Z1grad[trainIdx[dataset]:, ] = 0
    if isCuda:
        Z1grad = Z1grad.cuda()
        gtZ1 = gtZ1.cuda()

    tChainMM2 = time()
    for i in range(repeatCount):
        W1grad = torch.spmm(spdropH0.transpose(0, 1), torch.spmm(spA.transpose(0, 1), Z1grad))
    tChainMM2 = time() - tChainMM2
    # 第三个矩阵连乘的时间

    tVecSum0 = time()
    for i in range(repeatCount):
        B1grad = Z1grad.sum(dim=0)
    tVecSum0 = time() - tVecSum0

    ATZ1grad = torch.spmm(spA.transpose(0, 1), Z1grad)
    spATZ1grad = transSparse(ATZ1grad)

    tChainMM3 = time()
    for i in range(repeatCount):
        H0grad = torch.spmm(spATZ1grad, gtW1.T)
    tChainMM3 = time() - tChainMM3
    # 第四个矩阵乘的时间

    tRelu = time()
    for i in range(repeatCount):
        Z0grad = H0grad * (dropH0 > 0) * 2
    tRelu = time() - tRelu

    ATZ0grad = torch.spmm(spA.transpose(0, 1), Z0grad)

    tChainMM4 = time()
    for i in range(repeatCount):
        W0grad = torch.spmm(spX.transpose(0, 1), torch.spmm(spA.transpose(0, 1), Z0grad))
    tChainMM4 = time() - tChainMM4
    # 第五个矩阵连乘的时间

    tVecSum1 = time()
    for i in range(repeatCount):
        B0grad = Z0grad.sum(dim=0)
    tVecSum1 = time() - tVecSum1

    print("当前数据集:", dataset)
    printDensity(gtA.T, "AT")
    printDensity(dropH0.T, "dropH0T")
    printDensity(gtX, "X")
    printDensity(gtX.T, "XT")
    printDensity(gtW0, "W0")
    printDensity(gtW1, "W1")
    printDensity(dropH0, "dropH0")
    printDensity(Z1grad, "Z1grad")
    printDensity(Z0grad, "Z0grad")
    printDensity(ATZ1grad, "AT*Z1grad")
    printDensity(ATZ0grad, "AT*Z0grad")
    printDensity(torch.mm(gtX, gtW0), "XW0")
    printDensity(torch.mm(dropH0, gtW1), "H0W1")
    printDensity(H0grad, "H0grad")
    printDensity(W0grad, "W0grad")
    printDensity(W1grad, "W1grad")

    print(tChainMM0/repeatCount, tChainMM1/repeatCount, tChainMM2 /
          repeatCount, tChainMM3/repeatCount, tChainMM4/repeatCount)
    print("激活函数时间", tSoftmax/repeatCount, tRelu/repeatCount)
    print("向量和时间", tVecSum0/repeatCount, tVecSum1/repeatCount)
    # '''
    myZ1 = torch.spmm(spA, torch.mm(dropH0, gtW1)) + gtB1
    print("Z0正确性", torch.allclose(torch.spmm(spA, torch.mm(gtX, gtW0))+gtB0, gtZ0))
    print("Z1正确性", torch.max((myZ1 - gtZ1)).abs())

    print(gtW0grad.shape, gtW1grad.shape)
    print(W0grad.shape, W1grad.shape)

    print("W1梯度正确性", torch.max((W1grad - gtW1grad)).abs())
    print("W0梯度正确性", torch.max((W0grad - gtW0grad)).abs())
    print("B1梯度正确性", torch.allclose(B1grad, gtB1grad))
    print("B0梯度正确性", torch.allclose(B0grad, gtB0grad))
    # '''

    mat = gtX
    rowlist = (mat != 0).sum(dim=1)
    print((rowlist == 0).sum())
    print((rowlist[rowlist != 0]).float().mean()/mat.shape[1])

    gtA
    gtX
    gtW0
    gtH0
    gtW1

    XW0 = torch.mm(gtX, gtW0)
    