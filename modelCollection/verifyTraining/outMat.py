import torch
import torch.nn.functional as F
from time import time
import os
from os.path import join
# import torch.sparse. as spmm

datasets = ["Cora", "CiteSeer", "PubMed", "Nell", "Reddit"]
# 选择数据集

dataset = datasets[3]
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


with torch.no_grad():
    myXW = torch.mm(gtX, gtW0)
    myZ0 = torch.mm(gtA, myXW)

    myHW = torch.mm(dropH0, gtW1)
    myZ1 = torch.mm(gtA, myHW)

    gtZ1 = gtZ1.cpu()
    labels_onehot = torch.zeros(gtZ1.shape[0], gtlabels.max()+1).scatter_(1, gtlabels.view((-1, 1)), 1)
    Z1grad = F.softmax(gtZ1, dim=1) - labels_onehot

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

    ATZ1grad = torch.mm(gtA.T, Z1grad)
    W1grad = torch.mm(dropH0.T, ATZ1grad)

    B1grad = Z1grad.sum(dim=0)

    H0grad = torch.mm(ATZ1grad, gtW1.T)

    Z0grad = H0grad * (dropH0 > 0) * 2

    ATZ0grad = torch.mm(gtA.T, Z0grad)

    W0grad = torch.mm(gtX.T, ATZ0grad)

    B0grad = Z0grad.sum(dim=0)

    print("当前数据集:", dataset)

    myZ1 = torch.spmm(spA, torch.mm(dropH0, gtW1)) + gtB1
    print("Z0正确性", torch.allclose(torch.spmm(spA, torch.mm(gtX, gtW0))+gtB0, gtZ0))
    print("Z1正确性", torch.max((myZ1 - gtZ1)).abs())
    print("W1梯度正确性", torch.max((W1grad - gtW1grad)).abs())
    print("W0梯度正确性", torch.max((W0grad - gtW0grad)).abs())
    print("B1梯度正确性", torch.allclose(B1grad, gtB1grad))
    print("B0梯度正确性", torch.allclose(B0grad, gtB0grad))

    outputPath = join("outputs/", dataset)
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    os.system("rm -f %s/*" % outputPath)

    # inference phase 1 A*(X*W)
    leftMats = [gtX, gtA]
    rightMats = [gtW0, myXW]
    phase1Dict = {"leftMats": leftMats, "rightMats": rightMats, "annotation": "inference phase 1 A*(X*W)"}
    torch.save(phase1Dict, join(outputPath, "phase1Dict.pt"))

    # inference phase 2 A*(H*W)
    leftMats = [dropH0, gtA]
    rightMats = [gtW1, myHW]
    phase2Dict = {"leftMats": leftMats, "rightMats": rightMats, "annotation": "inference phase 2 A*(H*W)"}
    torch.save(phase2Dict, join(outputPath, "phase2Dict.pt"))

    # backward phase 3 dropH0T*(AT*Z1grad) and (AT*Z1grad)*W1T
    leftMats = [gtA.T, dropH0.T, ATZ1grad]
    rightMats = [Z1grad, ATZ1grad, gtW1.T]
    phase3Dict = {"leftMats": leftMats, "rightMats": rightMats,
                  "annotation": "backward phase 3 dropH0T*(AT*Z1grad) and (AT*Z1grad)*W1T"}
    torch.save(phase3Dict, join(outputPath, "phase3Dict.pt"))

    # backward phase 4 XT*(AT*Z0grad)
    leftMats = [gtA.T, gtX.T]
    rightMats = [Z0grad, ATZ0grad]
    phase4Dict = {"leftMats": leftMats, "rightMats": rightMats,
                  "annotation": "backward phase 4 XT*(AT*Z0grad)"}
    torch.save(phase4Dict, join(outputPath, "phase4Dict.pt"))
