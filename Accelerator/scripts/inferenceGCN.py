import math
import torch
# 不仿功能, 只算cycle
# cycle计算需每一个稀疏矩阵 X=AXW0, Z=AHW1

dataset = "Nell"

def loadMatrix(modelName):
    mats = torch.load("./exportMat/%s.pt" % modelName)
    matA, matX, matH = mats["matA"], mats["matX"], mats["matH"]
    return matA.cpu().to_dense(), matX.cpu(), matH.cpu()

A, X, H = loadMatrix(dataset)


# 计算一次遍历spmm矩阵产生的乘法cycle
def multiplyCycleMM(spmm):
    spmm1 = (spmm != 0)
    mutiplyCycle = 0
    # n = math.ceil(spmm1.shape[0] / 16) + 1
    # for i in range(0, spmm1.shape[0], n):
        # mutiplyCycle = max(mutiplyCycle, torch.sum(spmm1[i:i+n]).item())
    

    # 一次处理2048 * 16
    for i in range(0,spmm.shape[0],2048):
        for j in range(0, spmm.shape[1], 16):
            # baseline : 只用一个GCNAX的乘法阵列,即只有一个PE
            # mutiplyCycle += torch.sum(spmm[i:i+2048,j:j+16] != 0).item() 

            # 计算方式1: 16个PE,每个处理一列, 坏处是不同的PE可能会同时写一行
            tmpMM = spmm1[i:i+2048,j:j+16]
            n = math.ceil(tmpMM.shape[0] / 16)
            tmpCycle = 0
            for i in range(0,tmpMM.shape[0],n):
                tmpCycle = max(tmpCycle, torch.sum(tmpMM[i:i+n]).item())
            mutiplyCycle += tmpCycle
            # print(tmpMM.shape)
            # mutiplyCycle += torch.max(torch.sum(spmm1[i:i+2048,j:j+16] != 0,axis = 0)).item() 

    return mutiplyCycle

def multiplyCycleSum():
    if dataset in ["Cora", "CiteSeer", "PubMed"]:
        return multiplyCycleMM(A) * 2 + multiplyCycleMM(X) + multiplyCycleMM(H)
    elif dataset == "Nell":
        return multiplyCycleMM(A) * 16 + multiplyCycleMM(X) * 4 + multiplyCycleMM(H) * 12

print(multiplyCycleSum())