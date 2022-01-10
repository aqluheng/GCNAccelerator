import math
import torch
# 不仿功能, 只算cycle
# cycle计算需每一个稀疏矩阵 X=AXW0, Z=AHW1

dataset = "Reddit"

def loadMatrix(modelName):
    mats = torch.load("./exportMat/%s.pt" % modelName)
    matA, matX, matH = mats["matA"], mats["matX"], mats["matH"]
    return matA.coalesce(), matX, matH


A, X, H = loadMatrix(dataset)


def getSpmmSliceCount(spmm, sliceX, sliceY):
    indexList = spmm.indices()
    sliceOne = torch.bitwise_and(sliceX[0]<=indexList[0],indexList[0]<sliceX[1])
    sliceTwo = torch.bitwise_and(sliceY[0]<=indexList[1],indexList[1]<sliceY[1])
    return torch.sum(torch.bitwise_and(sliceOne,sliceTwo))
    
# for i in range(500):
#     print(getSpmmSliceCount(A,(0,2048),(0,16)))
# exit(0)

# 计算一次遍历spmm矩阵产生的乘法cycle
def multiplyCycleAB(spmm):
    mutiplyCycle = 0
    # 一次处理2048 * 16
    for i in range(0,spmm.shape[0],2048):
        for j in range(0, spmm.shape[1], 16):
            # baseline : 只用一个GCNAX的乘法阵列,即只有一个PE
            mutiplyCycle += getSpmmSliceCount(spmm,(i,i+2048),(j,j+16)).item() 

            # 计算方式1: 16个PE,每个处理一列, 坏处是不同的PE可能会同时写一行            
            # mutiplyCycle += torch.max(torch.sum(spmm[i:i+2048,j:j+16] != 0,axis = 0)).item() 

    return mutiplyCycle

# 计算一次遍历spmm矩阵产生的乘法cycle
def multiplyCycleXW(spmm):
    mutiplyCycle = 0
    # 一次处理2048 * 16
    for i in range(0,spmm.shape[0],2048):
        for j in range(0, spmm.shape[1], 16):
            # baseline : 只用一个GCNAX的乘法阵列,即只有一个PE
            mutiplyCycle += torch.sum(spmm[i:i+2048,j:j+16] != 0).item() 

            # 计算方式1: 16个PE,每个处理一列, 坏处是不同的PE可能会同时写一行            
            # mutiplyCycle += torch.max(torch.sum(spmm[i:i+2048,j:j+16] != 0,axis = 0)).item() 

    return mutiplyCycle

def multiplyCycleSum():
    if dataset == "Reddit":
        return multiplyCycleAB(A) * 7 + multiplyCycleXW(X) * 4 + multiplyCycleXW(H) * 3

print(multiplyCycleSum())