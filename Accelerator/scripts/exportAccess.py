import math
import torch
from os.path import join
import os
from math import ceil

# 不仿功能, 只算cycle
# cycle计算需每一个稀疏矩阵 X=AXW0, Z=AHW1

dataset = "CiteSeer"


def loadMatrix(modelName):
    mats = torch.load("./exportMat/%s.pt" % modelName)
    matA, matX, matH = mats["matA"], mats["matX"], mats["matH"]
    return matA.cpu().to_dense(), matX.cpu(), matH.cpu()


A, X, H = loadMatrix(dataset)

baseDir = "exportSlice/%s" % dataset
# os.system("rm -rf exportSlice/%s" % dataset)
os.system("mkdir exportSlice/%s" % dataset)


# 打印访问的顺序
def printSPMM(spmm, filename):
    sliceLen = []
    sliceList = []
    sliceStartXY = []
    spmm = spmm != 0

    # 一次处理2048 * 16
    for Ti in range(0, spmm.shape[0], 2048):
        for Tj in range(0, spmm.shape[1], 16):
            tmpSliceList = []
            for j in range(0, min(spmm.shape[1] - Tj, 16)):
                for i in range(0, min(spmm.shape[0] - Ti, 2048)):
                    if spmm[Ti + i, Tj + j] != 0:
                        tmpSliceList.append((i, j))
            sliceLen.append(len(tmpSliceList))
            sliceStartXY.append(Ti)
            sliceStartXY.append(Tj)
            sliceList.append(tmpSliceList)

    with open(join(baseDir, "%s_len.txt" % filename), "w") as f:
        for line in sliceLen:
            f.write("%s\n" % line)

    with open(join(baseDir, "%s_XY.txt" % filename), "w") as f:
        for idx in range(0, len(sliceStartXY), 2):
            f.write("%d %d\n" % (sliceStartXY[idx], sliceStartXY[idx + 1]))

    with open(join(baseDir, "%s_list.txt" % filename), "w") as f:
        for tmpArr in sliceList:
            for num in tmpArr:
                f.write("%04d %02d " % (num[0], num[1]))
            f.write("\n")


# printSPMM(A, "A")
printSPMM(X, "X")
