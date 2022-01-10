dataset = "CiteSeer"
baseDir = "exportSlice/%s" % dataset
MATNAME = "X"

from os.path import join
import numpy as np
import math


class sliceReader:
    def __init__(self) -> None:
        self.__idx = 0
        self.__fileLen = open(join(baseDir, "%s_len.txt"%MATNAME))
        self.__fileList = open(join(baseDir, "%s_list.txt"%MATNAME))
        self.__fileXY = open(join(baseDir, "%s_XY.txt"%MATNAME))

    def getSlice(self):
        sliceStr = self.__fileLen.readline().strip()
        if sliceStr == "":
            return None, None, None
        sliceLen = int(sliceStr)
        sliceXY = [int(x) for x in self.__fileXY.readline().strip().split(" ")]
        sliceList = [int(x) for x in self.__fileList.readline().strip().split(" ")]

        return sliceLen, sliceXY, sliceList


def splitUsePartition(x, partitionList, PEnum=4):
    for idx, partition in enumerate(partitionList):
        if x <= partition:
            return idx
    return None

def splitUseMod(x, PEnum=4):
    return x % PEnum


def splitSliceForPEs(inputSlice, PEnum=4):
    def specificPartition():
        # 将任务按照工作量进行平均分配
        countX = sorted([inputSlice[i] for i in range(0, len(inputSlice), 2)])
        partitionList = []
        for i in range(PEnum):
            partitionList.append(countX[int((i + 1) / PEnum * len(countX)) - 1])
        return partitionList

    def uniformPartition():
        # 将任务按照X坐标进行平均分配
        partitionList = []
        for i in range(PEnum):
            partitionList.append(int((i + 1) / PEnum * 2048) - 1)
        return partitionList

    partitionList = uniformPartition() # 这里是baseline
    # partitionList = specificPartition() # 优化一: 按任务量分配

    splitedList = []
    for i in range(PEnum):
        splitedList.append([])
    for idx in range(0, len(inputSlice), 2):
        x, y = inputSlice[idx], inputSlice[idx + 1]
        # splitedList[splitUsePartition(x, partitionList, PEnum)].append((x, y))
        splitedList[splitUseMod(x, PEnum)].append((x, y))
    print(len(splitedList[0]),len(splitedList[1]),len(splitedList[2]),len(splitedList[3]))
    exit()
    return splitedList


def processingPairs(pairList, PEnum=4):
    bankUsed = [0] * PEnum
    canProcess = np.zeros(PEnum).astype("int")
    for i, pair in enumerate(pairList):
        if pair[0] == -1:
            continue
        # 当前只考虑了B的版块
        if bankUsed[pair[1] % PEnum] == 0:
            bankUsed[pair[1] % PEnum] = 1
            canProcess[i] = 1
    return canProcess


def computeCycle(parallelSlice):
    cycle = 0
    processIdxList = np.zeros(len(parallelSlice)).astype("int")
    lengthList = []
    for Arr in parallelSlice:
        lengthList.append(len(Arr))
    lengthList = np.array(lengthList)
    while np.any(processIdxList < lengthList):
        currentPairs = []
        for i, processIdx in enumerate(processIdxList):
            if processIdx >= lengthList[i]:
                currentPairs.append((-1, -1))
            else:
                currentPairs.append(parallelSlice[i][processIdx])
        cycle += 1
        canProcess = processingPairs(currentPairs)
        print(currentPairs, canProcess) # 打印执行顺序
        processIdxList += canProcess
    return cycle


reader = sliceReader()
sumCycle = 0
bestCycle = 0
sumMultiply = 0
while True:
    sliceLen, sliceXY, sliceList = reader.getSlice()
    if sliceLen == None:
        print("乘法总数",sumMultiply)
        print("4PE最优",bestCycle)
        print("当前情况",sumCycle)
        exit()

    parallelSlice = splitSliceForPEs(sliceList)
    cycle = computeCycle(parallelSlice)
    sumCycle += cycle
    bestCycle += math.ceil(sliceLen / 4)
    sumMultiply += sliceLen
    exit()
