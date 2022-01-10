dataset = "Cora"
baseDir = "exportSlice/%s" % dataset
MATNAME = "A"

from os.path import join
import numpy as np
import math


class sliceReader:
    def __init__(self) -> None:
        self.__idx = 0
        self.__fileLen = open(join(baseDir, "%s_len.txt" % MATNAME))
        self.__fileList = open(join(baseDir, "%s_list.txt" % MATNAME))
        self.__fileXY = open(join(baseDir, "%s_XY.txt" % MATNAME))

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

    # partitionList = uniformPartition()  # 这里是baseline
    partitionList = specificPartition() # 优化一: 按任务量分配

    splitedList = []
    for i in range(PEnum):
        splitedList.append([])
    for idx in range(0, len(inputSlice), 2):
        x, y = inputSlice[idx], inputSlice[idx + 1]
        splitedList[splitUsePartition(x, partitionList, PEnum)].append((x, y))
    # print(len(splitedList[0]),len(splitedList[1]),len(splitedList[2]),len(splitedList[3]))
    # exit()
    return splitedList


def processingPairs(pairList, PEnum=4):
    canProcess = np.zeros(PEnum).astype("int")
    minIdx = -1
    for i, pair in enumerate(pairList):
        if pair[1] == -1:
            continue
        elif minIdx == -1 or minIdx > pair[1]:
            minIdx = pair[1]

    for i, pair in enumerate(pairList):
        if pair[1] == minIdx:
            canProcess[i] = 1
    return canProcess

def getNewReg(pairList, lastReg):
    newReg = lastReg.copy()
    readLineIdx = 100001 # 100001 是一个不会访问到的数值
    for i, pair in enumerate(pairList):
        if pair[1] != -1 and pair[1] != lastReg[i]:
            readLineIdx = min(readLineIdx, pair[1])
    for i, pair in enumerate(pairList):
        if pair[1] == readLineIdx:
            newReg[i] = readLineIdx
    return newReg

def processingPairsWithRegFile(pairList, Breg, PEnum=4):
    canProcess = np.zeros(PEnum).astype("int")
    for i, pair in enumerate(pairList):
        if pair[1] == Breg[i]:
            canProcess[i] = 1
    return canProcess




# 添加B寄存器,只考虑输出矩阵的板块冲突,计算cycle
def computeCycle(parallelSlice):
    cycle = 0

    processIdxList = np.zeros(len(parallelSlice)).astype("int")  # 每个切片的指针
    lengthList = []
    for Arr in parallelSlice:
        lengthList.append(len(Arr))
    lengthList = np.array(lengthList)  # 每个切片的长度

    lastReg = np.ones(4).astype("int") * -1
    while np.any(processIdxList < lengthList):
        # 如果存在未完成的切片则算出这个cycle哪些可以计算
        currentPairs = []
        for i, processIdx in enumerate(processIdxList):
            if processIdx >= lengthList[i]:
                currentPairs.append((-1, -1))
            else:
                currentPairs.append(parallelSlice[i][processIdx])

        # currentPairs 存储了当前处理元素的横坐标与纵坐标, 需要根据这两个计算板块冲突,以及所需cycle
        cycle += 1
        # 查看哪几个需要读取B的寄存器,并向B发送请求

        
        # canProcess = processingPairs(currentPairs)  # 无寄存器状态

        Breg = getNewReg(currentPairs, lastReg) # B有寄存器 语句1
        lastReg = Breg                          # B有寄存器 语句2
        canProcess = processingPairsWithRegFile(currentPairs, Breg) # B有寄存器 语句3
        
        
        # print(currentPairs, canProcess)  # 打印执行顺序
        processIdxList += canProcess
    return cycle


reader = sliceReader()
sumCycle = 0
bestCycle = 0
sumMultiply = 0
while True:
    sliceLen, sliceXY, sliceList = reader.getSlice()
    if sliceLen == None:
        print("乘法总数", sumMultiply)
        print("4PE最优", bestCycle)
        print("当前情况", sumCycle)
        exit()

    parallelSlice = splitSliceForPEs(sliceList)
    cycle = computeCycle(parallelSlice)
    sumCycle += cycle
    bestCycle += math.ceil(sliceLen / 4)
    sumMultiply += sliceLen
    # exit()
