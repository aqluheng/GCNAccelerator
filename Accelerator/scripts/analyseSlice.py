import math
import numpy as np
from os.path import join
dataset = "Cora"
baseDir = "exportSlice/%s" % dataset
MATNAME = "A"

# OMemoryModel = "1Bank4Port"
# OMemoryModel = "4Bank1Port"
# OMemoryModel = "8Bank1Port"
# OMemoryModel = "4Bank2Port"
OMemoryModel = "2Bank2Port"

reshuffleTask = True
# reshuffleTask = False

PEnum = 4


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


def splitSliceForPEs(inputSlice):
    def getSplitId(x, partitionList):
        for idx, partition in enumerate(partitionList):
            if x <= partition:
                return idx
        return None

    def specificPartition():
        # 将任务按照工作量进行平均分配
        countX = sorted([inputSlice[i] for i in range(0, len(inputSlice), 2)])
        partitionList = []
        for i in range(PEnum):
            partitionList.append(countX[int((i + 1) / PEnum * len(countX)) - 1])
        splitedList = []
        for i in range(PEnum):
            splitedList.append([])
        for idx in range(0, len(inputSlice), 2):
            x, y = inputSlice[idx], inputSlice[idx + 1]
            splitedList[getSplitId(x, partitionList)].append((x, y))
        return splitedList

    def uniformPartition():
        # 将任务按照X坐标进行平均分配
        partitionList = []
        for i in range(PEnum):
            partitionList.append(int((i + 1) / PEnum * 2048) - 1)
        splitedList = []
        for i in range(PEnum):
            splitedList.append([])
        for idx in range(0, len(inputSlice), 2):
            x, y = inputSlice[idx], inputSlice[idx + 1]
            splitedList[getSplitId(x, partitionList)].append((x, y))
        return splitedList

    # splitedList = uniformPartition()
    splitedList = specificPartition()

    return splitedList


def readBMat(readQuest):
    def ignoreBBank():
        # 横向使用4port或者4duplicate
        canProcess = np.ones(PEnum).astype("int")
        return canProcess
    return ignoreBBank()


def readOMat(readQuest):
    def ignoreOBank():
        # 横向使用4port或者4duplicate
        canProcess = np.ones(PEnum).astype("int")
        return canProcess

    def originBank():
        # 横向分四板块,每块1port
        canProcess = np.zeros(PEnum).astype("int")
        hasRead = np.zeros(PEnum).astype("int")
        for i, questId in enumerate(readQuest):
            if questId == -1:
                continue

            if hasRead[int(questId/2048*PEnum)] == 0:
                hasRead[int(questId/2048*PEnum)] = 1
                canProcess[i] = 1

        return canProcess

    def moreBank():
        # 横向分八板块,每块1port
        canProcess = np.zeros(PEnum).astype("int")
        hasRead = np.zeros(PEnum*2).astype("int")
        for i, questId in enumerate(readQuest):
            if questId == -1:
                continue

            if hasRead[int(questId/2048*PEnum*2)] == 0:
                hasRead[int(questId/2048*PEnum*2)] = 1
                canProcess[i] = 1

        return canProcess

    def morePort():
        # 横向分四板块,每块2port
        canProcess = np.zeros(PEnum).astype("int")
        hasRead = np.zeros(PEnum).astype("int")
        for i, questId in enumerate(readQuest):
            if questId == -1:
                continue

            if hasRead[int(questId/2048*PEnum)] < 2:
                hasRead[int(questId/2048*PEnum)] += 1
                canProcess[i] = 1

        return canProcess

    def lessBank():
        BankNum = PEnum // 2
        canProcess = np.zeros(PEnum).astype("int")
        hasRead = np.zeros(BankNum).astype("int")
        for i, questId in enumerate(readQuest):
            if questId == -1:
                continue

            if hasRead[int(questId/2048*BankNum)] < 2:
                hasRead[int(questId/2048*BankNum)] += 1
                canProcess[i] = 1

        return canProcess

    if OMemoryModel == "1Bank4Port":
        canProcess = ignoreOBank()
    elif OMemoryModel == "4Bank1Port":
        canProcess = originBank()
    elif OMemoryModel == "8Bank1Port":
        canProcess = moreBank()
    elif OMemoryModel == "4Bank2Port":
        canProcess = morePort()
    elif OMemoryModel == "2Bank2Port":
        canProcess = lessBank()
    else:
        exit("内存模型出问题了")

    return canProcess

# 只考虑对O的读取


def shuffleSlice(inputSlice):
    def shuffleFor4Port():
        def getSplitId(x, partitionList):
            for idx, partition in enumerate(partitionList):
                if x <= partition:
                    return idx
            return None
        # 将任务按照工作量进行平均分配
        countX = sorted([inputSlice[i] for i in range(0, len(inputSlice), 2)])
        partitionList = []
        for i in range(PEnum):
            partitionList.append(countX[int((i + 1) / PEnum * len(countX)) - 1])
        splitedList = []
        for i in range(PEnum):
            splitedList.append([])
        for idx in range(0, len(inputSlice), 2):
            x, y = inputSlice[idx], inputSlice[idx + 1]
            splitedList[getSplitId(x, partitionList)].append((x, y))
        return splitedList

    def shuffleFor4Bank():
        inPairs = [(inputSlice[i], inputSlice[i+1]) for i, x in enumerate(inputSlice) if(i % 2 == 0)]

        splitBank = []
        for i in range(PEnum):
            splitBank.append([x for x in inPairs if (i/PEnum*2048 <= x[0] < (i+1)/PEnum*2048)])

        return splitBank

    def shuffleFor8Bank():
        inPairs = [(inputSlice[i], inputSlice[i+1]) for i, x in enumerate(inputSlice) if(i % 2 == 0)]

        BankNum = PEnum * 2
        splitBank = []
        for i in range(BankNum):
            splitBank.append([x for x in inPairs if (i/BankNum*2048 <= x[0] < (i+1)/BankNum*2048)])

        splitTask = []
        for i in range(PEnum):
            splitTask.append([])

        nowIdx = np.zeros(BankNum).astype("int")
        maxLen = np.array([len(x) for x in splitBank]).astype("int")
        while True:
            leftoverNum = maxLen - nowIdx
            if np.all(leftoverNum == 0):
                break

            hasUsed = np.zeros(BankNum).astype("int")
            for idxForTask in range(PEnum):
                # 找到最长的任务,且那个端口尚未使用
                nowWorkJobLen = 0
                nowWorkJobId = -1
                for idxForSearchPort, leftTaskLen in enumerate(leftoverNum):
                    if hasUsed[idxForSearchPort] < 1 and leftTaskLen > nowWorkJobLen:
                        nowWorkJobId, nowWorkJobLen = idxForSearchPort, leftTaskLen
                if nowWorkJobId != -1:
                    # print(idxForTask,nowWorkJobId,nowIdx[nowWorkJobId])
                    splitTask[idxForTask].append(splitBank[nowWorkJobId][nowIdx[nowWorkJobId]])
                    nowIdx[nowWorkJobId] += 1
                    hasUsed[nowWorkJobId] += 1
                    leftoverNum[nowWorkJobId] -= 1

        return splitTask

    def shuffleFor2Port():
        inPairs = [(inputSlice[i], inputSlice[i+1]) for i, x in enumerate(inputSlice) if(i % 2 == 0)]

        splitBank = []
        for i in range(PEnum):
            splitBank.append([x for x in inPairs if (i/PEnum*2048 <= x[0] < (i+1)/PEnum*2048)])

        splitTask = []
        for i in range(PEnum):
            splitTask.append([])

        nowIdx = np.zeros(PEnum).astype("int")
        maxLen = np.array([len(x) for x in splitBank]).astype("int")
        while True:
            leftoverNum = maxLen - nowIdx
            if np.all(leftoverNum == 0):
                break

            hasUsed = np.zeros(PEnum).astype("int")
            for idxForTask in range(PEnum):
                # 找到最长的任务,且那个端口使用次数少于2
                nowWorkJobLen = 0
                nowWorkJobId = -1
                for idxForSearchPort, leftTaskLen in enumerate(leftoverNum):
                    if hasUsed[idxForSearchPort] < 2 and leftTaskLen > nowWorkJobLen:
                        nowWorkJobId, nowWorkJobLen = idxForSearchPort, leftTaskLen
                if nowWorkJobId != -1:
                    # print(idxForTask,nowWorkJobId,nowIdx[nowWorkJobId])
                    splitTask[idxForTask].append(splitBank[nowWorkJobId][nowIdx[nowWorkJobId]])
                    nowIdx[nowWorkJobId] += 1
                    hasUsed[nowWorkJobId] += 1
                    leftoverNum[nowWorkJobId] -= 1

        return splitTask

    def shuffleFor2Bank2Port():
        inPairs = [(inputSlice[i], inputSlice[i+1]) for i, x in enumerate(inputSlice) if(i % 2 == 0)]

        splitBank = []
        bankNum = PEnum // 2
        for i in range(bankNum):
            splitBank.append([x for x in inPairs if (i/bankNum*2048 <= x[0] < (i+1)/bankNum*2048)])

        splitTask = []
        for i in range(PEnum):
            splitTask.append([])

        nowIdx = np.zeros(bankNum).astype("int")
        maxLen = np.array([len(x) for x in splitBank]).astype("int")
        
        while True:
            leftoverNum = maxLen - nowIdx
            if np.all(leftoverNum == 0):
                break

            hasUsed = np.zeros(PEnum).astype("int")
            for idxForTask in range(PEnum):
                # 找到最长的任务,且那个端口使用次数少于2
                nowWorkJobLen = 0
                nowWorkJobId = -1
                for idxForSearchPort, leftTaskLen in enumerate(leftoverNum):
                    if hasUsed[idxForSearchPort] < 2 and leftTaskLen > nowWorkJobLen:
                        nowWorkJobId, nowWorkJobLen = idxForSearchPort, leftTaskLen
                if nowWorkJobId != -1:
                    # print(idxForTask,nowWorkJobId,nowIdx[nowWorkJobId])
                    splitTask[idxForTask].append(splitBank[nowWorkJobId][nowIdx[nowWorkJobId]])
                    nowIdx[nowWorkJobId] += 1
                    hasUsed[nowWorkJobId] += 1
                    leftoverNum[nowWorkJobId] -= 1

        return splitTask

    if OMemoryModel == "1Bank4Port":
        parallelOutput = shuffleFor4Port()
    elif OMemoryModel == "4Bank1Port":
        parallelOutput = shuffleFor4Bank()
    elif OMemoryModel == "8Bank1Port":
        parallelOutput = shuffleFor8Bank()
    elif OMemoryModel == "4Bank2Port":
        parallelOutput = shuffleFor2Port()
    elif OMemoryModel == "2Bank2Port":
        parallelOutput = shuffleFor2Bank2Port()
    else:
        exit("内存模型出问题了")

    return parallelOutput

# 一次处理一批任务,并计算需要多少cycle


def computeCycle(parallelSlice):
    cycle = 0

    processIdxList = np.zeros(len(parallelSlice)).astype("int")  # 每个切片的指针
    lengthList = []
    for Arr in parallelSlice:
        lengthList.append(len(Arr))
    lengthList = np.array(lengthList)  # 每个切片的长度

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

        readBQuest = [x[1] for x in currentPairs]
        readOQuest = [x[0] for x in currentPairs]

        canProcessB = readBMat(readBQuest)
        canProcessO = readOMat(readOQuest)

        canProcess = np.bitwise_and(canProcessB, canProcessO)
        # print(currentPairs, canProcess)  # 打印执行顺序
        processIdxList += canProcess
    return cycle


reader = sliceReader()
sumCycle = 0
bestCycle = 0
sumMultiply = 0
while True:
    sliceLen, sliceXY, sliceList = reader.getSlice()  # 读取下一个分块
    if sliceLen == None:
        print("数据集: %s 矩阵: %s" % (dataset, MATNAME))
        print("乘法总数", sumMultiply)
        print("4PE最优", bestCycle)
        print("当前情况", sumCycle)
        exit()
    if reshuffleTask == False:
        parallelSlice = splitSliceForPEs(sliceList)     # 将任务划分给不同的PE
    else:
        parallelSlice = shuffleSlice(sliceList)
    cycle = computeCycle(parallelSlice)             # 计算当前划分所需cycle
    sumCycle += cycle
    bestCycle += math.ceil(sliceLen / 4)
    sumMultiply += sliceLen
    # exit()
