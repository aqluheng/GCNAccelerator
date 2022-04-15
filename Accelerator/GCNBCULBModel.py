import CSCMat
class GCNBCULBModel():
    def __init__(self) -> None:
        pass
    def processOneSlice(self, inputA: CSCMat, inputB: CSCMat):
        PECycle = [0] * 16
        for PEIdx, BrowList in enumerate(inputB.rowIdxList):
            PECycle[PEIdx] = 0
            for BIdx in BrowList:
                PECycle[PEIdx] += inputA.lenList[BIdx]
        meanLen = 0
        runningCount = 0
        for cycleNum in PECycle:
            meanLen += cycleNum
            if cycleNum != 0:
                runningCount += 1
        if runningCount == 0:
            return 0
        else:
            return meanLen // runningCount