import CSCMat
# lenList = [0] * 16
# rowIdxList = [[]] * 16
class GCNBCUModel():
    def __init__(self) -> None:
        pass
    def processOneSlice(self, inputA: CSCMat, inputB: CSCMat):
        PECycle = [0] * 16
        for PEIdx, BrowList in enumerate(inputB.rowIdxList):
            PECycle[PEIdx] = 0
            for BIdx in BrowList:
                PECycle[PEIdx] += inputA.lenList[BIdx]
        maxLen = 0
        for cycleNum in PECycle:
            maxLen = max(maxLen, cycleNum)
        return maxLen