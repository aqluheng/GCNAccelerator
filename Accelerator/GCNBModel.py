import CSCMat
# lenList = [0] * 16
# rowIdxList = [[]] * 16

class GCNBModel():
    def __init__(self) -> None:
        pass
    def processOneSlice(self, inputA: CSCMat, inputB: CSCMat):
        PEtaskList = []
        for i in range(16):
            tmpTaskList = []
            for rowIdx in inputB.rowIdxList[i]:
                tmpTaskList = tmpTaskList + [(x,rowIdx) for x in inputA.rowIdxList[rowIdx]]
            PEtaskList.append(tmpTaskList)
        nowProcessIdx = [0] * 16
        targetIdx = [len(x) for x in PEtaskList]
        cycle = 0
        while nowProcessIdx != targetIdx:
            cycle += 1
            readQuest = []
            for i in range(16):
                if nowProcessIdx[i] < targetIdx[i]:
                    readQuest.append([i,PEtaskList[i][nowProcessIdx[i]]])
            # readQuest [(questPEID, [ArowId, AbankId]),(questPEID, [ArowId, AbankId])]
            canRead = []
            hasRead = [-1] * 16
            for questTuple in readQuest:
                ArowId, AbankId = questTuple[1]
                if hasRead[AbankId] == -1 or hasRead[AbankId] == ArowId:
                # if hasRead[AbankId] == -1:
                    hasRead[AbankId] = ArowId
                    canRead.append(questTuple[0])
            for PEidx in canRead:
                nowProcessIdx[PEidx] += 1
            # exit(0)
        return cycle