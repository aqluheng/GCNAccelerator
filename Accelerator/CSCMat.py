class CSCMat():
    def __init__(self, shape=(16,16)) -> None:
        self.__shape = shape
        self.lenList = [0] * shape[1]
        self.rowIdxList = []
        for i in range(shape[1]):
            self.rowIdxList.append([])
        
    def setMat(self, torchMat):
        self.lenList = [0] * self.__shape[1]
        self.rowIdxList = []
        for i in range(self.__shape[1]):
            self.rowIdxList.append([])
        for rowIdx in range(torchMat.shape[0]):
            for colIdx in range(torchMat.shape[1]):
                if torchMat[rowIdx][colIdx] != 0:
                    self.rowIdxList[colIdx].append(rowIdx)
        for i in range(self.__shape[1]):
            self.lenList[i] = len(self.rowIdxList[i])

    def getAllNZCount(self):
        NZnum = 0
        for i in range(self.__shape[1]):
            NZnum += self.lenList[i]
        return NZnum