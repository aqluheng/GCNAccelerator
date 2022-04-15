import CSCMat
class GCNAXModel():
    def __init__(self) -> None:
        pass
    def processOneSlice(self, inputA: CSCMat, inputB: CSCMat):
        return inputA.getAllNZCount()