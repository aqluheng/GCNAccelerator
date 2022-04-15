from CSCMat import CSCMat
from GCNBCULBModel import GCNBCULBModel
from GCNBCUModel import GCNBCUModel
from GCNBModel import GCNBModel
from GCNAXModel import GCNAXModel
import argparse
from cmath import phase
import torch
from os.path import join
import os

parser = argparse.ArgumentParser(description='GCNSim')
parser.add_argument('--model', type=str, default="GCNAX", choices=['GCNAX', 'GCNB', 'GCNBCU', 'GCNBCULB'],
                    help="Model to use GCNAX, GCNB baseline,GCNB with coalsecing unit, GCNB with coalsecing unit and load balance")
parser.add_argument('--dataset', type=str, default="Cora",
                    choices=['Cora', 'CiteSeer', 'PubMed', 'Nell'], help="Dataset to evaluation")
parser.add_argument('--refresh', default=False,action="store_true", help="强制刷新切片")
args = parser.parse_args()


SIMmodel = None
if args.model == 'GCNAX':
    SIMmodel = GCNAXModel()
elif args.model == 'GCNB':
    SIMmodel = GCNBModel()
elif args.model == 'GCNBCU':
    SIMmodel = GCNBCUModel()
elif args.model == 'GCNBCULB':
    SIMmodel = GCNBCULBModel()


basePath = args.dataset
# 此处以上为解析参数


def preprocessTorchDict(phaseNum):
    phaseDict = torch.load(join(basePath, "phase%dDict.pt"%phaseNum))
    leftMatList = phaseDict['leftMats']
    rightMatList = phaseDict['rightMats']

    def blockTorch(leftMat: torch.Tensor, rightMat: torch.Tensor):
        M, K, N = leftMat.shape[0], leftMat.shape[1], rightMat.shape[1]
        assert(K == rightMat.shape[0])

        blockedLeftMatCSC = []
        blockedRightMatCSC = []
        for BmStart in range(0, M, 2048):
            for BkStart in range(0, K, 16):
                for BnStart in range(0, N, 16):
                    leftMatCSC = CSCMat(shape=(2048, 16))
                    rightMatCSC = CSCMat(shape=(16, 16))
                    leftMatCSC.setMat(leftMat[BmStart:BmStart+2048, BkStart:BkStart+16])
                    rightMatCSC.setMat(rightMat[BkStart:BkStart+16, BnStart:BnStart+16])
                    blockedLeftMatCSC.append(leftMatCSC)
                    blockedRightMatCSC.append(rightMatCSC)
        return blockedLeftMatCSC, blockedRightMatCSC

    leftSliceList, rightSliceList = [], []
    for i in range(len(leftMatList)):
        mmLeftSliceList, mmRightSliceList = blockTorch(leftMatList[i], rightMatList[i])
        leftSliceList.append(mmLeftSliceList)
        rightSliceList.append(mmRightSliceList)

    sliceDict = {"leftSlice": leftSliceList, "rightSlice": rightSliceList}
    torch.save(sliceDict, join(basePath, "phase%dSlice.pt"%phaseNum))


def computePhase(phaseNum):
    pathName = join(basePath, "phase%dSlice.pt"%phaseNum)
    if not os.path.exists(pathName) or args.refresh:
        preprocessTorchDict(phaseNum)
    
    phaseSlice = torch.load(pathName)
    mmCount = len(phaseSlice["leftSlice"])
    phaseCycle = 0
    for mmIdx in  range(mmCount):
        nowMMcycle = 0
        for sliceIdx in range(len(phaseSlice["leftSlice"][mmIdx])):
            leftMMSlice = phaseSlice["leftSlice"][mmIdx][sliceIdx]
            rightMMSlice = phaseSlice["rightSlice"][mmIdx][sliceIdx]
            nowMMcycle += SIMmodel.processOneSlice(leftMMSlice, rightMMSlice)
        # print(nowMMcycle)
        phaseCycle += nowMMcycle
    print("Phase ",phaseNum,"cost:",phaseCycle)

computePhase(1)
computePhase(2)
computePhase(3)
computePhase(4)