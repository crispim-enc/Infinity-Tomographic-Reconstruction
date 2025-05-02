import numpy as np
import gc



# def crystalsRC_client(ratioMatrix,peakMatrix,numberOfCrystals):
#
#
#     """
#     Identification the crystals associated to each event
#
#     Parameters:
#     -----------
#     matrixRatio = to store the ratios
#         (Column 1 = ratio of RCA, Column 2 = ratio of RCB)
#     peakMatrix = (Column 1 = peaks of RCA,Column 2 = peaks of RCB)
#
#     numberOfCrystals = number of crystals
#
#
#     Returns:
#     -------
#     #crystalMatrix = to store the crystal information (number)
#         (Column 1 = crystal of RCA, Column 2 = crystal of RCB)
#
#     """
#
#     #crystal_matrix = to store the crystal information
#     #(Column 1 = crystal of RCA, Column 2 = crystal of RCB)
#     crystalMatrix = np.zeros(((len(ratioMatrix[:,0])),(2)))
#     ratioMatrix=ratioMatrix*10000
#     ratioMatrix=np.int16(ratioMatrix)
#     peakMatrix=peakMatrix*10000
#     peakMatrix=np.int16(peakMatrix)
#     numberOfCrystals = numberOfCrystals[0]*numberOfCrystals[1]
#     for i in range (0,2):
#
#         ratio =  np.matrix(ratioMatrix[:,i]).T
#         repeatRatio = np.repeat(ratio,numberOfCrystals, axis=1)
#         repeatPeaks = np.repeat((np.matrix(peakMatrix[:,i])),ratioMatrix.shape[0], axis=0)
#         repeatPeaks = np.abs(repeatRatio - repeatPeaks)
#         subMin = np.min(repeatPeaks, axis = 1)
#         find = np.where(repeatPeaks == subMin)
#
#
#         u, indices,counts = np.unique(find[0], return_index=True, return_counts=True)
#
#         moreThanOne = np.where(counts >1)
#         listToDelete = []
#         listToDelete.append([])
#         if (len(moreThanOne[0]) != 0):
#             for j in range (0, len(moreThanOne[0])):
#                 ind = np.where(find[0] == moreThanOne[0][j])
#                 add= ind[0][1:len(ind[0])]
#                 listToDelete[-1].append(add[0])
#
#
#         index_crystal =  np.delete(find[1],listToDelete[0]) +1
#
#
#         #crystal = np.array(crystal_index)
#         crystalMatrix[:,i]=index_crystal
#         del repeatPeaks, repeatRatio
#         gc.collect()
#
#     return crystalMatrix

def crystalsRC_client(ratioMatrix,peakMatrix,numberOfCrystals):
    crystalMatrix = np.zeros((len(ratioMatrix[:, 0]), 2), dtype=np.int16)
    ratioMatrix = ratioMatrix * 10000
    ratioMatrix = np.int16(ratioMatrix)
    peakMatrix = peakMatrix * 10000
    peakMatrix = np.int16(peakMatrix)
    # numberOfCrystals = numberOfCrystals[0] * numberOfCrystals[1]
    for i in range(0, 2):
        ratio = np.matrix(ratioMatrix[:, i]).T
        diff = abs(ratio - peakMatrix[:, i])
        crystalMatrix[:,i] = np.where(diff == np.min(diff, axis=1))[1] + 1

    gc.collect()


    return crystalMatrix

