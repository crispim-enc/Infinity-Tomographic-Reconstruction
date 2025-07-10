# *******************************************************
# * FILE: ratioEnergy_client.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

#--------------------------
#IMPORT
#--------------------------
import numpy as np
import gc


def ratioEnergy_client(data, threshold, numberOfCrystals):

       """
       Determine the ratios and peaks for each RC
    
       Parameters:
       -----------
       data = matrix with energies
       threshold = value to remove noise. It must be a positive number

       Returns:
       --------
       matrixRatio = ratios of energy
       """

       ratioA = (data[:,0] - data[:,1])/(data[:,0] + data[:,1])
       ratioB = (data[:, 3] - data[:, 2]) / (data[:, 3] + data[:, 2])
       index_remove_0_A = np.where(ratioA == 0)
       index_remove_0_B = np.where(ratioB == 0)
       #indexes_intersection_remove_ = np.intersect1d(index_remove_0_A, index_remove_0_B)

       # union_indexes_intersection
       union_indexes_intersection_remove = np.append(index_remove_0_A[0], index_remove_0_B[0])

       union_indexes_intersection_remove = np.unique(union_indexes_intersection_remove)


       data = np.delete(data, union_indexes_intersection_remove, axis=0)
       ratioA = np.delete(ratioA, union_indexes_intersection_remove, axis=0)
       ratioB = np.delete(ratioB, union_indexes_intersection_remove, axis=0)
       #np.save('ratioA.npy', ratioA)
       #np.save('ratioB.npy', ratioB)
       roiA = ((ratioA < threshold) & (ratioA > -threshold)) *1
       ratioA = roiA*ratioA
       indexToDeleteA = np.where(roiA == 0)


       roiB = ((ratioB < threshold) & (ratioB> -threshold)) *1
       ratioB = roiB*ratioB
       indexToDeleteB = np.where(roiB == 0)

       all_indexes = np.union1d(indexToDeleteA[0], indexToDeleteB[0])

       ratio = np.delete(ratioA, all_indexes)
       ratioMatrix = np.zeros((len(ratio),(2)))
       ratioMatrix[:,0] = ratio

       ratio = np.delete(ratioB, all_indexes)
       ratioMatrix[:,1] = ratio
       
       

       #remove the indexes on matrixData
       data = np.delete(data, all_indexes, axis = 0)
       del ratioA,ratioB, roiA, roiB, ratio
       gc.collect()
       return ratioMatrix,data

