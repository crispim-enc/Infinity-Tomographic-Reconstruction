#-------------------------------------IMPORT------------------
import matplotlib.pyplot as plt
import numpy as np
import os


#Functions
from .detect_peaks import detect_peaks

#-----------------------------------------------------------------

def peaksFile(data, threshold_filterData, numberOfCrystals, threshold_detectPeaks, mph, mpd , nbins, dir, file_name_1):

    ratioA = (data[:,0] - data[:,1])/(data[:,0] + data[:,1])
    print('(data[:,0] + data[:,1]) ' +str((data[:,0] + data[:,1])) )

    roiA = ((ratioA < threshold_filterData) & (ratioA > -threshold_filterData)) *1
    ratioA = roiA*ratioA
    indexToDeleteA = np.where(roiA == 0)

    ratioB = (data[:,3] - data[:,2])/(data[:,3] + data[:,2])
    roiB = ((ratioB < threshold_filterData) & (ratioB> -threshold_filterData)) *1
    print('ROIB '+ str(roiB))
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

    #peaks of RCA
    weightA, binsA= np.histogram(ratioMatrix[:,0], nbins,(-1,1))
    binsA = (binsA[:-1]+binsA[1:])/2
    ind = detect_peaks(weightA, mph = mph, mpd = mpd, threshold=threshold_detectPeaks)
    data_peak_y_A = weightA[ind]
    data_peak_x_A = binsA[ind]

    #peaks of RCB
    weightB, binsB= np.histogram(ratioMatrix[:,1], nbins,(-1,1))
    binsB = (binsB[:-1]+binsB[1:])/2
    ind = detect_peaks(weightB, mph = mph, mpd = mpd, threshold=threshold_detectPeaks)
    data_peak_y_B = weightB[ind]
    data_peak_x_B = binsB[ind]
    print('B ' + str(len(data_peak_x_B)))
    print('A ' + str(len(data_peak_x_A)))

    #--------------------------------
    #          PLOT RATIOS - CHECK PEAKS DETECTION

    # plt.hist(ratioMatrix[:,0], nbins,(-1,1), edgecolor= 'beige', facecolor = 'y')
    # plt.plot(data_peak_x_A, data_peak_y_A, 'r*')
    # plt.hist(ratioMatrix[:,1], nbins,(-1,1), edgecolor= 'ghostwhite', facecolor = 'c')
    # plt.plot(data_peak_x_B, data_peak_y_B, 'r*')
    
    #-------------------------------   

    path = os.path.dirname(__file__)
    numberOfCrystals = numberOfCrystals[0]*numberOfCrystals[1]
    if(((len(data_peak_x_B)) | (len(data_peak_x_A))) != numberOfCrystals):
        print('DETECT PEAKS FAILED - REDEFINE PARAMETERS VALUES ')

        return data_peak_x_A, data_peak_y_A, data_peak_x_B, data_peak_y_B, weightA, weightB, binsA

        
        
    else:
        peaksMatrix_x = np.zeros((numberOfCrystals*2))
        peaksMatrix_x[0:numberOfCrystals] = data_peak_x_A
        peaksMatrix_x[numberOfCrystals:(numberOfCrystals*2)] = data_peak_x_B
        peaksMatrix_y = np.zeros((numberOfCrystals*2))
        peaksMatrix_y[0:numberOfCrystals] = data_peak_y_A
        peaksMatrix_y[numberOfCrystals:(numberOfCrystals*2)] = data_peak_y_B
        head, tail = os.path.split(file_name_1)
        filename, file_extension = os.path.splitext(tail)
        path = path + "/calibrationdata/"
        file_name = path + filename + '_peaks.calbpeak'
        np.savetxt(file_name,peaksMatrix_x, delimiter=",")

        print('PEAKS FILE CREATED')
        

        return peaksMatrix_x, peaksMatrix_y, ratioMatrix, data, file_name
