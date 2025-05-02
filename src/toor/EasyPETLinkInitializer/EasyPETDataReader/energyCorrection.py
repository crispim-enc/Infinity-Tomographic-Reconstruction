import numpy as np


def energyCorrection(list_energy_crystals_RCA, list_indexes_crystals_RCA, energyA,list_energy_crystals_RCB,
                     list_indexes_crystals_RCB, energyB,numberOfCrystals,unique_ECA, unique_ECB, CFArray):

    #CORRECTION FACTOR FILE


    # path= os.path.dirname(os.path.abspath(__file__))
    # newest = max(glob.iglob(path+'/calibrationdata/*.calbenergy'), key=os.path.getctime)
    # #newest = path + '/calibrationdata/MatrixA24_05_2017_18h40m_CF.out'
    # #print('Name of calibration file: '+ newest)
    #
    # CF = []
    # file = open(newest,"r")
    # for line in file.readlines():
    #    CF.append([])
    #    for i in line.split():
    #        CF[-1].append(float(i))
    #
    # CFArray = np.array(CF).T

    # E_corrected = np.zeros((len(energyA),(2)))
    #print(np.shape(E_corrected))
    #print(np.shape(energyA))
    dim_A = len(unique_ECA)
    dim_B = len(unique_ECB)
    #print(energyA[0])
    #print(energyB)
    #print('dimA'+str(dim_A))
    EA_corrected = energyA
    EB_corrected = energyB
    numberOfCrystals = numberOfCrystals[0]*numberOfCrystals[1]
    for n in range(0,dim_A):
        #list_energy = list_energy_crystals_RCA[n]
        #print('n'+str(n))
        #print(list_indexes_crystals_RCA)
        list_indexs_crystals = list_indexes_crystals_RCA[n].astype(np.int64)
        energy = energyA
        start1 = 0
        end1 = numberOfCrystals
        #E = list_energy[n]
        CF_ARRAY = CFArray[0][start1:end1]
        list_indexs_crystals = list_indexs_crystals.astype(np.int64)
        z = unique_ECA[n]
        #print(CF_ARRAY)
        #print(z)

        #print(energy[0][list_indexs_crystals] * CF_ARRAY[z])
        EA_corrected[0][list_indexs_crystals] = energy[0][list_indexs_crystals] * (CF_ARRAY[np.int(z - 1)] * 511 / CFArray[0][-2])

    for n in range(0, dim_B):
        #list_energy = list_energy_crystals_RCB[n]
        list_indexs_crystals = list_indexes_crystals_RCB[n].astype(np.int64)
        energy = energyB
        start1 = numberOfCrystals
        end1 = numberOfCrystals*2
        #E = list_energy[n]
        CF_ARRAY = CFArray[0][start1:end1]
        list_indexs_crystals = list_indexs_crystals.astype(np.int64)
        z = unique_ECB[n]
        EB_corrected[0][list_indexs_crystals] = energy[0][list_indexs_crystals] * (CF_ARRAY[np.int(z - 1)] * 511 / CFArray[0][-1])



    return EA_corrected, EB_corrected

