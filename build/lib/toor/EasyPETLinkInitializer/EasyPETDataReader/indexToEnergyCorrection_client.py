# *******************************************************
# * FILE: indexToEnergyCorrection_client.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

#--------------------------
#IMPORT
#--------------------------
import numpy as np

def indexToEnergyCorrection_client(array1, array2 ,crystalMatrix):

    energy = array1 + array2
    energy1 = np.array([energy])
    energy = energy1.T

    crystals = np.array([crystalMatrix])
    crystals = crystals.T

    indexes = np.linspace(0,len(crystals), num = len(crystals), endpoint = False)
    
    indexes = np.array([indexes])
    indexes = indexes.T
    
    informationRC = np.concatenate((energy, indexes, crystals),axis=1)

    informationRC = informationRC[np.argsort(informationRC[:,2])]
    
    unique_EC, indices, counts = np.unique(informationRC[:,2], return_index = True, return_counts=True)
    
    list_energy_crystals = []
    list_indexes_crystals = []
    

    for i in range(0,len(unique_EC)):
        if (i < (len(unique_EC) - 1)):
            energyValues = (informationRC[:,0][indices[i]:indices[i+1]])
            indexes = (informationRC[:,1][indices[i]:indices[i+1]])
        else:
            energyValues = (informationRC[:,0][indices[i]:len(informationRC[:,1])])
            indexes = (informationRC[:,1][indices[i]:len(informationRC[:,1])])    
        
        list_energy_crystals.append(energyValues)
        list_indexes_crystals.append(indexes)
    

    
    return list_energy_crystals , list_indexes_crystals, energy1, unique_EC
   
