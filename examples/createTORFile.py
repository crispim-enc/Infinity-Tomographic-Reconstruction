#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: createTORFile
# * AUTHOR: Pedro Encarnação
# * DATE: 24/03/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

"""
This is an example how to create a TOR file for easyPETCT
"""
import os
import numpy as np
from src.Geometry.easyPETBased import EasyCTGeometry, testSourceDistance
from src.DetectionLayout.Modules import PETModule, easyPETModule
from src.DetectionLayout.RadiationProducer import GenericRadiativeSource
from src.Designer import DeviceDesignerStandalone
from src.TORFilesReader import ToRFile, AnimalType, PhantomType, AcquisitionInfo, Statistics, RadioisotopeInfo

# filename = "../../allvalues.npy"
filename = "C:\\Users\\pedro\\OneDrive\\Ambiente de Trabalho\\intelligent_scan-NewGeometries-CT\\allvalues.npy"
output_path = "C:\\Users\\pedro\\OneDrive\\Ambiente de Trabalho\\all_values.tor"
# if not os.path.exists(output_path):
#     os.makedirs(output_path)


# Set PET module type
_module = easyPETModule
# Set x-ray producer object
xrayproducer = GenericRadiativeSource()

# Set device
newDevice = EasyCTGeometry(detector_moduleA=_module, detector_moduleB=_module, x_ray_producer=xrayproducer)

# Set source
newDevice.xRayProducer.setFocalSpotInitialPositionWKSystem([-2, 0, 36.2 / 2])
newDevice.evaluateInitialSourcePosition()

# Set modules Side A
newDevice.setNumberOfDetectorModulesSideA(2)
moduleSideA_X_translation = np.array([-15, -15], dtype=np.float32)
moduleSideA_Y_translation = np.array([-2.175, 2.175], dtype=np.float32)
moduleSideA_Z_translation = np.array([36.2 / 2, 36.2 / 2], dtype=np.float32)
moduleSideA_alpha_rotation = np.array([0, 0], dtype=np.float32)
moduleSideA_beta_rotation = np.array([0, 0], dtype=np.float32)
moduleSideA_sigma_rotation = np.array([-90, -90], dtype=np.float32)

for i in range(newDevice.numberOfDetectorModulesSideA):
    newDevice.detectorModulesSideA[i].setXTranslation(moduleSideA_X_translation[i])
    newDevice.detectorModulesSideA[i].setYTranslation(moduleSideA_Y_translation[i])
    newDevice.detectorModulesSideA[i].setZTranslation(moduleSideA_Z_translation[i])
    newDevice.detectorModulesSideA[i].setAlphaRotation(moduleSideA_alpha_rotation[i])
    newDevice.detectorModulesSideA[i].setBetaRotation(moduleSideA_beta_rotation[i])
    newDevice.detectorModulesSideA[i].setSigmaRotation(moduleSideA_sigma_rotation[i])

newDevice.setNumberOfDetectorModulesSideB(2)
moduleSideB_X_translation = np.array([75, 75], dtype=np.float32)
moduleSideB_Y_translation = np.array([-2.175, 2.175], dtype=np.float32)
moduleSideB_Z_translation = np.array([36.2 / 2, 36.2 / 2], dtype=np.float32)
moduleSideB_alpha_rotation = np.array([0, 0], dtype=np.float32)
moduleSideB_beta_rotation = np.array([0, 0], dtype=np.float32)
moduleSideB_sigma_rotation = np.array([90, 90], dtype=np.float32)

for i in range(newDevice.numberOfDetectorModulesSideB):
    newDevice.detectorModulesSideB[i].setXTranslation(moduleSideB_X_translation[i])
    newDevice.detectorModulesSideB[i].setYTranslation(moduleSideB_Y_translation[i])
    newDevice.detectorModulesSideB[i].setZTranslation(moduleSideB_Z_translation[i])
    newDevice.detectorModulesSideB[i].setAlphaRotation(moduleSideB_alpha_rotation[i])
    newDevice.detectorModulesSideB[i].setBetaRotation(moduleSideB_beta_rotation[i])
    newDevice.detectorModulesSideB[i].setSigmaRotation(moduleSideB_sigma_rotation[i])

# newDevice
newDevice.setDeviceName("EasyCT")
newDevice.setDeviceType("CT")
newDevice.generateInitialCoordinates()
#-----------------------------------------
# create listMode
subject = AnimalType()
subject.setHealthy(True)

radioisotope = RadioisotopeInfo()
# radioisotope.setHalfLife(109.771)
# radioisotope.setDecayType("beta+")
# radioisotope.setDecayEnergy(511)

scanHeader = AcquisitionInfo()
scanHeader.setId(1)
# scanHeader.setScanType("PET")
scanHeader.setIndexesOfFrames([0, 1000])
scanHeader.setSubject(subject)
scanHeader.setRadioisotope(radioisotope)

listmode = np.load(filename)

ToRFile_creator = ToRFile(filepath=output_path)
ToRFile_creator.setListModeFields(["energyA", "energyB", "IDA", "IDB", "AXIAL_MOTOR", "FAN_MOTOR", "TIME"])
ToRFile_creator.setSystemInfo(newDevice)
ToRFile_creator.setAcquisitionInfo(scanHeader)
ToRFile_creator.setListMode(listmode)
ToRFile_creator.write()


ToRFile_reader = ToRFile(filepath=output_path)
ToRFile_reader.read()
print(ToRFile_reader.systemInfo)
# ToRFile_creator.setAcquisitionInfo(scanHeader)
# ToRFile_creator.setListMode(listmode)

