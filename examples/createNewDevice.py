#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: createNewDevice
# * AUTHOR: Pedro Encarnação
# * DATE: 25/03/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************
import numpy as np
from src.Geometry.easyPETBased import EasyCTGeometry, testSourceDistance
from src.DetectionLayout.Modules import PETModule, easyPETModule
from src.DetectionLayout.RadiationProducer import GenericRadiativeSource
from src.Device import StoreDeviceInFo
from src.Designer import DeviceDesignerStandalone

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

# newDevice.generateDeviceUUID() # one time only
# newDevice.createDirectory()  # one time only
# storeDevice = StoreDeviceInFo(device_directory=newDevice.deviceDirectory)  # one time only
# storeDevice.createDeviceInDirectory(object=newDevice)
print(newDevice.deviceUUID)
print(newDevice.deviceName)

device_path = "C:\\Users\\pedro\\OneDrive\\Documentos\\GitHub\\Infinity-Tomographic-Reconstruction\\configurations\\08d98d7f-a3c1-4cdf-a037-54655c7bdbb7_EasyCT"

getDevice = StoreDeviceInFo(device_directory=device_path)
deviceRead = getDevice.readDeviceFromDirectory()
print(deviceRead)