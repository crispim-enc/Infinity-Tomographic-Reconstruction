#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: createNewDevice
# * AUTHOR: Pedro Encarnação
# * DATE: 25/03/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

"""
This is an example how to create a new device. In this case a new system for EasyCT
The device should be run only one time to create a new device.  A folder with a unique identifier will be created
Afterwars the device can be read from the folder and added to the new TOR files created
"""
import matplotlib.pyplot as plt
import numpy as np
from src.Geometry.easyPETBased import EasyCTGeometry, testSourceDistance
from src.DetectionLayout.Modules import PETModule, easyPETModule
from src.DetectionLayout.RadiationProducer import GenericRadiativeSource
from src.Designer import DeviceDesignerStandalone
from src.Device import StoreDeviceInFo

# Set PET module type
_module = easyPETModule

# Set x-ray producer object
xrayproducer = GenericRadiativeSource()
xrayproducer.setSourceName("Am-241")
xrayproducer.setSourceActivity(1.0 * 37000)
xrayproducer.setFocalSpotDiameter(1)
xrayproducer.setShieldingShape("Cylinder")
xrayproducer.setShieldingMaterial("Lead")
xrayproducer.setShieldingDensity(11.34)
xrayproducer.setShieldingThickness(0.5)
xrayproducer.setShieldingHeight(4)
xrayproducer.setShieldingRadius(12.5)
xrayproducer.setMainEmissions({1: {"energy": 59.54, "intensity": 0.36},
                                 2: {"energy": 26.34, "intensity": 0.024},
                                 })
# xrayproducer.setFocalSpotInitialPositionWKSystem([-2, 0, -(32*2+31*0.28)/2])
# Set device
newDevice = EasyCTGeometry(detector_moduleA=_module, detector_moduleB=_module, x_ray_producer=xrayproducer)
# Set source
newDevice.setDistanceBetweenMotors(30)
newDevice.setDistanceFanMotorToDetectorModulesOnSideA(0)
newDevice.setDistanceFanMotorToDetectorModulesOnSideB(60)
newDevice.xRayProducer.setFocalSpotInitialPositionWKSystem([12.55, 3, 0])
# newDevice.xRayProducer.setFocalSpotInitialPositionWKSystem([12.55, 4, (32*2+31*0.28)/2])

newDevice.evaluateInitialSourcePosition()

# Set modules Side A
newDevice.setNumberOfDetectorModulesSideA(1)

moduleSideA_X_translation = np.array([15], dtype=np.float32)
moduleSideA_Y_translation = np.array([0], dtype=np.float32)
moduleSideA_Z_translation = np.array([0], dtype=np.float32)
moduleSideA_alpha_rotation = np.array([0], dtype=np.float32)
moduleSideA_beta_rotation = np.array([0], dtype=np.float32)
moduleSideA_sigma_rotation = np.array([0], dtype=np.float32)

for i in range(newDevice.numberOfDetectorModulesSideA):
    newDevice.detectorModulesSideA[i].model32()
    newDevice.detectorModulesSideA[i].setXTranslation(moduleSideA_X_translation[i])
    newDevice.detectorModulesSideA[i].setYTranslation(moduleSideA_Y_translation[i])
    newDevice.detectorModulesSideA[i].setZTranslation(moduleSideA_Z_translation[i])
    newDevice.detectorModulesSideA[i].setAlphaRotation(moduleSideA_alpha_rotation[i])
    newDevice.detectorModulesSideA[i].setBetaRotation(moduleSideA_beta_rotation[i])
    newDevice.detectorModulesSideA[i].setSigmaRotation(moduleSideA_sigma_rotation[i])

newDevice.setNumberOfDetectorModulesSideB(1)
moduleSideB_X_translation = np.array([-75], dtype=np.float32)
moduleSideB_Y_translation = np.array([0], dtype=np.float32)
moduleSideB_Z_translation = np.array([0], dtype=np.float32)
moduleSideB_alpha_rotation = np.array([0], dtype=np.float32)
moduleSideB_beta_rotation = np.array([0], dtype=np.float32)
moduleSideB_sigma_rotation = np.array([180], dtype=np.float32)

for i in range(newDevice.numberOfDetectorModulesSideB):
    newDevice.detectorModulesSideB[i].model32()
    newDevice.detectorModulesSideB[i].setXTranslation(moduleSideB_X_translation[i])
    newDevice.detectorModulesSideB[i].setYTranslation(moduleSideB_Y_translation[i])
    newDevice.detectorModulesSideB[i].setZTranslation(moduleSideB_Z_translation[i])
    newDevice.detectorModulesSideB[i].setAlphaRotation(moduleSideB_alpha_rotation[i])
    newDevice.detectorModulesSideB[i].setBetaRotation(moduleSideB_beta_rotation[i])
    newDevice.detectorModulesSideB[i].setSigmaRotation(moduleSideB_sigma_rotation[i])

# newDevice
newDevice.setDeviceName("EasyCT")
newDevice.setDeviceType("CT")

newDevice.generateInitialCoordinatesWKSystem()
newDevice.generateInitialCoordinatesXYSystem()

# newDevice.generateDeviceUUID() # one time only
# newDevice.createDirectory()  # one time only

# storeDevice = StoreDeviceInFo(device_directory=newDevice.deviceDirectory)  # one time only
# storeDevice = StoreDeviceInFo(device_directory=device_path)  # one time only
# storeDevice.createDeviceInDirectory(object=newDevice)

# ----------------------------------------------------------
# TESTS PART
unique_header = np.repeat(np.arange(0,32), 13)
axial_motor_angles = (np.zeros(32*13))
fan_motor_angles = np.tile(np.arange(-90, 105, 15), 32)

newDevice.detectorSideBCoordinatesAfterMovement(axial_motor_angles, fan_motor_angles, unique_header)
device_path = "C:\\Users\\pedro\\OneDrive\\Documentos\\GitHub\\Infinity-Tomographic-Reconstruction\\configurations\\08d98d7f-a3c1-4cdf-a037-54655c7bdbb7_EasyCT"

#
# getDevice = StoreDeviceInFo(device_directory=device_path)
# deviceRead = getDevice.readDeviceFromDirectory()
# print(deviceRead)
axial_motor_angles = np.array([0, 0], dtype=np.float32)
fan_motor_angles = np.array([0, 0], dtype=np.float32)
newDevice.sourcePositionAfterMovement(axial_motor_angles, fan_motor_angles)
plt.figure(figsize=(10, 10))
plt.plot(newDevice.originSystemWZ[0], newDevice.originSystemWZ[1], 'ro', label='Origin Fan Motor')
# plot source center
plt.plot(newDevice.sourceCenter[:, 0], newDevice.sourceCenter[:, 1], 'bo', label='Source Center')
plt.plot(newDevice.originSystemXY[0], newDevice.originSystemXY[1], 'ko', label='Origin FOV')
plt.plot(newDevice.centerFace[:, 0], newDevice.centerFace[:, 1], 'go', label='Center Face Detector Module B')
plt.plot(newDevice._verticesB[:, :, 0], newDevice._verticesB[:,:, 1], 'mo', label='Vertices Base Detector Module B')
#plot a line linking the originwz to the center face
plt.plot([np.ones(newDevice.centerFace.shape[0]) * newDevice.originSystemWZ[0,0],
            newDevice.centerFace[:, 0]], [np.ones(newDevice.centerFace.shape[0]) * newDevice.originSystemWZ[1,0],
                                        newDevice.centerFace[:, 1]], '-')
# plt.xlim(-5,40)
# plt.ylim(25, 70)
plt.figure(figsize=(10, 10))

# x an Z direction
plt.plot(newDevice.originSystemWZ[0], newDevice.originSystemWZ[2], 'ro', label='Origin Fan Motor')
# plot source center
plt.plot(newDevice.sourceCenter[:, 0], newDevice.sourceCenter[:, 2], 'bo', label='Source Center')
plt.plot(newDevice.originSystemXY[0], newDevice.originSystemXY[2], 'ko', label='Origin FOV')
plt.plot(newDevice.centerFace[:, 0], newDevice.centerFace[:, 2], 'go', label='Center Face Detector Module B')
plt.plot(newDevice._verticesB[:, :, 0], newDevice._verticesB[:,:, 2], 'mo', label='Vertices Base Detector Module B')
# plot a line linking the originwz to the center face

plt.show()
designer = DeviceDesignerStandalone(device=newDevice)
designer.addDevice()
designer.addxRayProducerSource()
designer.startRender()
