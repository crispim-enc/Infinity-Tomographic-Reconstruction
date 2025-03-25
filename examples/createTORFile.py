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
import matplotlib.pyplot as plt

from src.TORFilesReader import ToRFile, AnimalType, PhantomType, AcquisitionInfo, Statistics, RadioisotopeInfo

# filename = "../../allvalues.npy"
filename = "C:\\Users\\pedro\\OneDrive\\Ambiente de Trabalho\\intelligent_scan-NewGeometries-CT\\allvalues.npy"
output_path = "C:\\Users\\pedro\\OneDrive\\Ambiente de Trabalho\\all_values.tor"
# if not os.path.exists(output_path):
#     os.makedirs(output_path)



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

#######CHECK TESTS###################
#######UNCOMMENT TO CHECK FILE AND GEOMETRY INTEGRATY############
ToRFile_reader = ToRFile(filepath=output_path)
ToRFile_reader.read()
print(ToRFile_reader.systemInfo)

deviceFromTOR = ToRFile_reader.systemInfo

axial_motor_angles = np.deg2rad(np.arange(0, 360, 45))

fan_motor_angles = np.deg2rad(np.arange(-45, 60, 15))
# repeat the fan motor angles for each axial motor angle
fan_motor_angles = np.repeat(fan_motor_angles, len(axial_motor_angles))
axial_motor_angles = np.tile(axial_motor_angles, len(fan_motor_angles) // len(axial_motor_angles))

deviceFromTOR.sourcePositionAfterMovement(axial_motor_angles, fan_motor_angles)
plt.plot(deviceFromTOR.originSystemWZ[0], deviceFromTOR.originSystemWZ[1], 'ro', label='Origin Fan Motor')
# plot source center
plt.plot(deviceFromTOR.sourceCenter[:, 0], deviceFromTOR.sourceCenter[:, 1], 'bo', label='Source Center')
# plot a line from the origin to the source center at fan motor angle 0
testSourceDistance(deviceFromTOR.xRayProducer.focalSpotInitialPositionWKSystem, deviceFromTOR.sourceCenter,
                   deviceFromTOR.originSystemWZ.T)
index_fan_motor_angle_0 = np.where(fan_motor_angles == 0)
source_center_fan_motor_angle_0 = deviceFromTOR.sourceCenter[index_fan_motor_angle_0]
origin_fan_motor_angle_0 = deviceFromTOR.originSystemWZ.T[index_fan_motor_angle_0]

# plt.plot(origin_fan_motor_angle_0[0], origin_fan_motor_angle_0[1], 'x')
plt.plot(source_center_fan_motor_angle_0[:, 0], source_center_fan_motor_angle_0[:, 1], 'gx')

plt.plot([origin_fan_motor_angle_0[:, 0], source_center_fan_motor_angle_0[:, 0]],
         [origin_fan_motor_angle_0[:, 1], source_center_fan_motor_angle_0[:, 1]], '-')
plt.legend()
plt.title("Configuration Source side of detector module A")
plt.title("Configuration Source in front module")
plt.show()

designer = DeviceDesignerStandalone(device=deviceFromTOR)
designer.addDevice()
designer.addxRayProducerSource()
designer.startRender()

# ToRFile_creator.setAcquisitionInfo(scanHeader)
# ToRFile_creator.setListMode(listmode)

