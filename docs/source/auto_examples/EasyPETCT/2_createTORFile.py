#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: createTORFile
# * AUTHOR: Pedro Encarnação
# * DATE: 24/03/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

"""
TOR FILE
======================
This is an example how to create a TOR file for easyPETCT
The file should be run one time to each new acquisition
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import uuid
import time
from TORFilesReader import ToRFile, AnimalType, PhantomType, AcquisitionInfo, ListModeBody, RadioisotopeInfo, Technician
from Device import StoreDeviceInFo
from Phantoms import NEMAIQ2008NU

print(np.__version__)
print(np.__file__)


# filename = "../../allvalues.npy"
filename = "C:\\Users\\pedro\\OneDrive\\Ambiente de Trabalho\\intelligent_scan-NewGeometries-CT\\allvalues.npy"
# filename = "C:\\Users\\pedro\\OneDrive\\Ambiente de Trabalho\\listmode_whitescan_32x1.npy"
output_path = "C:\\Users\\pedro\\OneDrive\\Ambiente de Trabalho\\all_values.tor"
# output_path = "C:\\Users\\pedro\\OneDrive\\Ambiente de Trabalho\\listmode_whitescan_32x1 (1).tor"
#
# if not os.path.exists(output_path):
#     os.makedirs(output_path)


device_path = "C:\\Users\\pedro\\OneDrive\\Documentos\\GitHub\\Infinity-Tomographic-Reconstruction\\configurations\\08d98d7f-a3c1-4cdf-a037-54655c7bdbb7_EasyCT"

getDevice = StoreDeviceInFo(device_directory=device_path)
newDevice = getDevice.readDeviceFromDirectory()
print(newDevice)
#-----------------------------------------
# create listMode
# IF Animal
# subject = AnimalType()
# subject.setHealthy(True)
# ....

# IF Phantom
subject = PhantomType()
subject.setPhantomName("NEMA IQ 2008 NU")
subject.setPhantomPurpose("Calibration")
subject.setPhantomDescription("NEMA IQ 2008 NU phantom for calibration")
subject.setDigitalPhantomCopy(NEMAIQ2008NU())

# If PET/SPECT/COMPTON
radioisotope = RadioisotopeInfo()
radioisotope.setTracers(["18F"])
radioisotope.setHalfLifes([float(109.771 * 60) ])
radioisotope.setDecayTypes(["BetaPlus"])
radioisotope.setDecayEnergies([511])

# Tecnhician
tecnhician = Technician()
tecnhician.setName("Pedro Encarnação")
tecnhician.setRole("Researcher")
tecnhician.setOrganization("Universidade de Aveiro")


scanHeader = AcquisitionInfo()
scanHeader.setId(1)
scanHeader.setScanType("CT")
scanHeader.setIndexesOfFrames([0, 1000, 2000, 3000, 4000])
scanHeader.setSubject(subject)
scanHeader.setTecnhician(tecnhician)
# scanHeader.setNumberOfFrames(1)
scanHeader.setInstanceUID(str(uuid.uuid4()))
scanHeader.setStudyInstanceUID(str(uuid.uuid4()))
scanHeader.setFrameOfReferenceUID(str(uuid.uuid4()))
scanHeader.setDate(time.strftime("%Y-%m-%d %H:%M:%S"))
# IF PET/SPECT/COMPTON
# scanHeader.setRadioisotope(radioisotope)

listmode = np.load(filename)
listmode[:,3] = np.copy(listmode[:,2])# invert ID_A and ID_B
listmode[:,2] = 0
listmode[:,1] = np.copy(listmode[:,0]) * 1000
listmode[:,0] = 0

listModeBody = ListModeBody()
listModeBody.setListmode(listmode)
listModeBody.setListmodeFields(["ENERGYA", "ENERGYB", "IDA", "IDB", "AXIAL_MOTOR", "FAN_MOTOR", "TIME"])
listModeBody.setFrameStartIndexes(scanHeader.indexesOfFrames)
listModeBody.generateStatistics()
listModeBody.printStatistics()
listModeBody.setGlobalDetectorID()
listModeBody.setCountsPerGlobalID()

plt.figure()
plt.hist(listModeBody["IDB"], bins=32)
plt.show()

ToRFile_creator = ToRFile(filepath=output_path)
ToRFile_creator.setSystemInfo(newDevice)
ToRFile_creator.setAcquisitionInfo(scanHeader)
ToRFile_creator.setfileBodyData(listModeBody)
ToRFile_creator.write()

#######CHECK TESTS###################
#######UNCOMMENT TO CHECK FILE AND GEOMETRY INTEGRATY############
ToRFile_reader = ToRFile(filepath=output_path)
ToRFile_reader.read()
listModeBody_read = ToRFile_reader.fileBodyData

plt.hist(listModeBody_read["ENERGYB"], bins=500)
plt.figure()
plt.hist2d(listModeBody_read["AXIAL_MOTOR"], listModeBody_read["FAN_MOTOR"],
           bins=(listModeBody_read.uniqueValuesCount[4], listModeBody_read.uniqueValuesCount[5]))
plt.show()
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
# testSourceDistance(deviceFromTOR.xRayProducer.focalSpotInitialPositionWKSystem, deviceFromTOR.sourceCenter,
#                    deviceFromTOR.originSystemWZ.T)
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

# designer = DeviceDesignerStandalone(device=deviceFromTOR)
# designer.addDevice()
# designer.addxRayProducerSource()
# designer.startRender()

# ToRFile_creator.setAcquisitionInfo(scanHeader)
# ToRFile_creator.setListMode(listmode)

