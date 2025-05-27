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
import numpy as np
import matplotlib.pyplot as plt
import uuid
import time

# from toor.StandaloneInitializer.angletoparametricMLEM import output_path
from toor.TORFilesReader import ToRFile, PhantomType, AcquisitionInfo, ListModeBody, RadioisotopeInfo, Technician
from toor.Device import StoreDeviceInFo
# from toor.Phantoms import NEMAIQ2008NU
from toor.Corrections.General import DetectorSensitivityResponse
from toor.CalibrationWrapper import CalibrationWrapper
from toor.EasyPETLinkInitializer.EasyPETDataReader import binary_data

print(np.__version__)
print(np.__file__)

# filename = "../../allvalues.npy"
filename = "C:\\Users\\regina.oliveira\\PycharmProjects\\EasyPETCT\\listmodes\\Am_Na.npy"
filename = "D:\\Pedro\\listmode_wirephantom.npy"
filename = "E:\\simulatedsinogram_matrix.npy"
# filename = "D:\\Pedro\\sensitivity_sim\\listmode_sensitivity_sim.npy"
# filename = "C:\\Users\\regina.oliveira\\PycharmProjects\\EasyPETCT\\listmodes\\listmode_sensitivity.npy"
# filename = "C:\\Users\\pedro\\OneDrive\\Ambiente de Trabalho\\listmode_whitescan_32x1.npy"
# output_path = "C:\\Users\\pedro\\OneDrive\\Ambiente de Trabalho\\all_values.tor"
# output_path = "C:\\Users\\pedro\\OneDrive\\Ambiente de Trabalho\\listmode_whitescan_32x1 (1).tor"
output_path = "C:\\Users\\regina.oliveira\\PycharmProjects\\EasyPETCT\\listmodes\\listmode_24Mar2025-17h54m00s.tor"
output_path = "D:\\Pedro\\listmode_wirephantom.tor"
output_path = "E:\\simulatedsinogram_matrix.tor"
# output_path = "E:\\sensitivity_sim.tor"
# output_path = "C:\\Users\\regina.oliveira\\PycharmProjects\\EasyPETCT\\listmodes\\listmode_sensitivity.tor"
#
# if not os.path.exists(output_path):
#     os.makedirs(output_path)


# device_path = "C:\\Users\\regina.oliveira\\PycharmProjects\\EasyPETCT\\.venv\\Lib\\site-packages\\configurations\\fda0f3b2-a0ae-470d-b30a-40e85b741c13_EasyCT"
device_path = r"C:\Users\pedro\OneDrive\Documentos\GitHub\Infinity-Tomographic-Reconstruction\configurations\b4593ba9-7193-43c8-abed-7a07eeeabb8d_EasyCT_simulation_16_2"
device_path = r"C:\Users\pedro\OneDrive\Documentos\GitHub\Infinity-Tomographic-Reconstruction\configurations\5433bf80-06b5-468f-9692-674f4b007605_EasyCT_simulation_16_2_special"

getDevice = StoreDeviceInFo(device_directory=device_path)
newDevice = getDevice.readDeviceFromDirectory()
print(newDevice)
# -----------------------------------------
# create listMode
# IF Animal
# subject = AnimalType()
# subject.setHealthy(True)
# ....

# IF Phantom
subject = PhantomType()
subject.setPhantomName("Na-22 source")
subject.setPhantomPurpose("CT scan")
subject.setPhantomDescription("CT scan of the Na-22 point source")
# subject.setDigitalPhantomCopy(NEMAIQ2008NU())

# If PET/SPECT/COMPTON
radioisotope = RadioisotopeInfo()
radioisotope.setTracers(["Na22"])
radioisotope.setHalfLifes([float(2.60 * 365 * 24 * 3600)])
radioisotope.setDecayTypes(["BetaPlus"])
radioisotope.setDecayEnergies([511])

# Tecnhician
tecnhician = Technician()
tecnhician.setName("Regina Oliveira")
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
listmode[:, 3] = np.copy(listmode[:, 2])  # invert ID_A and ID_B
listmode[:, 2] = 0
listmode[:, 1] = np.copy(listmode[:, 0]) * 1000
listmode[:, 0] = 0

even_mask = listmode[:, 3] % 2 == 0
odd_mask = listmode[:, 3] % 2 != 0
listmode[:, 3][even_mask] += 1
listmode[:, 3][odd_mask] -= 1

listModeBody = ListModeBody()
listModeBody.setListmode(listmode)
listModeBody.setListmodeFields(["ENERGYA", "ENERGYB", "IDA", "IDB", "AXIAL_MOTOR", "FAN_MOTOR", "TIME"])
listModeBody.setFrameStartIndexes(scanHeader.indexesOfFrames)
listModeBody.generateStatistics()
listModeBody.printStatistics()
listModeBody.setGlobalDetectorID()
listModeBody.setCountsPerGlobalID()

# %% [markdown]
# Generate detector sensitivity response (It is necessary to create the device one time first then generate the TOR file for the white scan and then generate the new device)
calibrations = CalibrationWrapper()
# file_white_scan = r"C:\Users\regina.oliveira\PycharmProjects\EasyPETCT\listmodes\listmode_whitescan_32x1 (1).tor"
# file_white_scan = r"C:\Users\regina.oliveira\PycharmProjects\EasyPETCT\listmodes\listmode_sensitivity.tor"
file_white_scan = "E:\\sensitivity_sim.tor"
# load FILE
ToRFile_sensitivity = ToRFile(filepath=file_white_scan)
ToRFile_sensitivity.read()

energies = np.array([30, 59.6, 511])
energy_windows = np.array([[10, 40], [45, 80], [511, 511]])
# comment this if the resolutionfucntion was not set
detector_sensitivity = DetectorSensitivityResponse(use_detector_energy_resolution=False)
detector_sensitivity.setEnergyPeaks(energies)
detector_sensitivity.setEnergyWindows(energyWindows=energy_windows,
                                      torFile=None)  # can set manually the energy windows. Put flag to use_detector_energy_resolution to False
detector_sensitivity.setDetectorSensitivity(torFile=ToRFile_sensitivity)
# detector_sensitivity.setDetectorSensitivity(generate_uniform=True, fileBodyData=listModeBody)
calibrations.setSystemSensitivity(detector_sensitivity)

plt.figure()
plt.hist(listModeBody["IDB"], bins=32)
plt.show()

ToRFile_creator = ToRFile(filepath=output_path)
ToRFile_creator.setSystemInfo(newDevice)
ToRFile_creator.setAcquisitionInfo(scanHeader)
ToRFile_creator.setCalibrations(calibrations)
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
