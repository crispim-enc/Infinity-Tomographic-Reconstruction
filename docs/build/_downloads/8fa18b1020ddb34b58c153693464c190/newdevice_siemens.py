#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: Siemens_Intevo_Bold_SPECT_CT/CT.py
# * AUTHOR: Pedro Encarnação
# * DATE: 31/03/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************
"""Siemens Intevo Bold SPECT/CT
================
"""""
import pydicom
import os
import matplotlib.pyplot as plt
import numpy as np

from skimage.transform import iradon

from src.DetectionLayout.Modules import PETModule
from src.DetectionLayout.Modules import easyPETModule
from src.DetectionLayout.Modules.SPECTModuleGeneric import SPECTHeadGeneric
from src.DetectionLayout.Photodetectors.Crystals import GenericCrystal
from src.Geometry.Standard import PlanarGeometry
from src.Designer import DeviceDesignerStandalone


file_path = "C:\\Users\\pedro\\Downloads\\phase_1_challenge_data_03_05_2025\\NEMA_phantom-20250221T115435Z-001\\NEMA_phantom\\SPECT-projections\\"

file_path = os.path.join(file_path, "DICOM\\25011616\\21060000\\31702333")
ds = pydicom.dcmread(file_path)
print(ds)
#%%
if hasattr(ds, "PixelData"):
    num_frames = getattr(ds, "NumberOfFrames", 1)  # Multi-frame images
    rows, cols = ds.Rows, ds.Columns
#%%
# detector information sequence
detector_info = ds.DetectorInformationSequence[0]
radial_positions = detector_info.RadialPosition
FOV_shape = detector_info.FieldOfViewShape
FOV_dim = detector_info.FieldOfViewDimensions
Focal_distance = detector_info.FocalDistance
start_angle = detector_info.StartAngle
#pixel spacing
pixel_spacing = ds.PixelSpacing

# print(f"Radial Positions: {radial_positions}")
rotationVector = ds.RotationVector
rotationInformationSequence = ds.RotationInformationSequence[0]
RotationDirection = rotationInformationSequence.RotationDirection
scanArc = rotationInformationSequence.ScanArc
angularStep = rotationInformationSequence.AngularStep
# numberOfFrames = rotationInformationSequence.NumberOfFrames

# arcOfRotation = rotationVector.ScanArc

angularViewVector = np.array(ds.AngularViewVector)
detectorVector = np.array(ds.DetectorVector)

print(f"Rotation Vector: {rotationVector}")


# module_spect = SPECTHeadGeneric()


# module_.setHighEnergyLightDetectorBlock(2)

#
newDevice = PlanarGeometry(distance_between_planes=537)
newDevice.setDeviceName("Siemens SYmbia Intevo Bold SPEC/CT")
newDevice.setNumberOfModulesZ(1)
newDevice.setNumberOfModulesPerSide(1)
newDevice.setNumberOfModulesPhi(2)
print("Number of modules: ", newDevice.numberOfModules)

modules_ = [PETModule(i) for i in range(newDevice.numberOfModules)]
newDevice.setDetectorModule(modules_)

for i in range(newDevice.numberOfModules):
    newDevice.detectorModule[i].setModuleID(i)
    newDevice.detectorModule[i].updateNumberHighEnergyLightDetectors(128,128)

    print("Number of high energy light detectors: ", newDevice.detectorModule[i].totalNumberHighEnergyLightDetectors)

    newDevice.detectorModule[i].setModelHighEnergyLightDetectors([GenericCrystal(k) for k in
                                                range(newDevice.detectorModule[i].totalNumberHighEnergyLightDetectors)])
    for j in range(newDevice.detectorModule[i].totalNumberHighEnergyLightDetectors):

        newDevice.detectorModule[i].modelHighEnergyLightDetectors[j].setCrystalID(j)
        newDevice.detectorModule[i].modelHighEnergyLightDetectors[j].setCristalSize(pixel_spacing[0], pixel_spacing[0], 30)
    newDevice.detectorModule[i].setReflectorThicknessX(0)
    newDevice.detectorModule[i].setReflectorThicknessY(0)

    newDevice.detectorModule[i].setHighEnergyLightDetectorBlock()

newDevice.calculateInitialGeometry()

designer = DeviceDesignerStandalone(device=newDevice)
designer.addDevice()
designer.startRender()

print(newDevice.getDeviceName())



#%%



#%%


# Check if the file contains Pixel Data (projection images)
if hasattr(ds, "PixelData"):
    pixel_array = ds.pixel_array  # Convert pixel data to NumPy array

    num_frames = getattr(ds, "NumberOfFrames", 1)  # Check number of projection frames
    print(f"Projection Data Found: {num_frames} frames of {ds.Rows}x{ds.Columns} pixels.")

    # Display first few projections
    num_display = min(num_frames, 10)  # Show up to 6 projections
    fig, axes = plt.subplots(1, num_display, figsize=(15, 5))
    for i in range(num_display):
        axes[i].imshow(pixel_array[i], cmap="gray")
        axes[i].set_title(f"Projection {i+1}")
        axes[i].axis("off")
    # plt.show()

else:
    print("No Pixel Data found.")
#%%
#create a sinogram from the dicom information
print(pixel_array.shape)
sinogram = pixel_array[40]
# theta = np.linspace()
# image = iradon(sinogram, theta=theta, circle=True)
# plt.imshow(image, cmap='gray')/