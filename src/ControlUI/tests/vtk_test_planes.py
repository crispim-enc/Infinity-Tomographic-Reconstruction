#!/usr/dataFiles/env python

# A simple vtkInteractorStyleImage example for
# 3D image viewing with the vtkImageResliceMapper.

# Drag Left mouse button to window/level
# Shift-Left drag to rotate (oblique slice)
# Shift-Middle drag to slice through image
# OR Ctrl-Right drag to slice through image

import vtk
from vtk.util.misc import vtkGetDataRoot
import os
import numpy as np
VTK_DATA_ROOT = vtkGetDataRoot()

# reader = vtk.vtkImageReader()
# reader.ReleaseDataFlagOff()
# reader.SetDataByteOrderToLittleEndian()
# reader.SetDataMask(0x7fff)
# reader.SetDataExtent(0,63,0,63,1,93)
# reader.SetDataSpacing(3.2,3.2,1.5)
# reader.SetFilePrefix("" + str(VTK_DATA_ROOT) + "/Data/headsq/quarter")


dir = os.path.dirname(__file__)
dir= (os.path.abspath(os.path.join(dir, os.pardir)))
colormap='hot'
path = os.path.join(dir, 'colormap_files')
# path = dir + "/colormap_files/"

file_name = "D:\\github_easypet\\easyPETtraining\\EasyPET pre-clinical versions\\demo_acqs_files\\Easypet Scan NAF_FDG\\static_image\\im.npy"
volume = np.load(file_name)
data3D=volume
pixel_size_reconstruct_file=[1,1,1]
z_pos = data3D.shape[2]

#  Volume inversion
data3D = np.flip(data3D, axis=2)
# try:
#     self.ren_volume.RemoveVolume(self.Volume4VTK)
# except AttributeError:
#     print('no volume yet')
# if init is False:
#     self.ren_volume.RemoveActor(self.axial_line)
#     self.ren_volume.RemoveActor(self.coronal_line)
#     self.ren_volume.RemoveActor(self.sagittal_line)
#     self.ren_volume.RemoveActor(self.sagittal)
#     self.ren_volume.RemoveActor(self.coronal)
#     self.ren_volume.RemoveActor(self.axial)

size = data3D.shape
w = size[0]
h = size[1]
d = size[2]
stack = np.zeros((w, d, h))

for j in range(0, z_pos):
    stack[:, j, :] = data3D[:, :, j]

stack = np.require(stack, dtype=np.float32)  # stack = np.require(data3D,dtype=np.uint16)
normalize_colormap = np.max(volume)
# start_color = int(abs(self.rs2.max() - self.rs2.end()) * 10.24)
# end_color = int(abs(self.rs2.max() - self.rs2.start()) * 10.24)

start_color = int(abs(0) * 10.24)
end_color = int(abs(100) * 10.24)

# --------IMPORT DATA-----------------

dataImporter = vtk.vtkImageImport()
# The preaviusly created array is converted to a string of chars and imported.
data_string = stack.tostring()
dataImporter.CopyImportVoidPointer(data_string, len(data_string))
# dataImporter.SetDataScalarTypeToUnsignedShort()
dataImporter.SetDataScalarTypeToFloat()
dataImporter.SetDataSpacing(pixel_size_reconstruct_file[0], pixel_size_reconstruct_file[2],
                            pixel_size_reconstruct_file[1])
print('pixel_size{}'.format(dataImporter.GetDataSpacing()))
# Because the data that is imported only contains an intensity value (it isnt RGB-coded or someting similar), the importer
# must be told this is the case.
dataImporter.SetNumberOfScalarComponents(1)
# ---------------STORE DATA-------------------------------
dataImporter.SetDataExtent(0, w - 1, 0, d - 1, 0, h - 1)
dataImporter.SetWholeExtent(0, w - 1, 0, d - 1, 0, h - 1)
# dataImporter.SetTransform(transL1)

# -----------------------Scalar range-------------
dataImporter.Update()

# -----------------------------------------------------------
# The following class is used to store transparencyv-values for later retrival. In our case, we want the value 0 to be
# completly opaque whereas the three different cubes are given different transperancy-values to show how it works.
alphaChannelFunc = vtk.vtkPiecewiseFunction()
colorTransferFunction = vtk.vtkColorTransferFunction()
colorTransferFunction.SetColorSpaceToRGB()

# ----------------OPACITYMAP-----------------------------

# http://www.kennethmoreland.com/color-advice/

file_name = os.path.join(path, '{}.cl'.format(colormap))
reader = np.loadtxt(file_name)
bins, res = np.histogram(stack.ravel(), len(reader), (stack.min(), stack.max()))
res2 = np.interp(res, [stack.min(), stack.max()], [0, 1])
opacitymap = np.vstack((res, res2)).T
opacitymap = opacitymap.astype('float32')

increment_color = 0
for row in range(0, len(reader)):
    if row < start_color:
        colorTransferFunction.AddRGBPoint(float(reader[row][0]) * normalize_colormap,
                                          float(reader[0][1]), float(reader[0][2]), float(reader[0][3]))

    elif row > end_color:
        colorTransferFunction.AddRGBPoint(float(reader[row][0]) * normalize_colormap,
                                          float(reader[-1][1]), float(reader[-1][2]), float(reader[-1][3]))
    else:
        try:
            increment_color += len(reader) / abs(start_color - end_color)
            if int(np.round(increment_color, 0)) < len(reader):
                colorTransferFunction.AddRGBPoint(float(reader[row][0]) * normalize_colormap,
                                                  float(reader[int(np.round(increment_color, 0))][1]),
                                                  float(reader[int(np.round(increment_color, 0))][2]),
                                                  float(reader[int(np.round(increment_color, 0))][3]))
        except ZeroDivisionError as e:
            print(e)

alphaChannelFunc.AddPoint(opacitymap[start_color, 0], 0)
alphaChannelFunc.AddPoint(opacitymap[end_color, 0], 1)

reader = dataImporter
# Create the RenderWindow, Renderer
ren1 = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren1)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

im = vtk.vtkImageResliceMapper()
im.SetInputConnection(reader.GetOutputPort())
im.SliceFacesCameraOn()
im.SliceAtFocalPointOn()
im.BorderOff()

ip = vtk.vtkImageProperty()
ip.SetColorWindow(2000)
ip.SetColorLevel(1000)
ip.SetAmbient(0.0)
ip.SetDiffuse(1.0)
ip.SetOpacity(1.0)
ip.SetInterpolationTypeToLinear()

ia = vtk.vtkImageSlice()
ia.SetMapper(im)
ia.SetProperty(ip)

ren1.AddViewProp(ia)
ren1.SetBackground(0.1,0.2,0.4)
renWin.SetSize(300,300)

iren = vtk.vtkRenderWindowInteractor()
style = vtk.vtkInteractorStyleImage()
style.SetInteractionModeToImage3D()
iren.SetInteractorStyle(style)
renWin.SetInteractor(iren)

# render the image
renWin.Render()
cam1 = ren1.GetActiveCamera()
cam1.ParallelProjectionOn()
ren1.ResetCameraClippingRange()
renWin.Render()

iren.Start()