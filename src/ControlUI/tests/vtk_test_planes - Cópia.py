#!/usr/bin/env python

import vtk
import numpy as np
import os

from vtk.util.misc import vtkGetDataRoot


# VTK_DATA_ROOT = vtkGetDataRoot()
#
# reader = vtk.vtkImageReader2()
#
# reader.SetFilePrefix(VTK_DATA_ROOT + "/Data/headsq/quarter")
#
# reader.SetDataExtent(0, 63, 0, 63, 1, 93)
#
# reader.SetDataSpacing(3.2, 3.2, 1.5)
#
# reader.SetDataOrigin(0.0, 0.0, 0.0)
#
# reader.SetDataScalarTypeToUnsignedShort()
#
# reader.UpdateWholeExtent()
file_name = "D:\\github_easypet\\easyPETtraining\\EasyPET pre-clinical versions\\demo_acqs_files\\Easypet Scan NAF_FDG\\static_image\\im.npy"
volume = np.load(file_name)
data3D=volume
pixel_size_reconstruct_file=[1,1,1]
z_pos = data3D.shape[2]

#  Volume inversion
data3D = np.flip(data3D, axis=2)

size = data3D.shape
w = size[0]
h = size[1]
d = size[2]


stack = np.zeros((w, d, h))
for j in range(0, z_pos):
    stack[:, j, :] = data3D[:, :, j]

stack = np.require(stack, dtype=np.float32)  # stack = np.require(data3D,dtype=np.uint16)
normalize_colormap = np.max(volume)

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

  # Calculate the center of the volume
reader=dataImporter
# reader.Update()
(xMin, xMax, yMin, yMax, zMin, zMax) = reader.GetExecutive().GetWholeExtent(reader.GetOutputInformation(0))
(xSpacing, ySpacing, zSpacing) = reader.GetOutput().GetSpacing()
(x0, y0, z0) = reader.GetOutput().GetOrigin()


center = [x0 + xSpacing * 0.5 * (xMin + xMax),
          y0 + ySpacing * 0.5 * (yMin + yMax),
          z0 + zSpacing * 0.5 * (zMin + zMax)]

 # Matrices for axial, coronal, sagittal, oblique view orientations

axial = vtk.vtkMatrix4x4()

axial.DeepCopy((1, 0, 0, center[0],
                0, 1, 0, center[1],
                0, 0, 1, center[2],
                0, 0, 0, 1))

coronal = vtk.vtkMatrix4x4()

coronal.DeepCopy((1, 0, 0, center[0],
                  0, 0, 1, center[1],
                  0, -1, 0, center[2],
                  0, 0, 0, 1))

sagittal = vtk.vtkMatrix4x4()
sagittal.DeepCopy((0, 0, -1, center[0],
                   1, 0, 0, center[1],
                   0, -1, 0, center[2],
                   0, 0, 0, 1))

oblique = vtk.vtkMatrix4x4()
oblique.DeepCopy((1, 0, 0, center[0],
                  0, 0.866025, -0.5, center[1],
                  0, 0.5, 0.866025, center[2],
                  0, 0, 0, 1))

reslice = vtk.vtkImageReslice()

reslice.SetInputConnection(reader.GetOutputPort())

reslice.SetOutputDimensionality(2)

reslice.SetResliceAxes(axial)

reslice.SetInterpolationModeToLinear()

table = vtk.vtkLookupTable()

table.SetRange(0, normalize_colormap)  # image intensity range

table.SetValueRange(0.0, 1.0)  # from black to white

table.SetSaturationRange(0.0, 0.0)  # no color saturation

table.SetRampToLinear()

table.Build()

color = vtk.vtkImageMapToColors()

color.SetLookupTable(table)

color.SetInputConnection(reslice.GetOutputPort())

actor = vtk.vtkImageActor()

actor.GetMapper().SetInputConnection(color.GetOutputPort())

renderer = vtk.vtkRenderer()

renderer.AddActor(actor)

window = vtk.vtkRenderWindow()

window.AddRenderer(renderer)

interactorStyle = vtk.vtkInteractorStyleImage()

interactor = vtk.vtkRenderWindowInteractor()

interactor.SetInteractorStyle(interactorStyle)

window.SetInteractor(interactor)

window.Render()


actions = {}

actions["Slicing"] = 0


def ButtonCallback(obj, event):

    if event == "LeftButtonPressEvent":
        actions["Slicing"] = 1
    else:
        actions["Slicing"] = 0

def MouseMoveCallback(obj, event):
    (lastX, lastY) = interactor.GetLastEventPosition()
    (mouseX, mouseY) = interactor.GetEventPosition()
    if actions["Slicing"] == 1:
        deltaY = mouseY - lastY
        reslice.Update()
        sliceSpacing = reslice.GetOutput().GetSpacing()[2]
        matrix = reslice.GetResliceAxes()
        center = matrix.MultiplyPoint((0, 0, sliceSpacing * deltaY, 1))
        matrix.SetElement(0, 3, center[0])
        matrix.SetElement(1, 3, center[1])
        matrix.SetElement(2, 3, center[2])
        window.Render()
    else:
        interactorStyle.OnMouseMove()



interactorStyle.AddObserver("MouseMoveEvent", MouseMoveCallback)

interactorStyle.AddObserver("LeftButtonPressEvent", ButtonCallback)

interactorStyle.AddObserver("LeftButtonReleaseEvent", ButtonCallback)

interactor.Start()

