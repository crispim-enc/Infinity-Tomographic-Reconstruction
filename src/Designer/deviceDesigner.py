import numpy as np
import vtk


class DeviceDesignerStandalone:
    def __init__(self, device=None):
        self.device = device
        self.ren = vtk.vtkRenderer()
        self.renderWin = vtk.vtkRenderWindow()
        self.renderWin.AddRenderer(self.ren)
        WIDTH = 640
        HEIGHT = 480
        self.renderWin.SetSize(WIDTH, HEIGHT)

        # create a renderwindowinteractor
        self.renderInteractor = vtk.vtkRenderWindowInteractor()
        self.renderInteractor.SetRenderWindow(self.renderWin)
        self.renderInteractor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.ren.ResetCamera()
        self.renderInteractor.Initialize()

    def addDevice(self):
        if self.device.geometryType == "cylindrical":
            for module in self.device.detectorModule:
                self.addModule(module)
        elif self.device.geometryType == "dualRotationSystemGeneric":
            for module in self.device._detectorModuleA:
                self.addModule(module)
            # for module in self.device._detectorModuleB:
            #     self.addModule(module)

    def addModule(self, module):
        if module is None:
            module = self.device
        # centers = [i.centroid for i in self.device.modelHighEnergyLightDetectors]
        # centers = np.array(centers).T
        # number_of_detectors = len(self.geometry_vector[0])
        # number_of_detectors = 8000
        for detector in module.modelHighEnergyLightDetectors:
            # Create a cube
            cube = vtk.vtkCubeSource()
            cube.SetXLength(detector.crystalSizeX)
            cube.SetYLength(detector.crystalSizeZ)
            cube.SetZLength(detector.crystalSizeY)
            cube.Update()
            # Rotate the cube
            translateRotationCube = vtk.vtkTransform()
            translateRotationCube.Translate(detector.centroid[0], detector.centroid[1], detector.centroid[2])

            translateRotationCube.RotateX(detector.alphaRotation)
            translateRotationCube.RotateY(detector.betaRotation)
            translateRotationCube.RotateZ(detector.sigmaRotation)

            labelTransform = vtk.vtkTransformPolyDataFilter()
            labelTransform.SetTransform(translateRotationCube)
            labelTransform.SetInputConnection(cube.GetOutputPort())

            colormap = vtk.vtkPolyDataMapper()
            colormap.SetInputConnection(labelTransform.GetOutputPort())
            detectorActor = vtk.vtkActor()
            detectorActor.SetMapper(colormap)
            detectorActor.GetProperty().SetColor([0.38, 0.34, 1])
            detectorActor.GetProperty().SetOpacity(0.85)

            # ca.color(red)
            self.ren.AddActor(detectorActor)


        for SiPM in module.modelVisibleLightSensors:
            # Create a cube
            cube = vtk.vtkCubeSource()
            cube.SetXLength(SiPM.blockSPiMWidth)
            cube.SetZLength(SiPM.blockSPiMHeight)
            cube.SetYLength(SiPM.blockSPiMDepth-SiPM.resinThickness)
            cube.Update()

            # Rotate the cube
            translateRotationCube = vtk.vtkTransform()
            translateRotationCube.Translate(SiPM.centerSiPMModule[0], SiPM.centerSiPMModule[1], SiPM.centerSiPMModule[2])
            translateRotationCube.RotateX(SiPM.alphaRotation)
            translateRotationCube.RotateY(SiPM.betaRotation)
            translateRotationCube.RotateZ(SiPM.sigmaRotation)

            labelTransform = vtk.vtkTransformPolyDataFilter()
            labelTransform.SetTransform(translateRotationCube)
            labelTransform.SetInputConnection(cube.GetOutputPort())

            colormap = vtk.vtkPolyDataMapper()
            colormap.SetInputConnection(labelTransform.GetOutputPort())
            detectorActor = vtk.vtkActor()
            detectorActor.SetMapper(colormap)
            detectorActor.GetProperty().SetColor([0.04, 0.36, 0.27])
            detectorActor.GetProperty().SetOpacity(0.75)

            for channel in range(len(SiPM.channelCentrePosition)):
                center = SiPM.channelCentrePosition[channel]
                individualChannel = vtk.vtkCubeSource()
                individualChannel.SetXLength(SiPM.effectiveWidth)
                individualChannel.SetZLength(SiPM.effectiveHeight)
                individualChannel.SetYLength(SiPM.resinThickness)
                individualChannel.Update()

                individualChannelTransform = vtk.vtkTransform()

                individualChannelTransform.Translate(center[0], center[1], center[2])
                individualChannelTransform.RotateX(SiPM.alphaRotation)
                individualChannelTransform.RotateY(SiPM.betaRotation)
                individualChannelTransform.RotateZ(SiPM.sigmaRotation)


                individualChannelLabelTransform = vtk.vtkTransformPolyDataFilter()
                individualChannelLabelTransform.SetTransform(individualChannelTransform)
                individualChannelLabelTransform.SetInputConnection(individualChannel.GetOutputPort())

                individualChannelColormap = vtk.vtkPolyDataMapper()
                individualChannelColormap.SetInputConnection(individualChannelLabelTransform.GetOutputPort())
                individualChannelActor = vtk.vtkActor()
                individualChannelActor.SetMapper(individualChannelColormap)
                individualChannelActor.GetProperty().SetColor([0.64, 0.64, 0.64])
                individualChannelActor.GetProperty().SetOpacity(1)
                self.ren.AddActor(individualChannelActor)


            # ca.color(red)
            self.ren.AddActor(detectorActor)

    def startRender(self):
        self.renderWin.Render()
        self.renderInteractor.Start()
