import matplotlib.pyplot as plt
import vtk
import os
import numpy as np
from scipy import ndimage
import tkinter as tk
from vtk.util import numpy_support
from src.Phantoms import NEMAIQ2008NU



class SliceViewer:
    def __init__(self, master, data3D=None, scale_z=0.5, scale_x=0.5, scale_y=0.5, colormap='hot.cl' ):
        self.data3D = data3D
        self.colormap = colormap
        direct = os.path.dirname(os.path.dirname(__file__))
        self.path_colormap = os.path.join(direct, "ControlUI", "colormap_files")
        self.phantom = NEMAIQ2008NU(centerPhantom=[0, 0, scale_z*data3D.shape[2] / 2])
        self.rods = [ self.phantom._rod1mm,  self.phantom._rod2mm,  self.phantom._rod3mm, self.phantom._rod4mm, self.phantom._rod5mm]

        self.cold_rods = [self.phantom._waterChamberFilling,  self.phantom._airChamberFilling]

        self.scale_x = scale_x

        self.scale_y = scale_y
        self.scale_z = scale_z
        self.master = master
        self.master.title("VTK with Tkinter Scale")

        self.frame = tk.Frame(master)
        self.frame.pack(fill=tk.BOTH, expand=1)

        self.vtk_widget = tk.Canvas(self.frame, width=800, height=600)
        self.vtk_widget.pack(fill=tk.BOTH, expand=1)

        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.interactorStyleImage = vtk.vtkInteractorStyleImage()
        self.interactorStyleImage.SetInteractionModeToImageSlicing()
        self.interactorStyle3D = vtk.vtkInteractorStyleTrackballCamera()


        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetInteractorStyle(self.interactorStyleImage)

        self.interactor.SetRenderWindow(self.render_window)
        self.renderer_entered = None
        self._listOfRenderers = []
        self._resliced_main_window = []
        self._min_color = 40
        self._max_color = 1300
        self._ambient = 1.1
        self._specular = 0.4
        self._volumeRotationX = 0
        self._volumeRotationY = 0
        self._volumeRotationZ = 0
        self._zoomVolume = 1
        self._maxValueForVolume = data3D.max()
        self._minValueForVolume = data3D.min()

    def maxValueForVolume(self):
        return self._maxValueForVolume

    def minValueForVolume(self):
        return self._minValueForVolume

    def setMaxValueForVolume(self, max_value):
        self._maxValueForVolume = max_value


    def setZoomVolume(self, zoom):
        self._zoomVolume = zoom

    def setVolumeRotation(self, alpha, beta, gamma):
        self._volumeRotationX = alpha
        self._volumeRotationY = beta
        self._volumeRotationZ = gamma

    def setColorMapLimits(self, v_lim):
        self._min_color = v_lim[0]
        self._max_color = v_lim[1]

    def setAmbient(self, ambient):
        self._ambient = ambient

    def setSpecular(self, specular):
        self._specular = specular

    def addObservers(self):
        self.interactorStyleImage.AddObserver("MouseMoveEvent",
                                              lambda obj, event: self.MouseMoveCallback(obj, event))
        self.interactorStyleImage.AddObserver("MouseWheelForwardEvent",
                                              lambda obj, event: self.wheelForwardCallback(obj, event))
        self.interactorStyleImage.AddObserver("MouseWheelBackwardEvent",
                                              lambda obj, event: self.wheelForwardCallback(obj, event))

    def addVolumeRendering(self):
        dims = self.data3D.shape
        dataImporter = vtk.vtkImageImport()
        data_string = self.data3D.tobytes()
        dataImporter.CopyImportVoidPointer(data_string, len(data_string))
        dataImporter.SetDataScalarTypeToFloat()
        dataImporter.SetNumberOfScalarComponents(1)
        dataImporter.SetDataExtent(0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)
        # dataImporter.SetDataExtent(-dims[0]*self.scale_x/2, dims[0]*self.scale_x/2, -dims[1]*self.scale_y/2, dims[1]*self.scale_y/2, -dims[2]*self.scale_z/2, dims[2]*self.scale_z/2)
        dataImporter.SetWholeExtent(0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)
        # dataImporter.SetWholeExtent(-dims[0]*self.scale_x/2, dims[0]*self.scale_x/2, -dims[1]*self.scale_y/2, dims[1]*self.scale_y/2, -dims[2]*self.scale_z/2, dims[2]*self.scale_z/2)
        dataImporter.Update()

        reader = dataImporter
        self.image_data = dataImporter.GetOutput()

        # Volume rendering setup
        volume_mapper = vtk.vtkSmartVolumeMapper()
        volume_mapper.SetInputConnection(reader.GetOutputPort())
        # volume_mapper.SetInterpolationModeToCubic()

        volume_property = vtk.vtkVolumeProperty()
        volume_property.ShadeOn()

        # volume_property.SetInterpolationTypeToLinear()
        # volume_property.SetInterpolationTypeToCubic()

        # composite_opacity = vtk.vtkPiecewiseFunction()
        # composite_opacity.AddPoint(0, 0.0)
        # composite_opacity.AddPoint(self.data3D.max(), 1.0)
        # volume_property.SetScalarOpacity(composite_opacity)

        color = vtk.vtkColorTransferFunction()
        color.SetColorSpaceToRGB()
        file_name = os.path.join(self.path_colormap, self.colormap)
        color_map_file = np.loadtxt(file_name)

        normalize_colormap = self._maxValueForVolume
        start_color = self._min_color  # 40
        end_color = self._max_color  # 1300

        increment_color = 0
        for row in range(0, len(color_map_file)):
            if row < start_color:
                color.AddRGBPoint(float(color_map_file[row][0]) * normalize_colormap,
                                  float(color_map_file[0][1]), float(color_map_file[0][2]), float(color_map_file[0][3]))
            elif row > end_color:
                color.AddRGBPoint(float(color_map_file[row][0]) * normalize_colormap,
                                  float(color_map_file[-1][1]), float(color_map_file[-1][2]),
                                  float(color_map_file[-1][3]))
            else:
                increment_color += len(color_map_file) / abs(start_color - end_color)
                if int(np.round(increment_color, 0)) < len(color_map_file):
                    color.AddRGBPoint(float(color_map_file[row][0]) * normalize_colormap,
                                      float(color_map_file[int(np.round(increment_color, 0))][1]),
                                      float(color_map_file[int(np.round(increment_color, 0))][2]),
                                      float(color_map_file[int(np.round(increment_color, 0))][3]))

        color.AddRGBPoint(0, float(color_map_file[0][1]), float(color_map_file[0][2]), float(color_map_file[0][3]))
        color.AddRGBPoint(self._maxValueForVolume, float(color_map_file[-1][1]), float(color_map_file[-1][2]),
                          float(color_map_file[-1][3]))

        alphaChannelFunc = vtk.vtkPiecewiseFunction()
        alphaChannelFunc.AddPoint(0, 0.0)
        alphaChannelFunc.AddPoint(self._maxValueForVolume * 0.07, 0.0)
        # alphaChannelFunc.AddPoint(self._maxValueForVolume * 0.2, 0.005)  # 0.03  #mice 0.005
          # 0.03  #mice 0.005

        alphaChannelFunc.AddPoint(self._maxValueForVolume, 1)
        # alphaChannelFunc.AddPoint(self._maxValueForVolume, 0.5)
        volume_property.SetColor(color)
        volume_property.SetInterpolationType(2)
        volume_property.SetAmbient(self._ambient)
        volume_property.SetSpecular(self._specular)
        volume_property.SetScalarOpacity(alphaChannelFunc)

        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)
        # set scale
        volume.SetScale(self.scale_x, self.scale_y, self.scale_z)

        volume_renderer = vtk.vtkRenderer()
        volume_renderer.AddVolume(volume)
        volume_renderer.SetBackground(1, 1, 1)
        volume_renderer.SetBackground(0, 0, 0)
        # volume_renderer.SetBackground(color_map_file[0][1], color_map_file[0][2], color_map_file[0][3])

        volume.RotateX(self._volumeRotationX) # 90
        volume.RotateY(self._volumeRotationY) # 75
        volume.RotateZ(self._volumeRotationZ) # 60
        volume_renderer.ResetCamera()
        volume_renderer.GetActiveCamera().Zoom(self._zoomVolume)

        return reader, volume_renderer, color_map_file, color

    def derenzo(self, outputFilename=None, recalculate_max_value=True):
        # Read the volume data
        self.data3D = self.prepare_data(voxeldata=self.data3D)
        if recalculate_max_value:
            self._maxValueForVolume = self.data3D.max()
            self._minValueForVolume = self.data3D.min()
        # Convert self.data3D to VTK format
        reader, volume_renderer, color_map_file, color = self.addVolumeRendering()
        self.interactor.SetInteractorStyle(self.interactorStyle3D)
        self.render_window.AddRenderer(volume_renderer)

        # Add title
        self.renderer = volume_renderer
        self.renderWindow((600,600))
        self.saveImageInPDF(outputFilename=outputFilename)

        self.interactor.Start()


    def renderWindow(self, size=(1875, 1875)): #1875
        self.render_window.SetSize(size)
        # self.save_screenshot()
        self.interactor.Initialize()
        self.render_window.Render()

    def saveImageInPDF(self, outputFilename=None):
        pdfExporter = vtk.vtkGL2PSExporter()
        pdfExporter.SetRenderWindow(self.render_window)

        if outputFilename is None:
            outputFilename = "../../outputs/teste"

        pdfExporter.SetFileFormatToPDF()  # save as pdf
        pdfExporter.SetFilePrefix(outputFilename)
        pdfExporter.Write()

        # save as png
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(self.render_window)
        window_to_image_filter.SetScale(1)  # Image quality
        window_to_image_filter.SetInputBufferTypeToRGBA()
        window_to_image_filter.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(outputFilename + ".png")
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        # 300 dpi

        writer.Write()





    def MiceNaf(self, filename=None, recalculate_max_value=True, only_scalar_bar=False, render_volume=True, window_size=(600, 600)):
        self.data3D = self.prepare_data(voxeldata=self.data3D)
        if recalculate_max_value:
            self._maxValueForVolume = self.data3D.max()
            self._minValueForVolume = self.data3D.min()

        # Convert self.data3D to VTK format

        reader, volume_renderer, color_map_file, color = self.addVolumeRendering()
        if only_scalar_bar:
            filename = filename + "_scalar_bar"
            overlay_renderer = vtk.vtkRenderer()
            overlay_renderer.SetViewport(0.0, 0.0, 1.0, 1.0)
            overlay_renderer.SetLayer(1)
            self.render_window.SetNumberOfLayers(2)
            self.render_window.AddRenderer(overlay_renderer)
            #add colorbar
            scaleActor = vtk.vtkScalarBarActor()
            scaleActor.SetLookupTable(color)
            scaleActor.SetOrientationToVertical()


            scaleActor.SetNumberOfLabels(4)
            scaleActor.SetLabelFormat("%2.1f")
            scaleActor.SetPosition(0.1, 0.1)
            scaleActor.SetWidth(0.5)
            scaleActor.SetHeight(0.8)
            # lavls in black
            scaleActor.GetLabelTextProperty().SetColor(0, 0, 0)

            # scaleActor.SetTitle("Activity (MBq/ml)")
            # Create a text actor for the title
            title_actor = vtk.vtkTextActor()
            title_actor.SetInput("SUV")

            # Customize title text properties
            title_text_property = title_actor.GetTextProperty()
            title_text_property.SetFontSize(24)
            title_text_property.SetJustificationToCentered()
            title_text_property.SetVerticalJustificationToCentered()
            title_text_property.SetOrientation(90)  # Rotate text 90 degrees
            title_text_property.SetColor(0, 0, 0)  # White color text for better visibility

            # Position the title actor to the right of the scalar bar
            title_actor.SetPosition(92, 300)

            overlay_renderer.AddActor(scaleActor)
            overlay_renderer.AddActor((title_actor))
            # self.renderWindow(size=(73, 600))
            self.renderWindow(size=(73, 600))
        if render_volume:

            self.render_window.AddRenderer(volume_renderer)
            self.interactor.SetInteractorStyle(self.interactorStyle3D)
            # Add title
            self.renderer.SetUseDepthPeeling(1)
            self.renderWindow(size=window_size)


        self.saveImageInPDF(outputFilename=filename)
        #
        # self.createMovie(filename=filename + ".ogv")
        self.interactor.Start()

    def NemaIQ2008(self, file_to_save=None, only_scalar_bar=True, recalculate_max_value=True):

        # Read the volume data
        self.data3D = self.prepare_data(voxeldata=self.data3D)
        if recalculate_max_value:
            self._maxValueForVolume = self.data3D.max()
            self._minValueForVolume = self.data3D.min()

        # Convert self.data3D to VTK format
        reader, volume_renderer, color_map_file, color = self.addVolumeRendering()
        self.renderer.SetUseDepthPeeling(1)

        slice_x_renderer, slice_y_renderer, slice_z_renderer, volume_renderer, mip_reslice_x, mip_reslice_y, mip_reslice_z = self.addViewportsNema(reader, volume_renderer, color_map_file, color)

        self._listOfRenderers = [volume_renderer, slice_x_renderer, slice_y_renderer, slice_z_renderer]
        self._resliced_main_window = [mip_reslice_x, mip_reslice_y, mip_reslice_z, None]
        # add observers
        # self.interactor.AddObserver("MouseMoveEvent", self.MouseMoveCallback)

        self.addObservers()


        # Create an overlay renderer for the scalar bar
        if only_scalar_bar:
            overlay_renderer = vtk.vtkRenderer()
            overlay_renderer.SetViewport(0.0, 0.0, 1.0, 1.0)
            overlay_renderer.SetLayer(1)
            self.render_window.SetNumberOfLayers(2)
            self.render_window.AddRenderer(overlay_renderer)
            #add colorbar
            scaleActor = vtk.vtkScalarBarActor()
            scaleActor.SetLookupTable(color)
            scaleActor.SetOrientationToVertical()


            scaleActor.SetNumberOfLabels(4)
            scaleActor.SetLabelFormat("%2.1f")
            scaleActor.SetPosition(0.1, 0.1)
            scaleActor.SetWidth(0.5)
            scaleActor.SetHeight(0.8)
            # lavls in black
            scaleActor.GetLabelTextProperty().SetColor(0, 0, 0)

            # scaleActor.SetTitle("Activity (MBq/ml)")
            # Create a text actor for the title
            title_actor = vtk.vtkTextActor()
            title_actor.SetInput("Activity (MBq/ml)")

            # Customize title text properties
            title_text_property = title_actor.GetTextProperty()
            title_text_property.SetFontSize(24)
            title_text_property.SetJustificationToCentered()
            title_text_property.SetVerticalJustificationToCentered()
            title_text_property.SetOrientation(90)  # Rotate text 90 degrees
            title_text_property.SetColor(0, 0, 0)  # White color text for better visibility

            # Position the title actor to the right of the scalar bar
            title_actor.SetPosition(92, 300)

            overlay_renderer.AddActor(scaleActor)
            overlay_renderer.AddActor((title_actor))
            self.renderWindow(size=(73, 600))
        else:
        # Add title
            self.render_window.AddRenderer(volume_renderer)
            self.render_window.AddRenderer(slice_x_renderer)
            self.render_window.AddRenderer(slice_y_renderer)
            self.render_window.AddRenderer(slice_z_renderer)
            self.renderWindow()
        self.saveImageInPDF(outputFilename=file_to_save)
        #
        self.interactor.Start()



    def addViewportsNema(self, reader, volume_renderer, color_map_file, color):

        slice_x_renderer = vtk.vtkRenderer()
        slice_y_renderer = vtk.vtkRenderer()
        slice_z_renderer = vtk.vtkRenderer()

        self.create_viewport(slice_z_renderer, [0.0, 0.0, 0.5, 0.5])
        self.create_viewport(slice_x_renderer, [0.0, 0.5, 0.5, 1.0])
        self.create_viewport(slice_y_renderer, [0.5, 0.5, 1.0, 1.0])
        self.create_viewport(volume_renderer, [0.5, 0.0, 1.0, 0.5])
        center = self.rods[0].center
        print(center)
        center[2] -= 5+1
        center[0] /= self.scale_x
        center[1] /= self.scale_y
        center[2] /= self.scale_z
        mip_reslice_x = self.create_reslice(reader.GetOutputPort(), slab_thickness=7, slab_mode="Max", center=center)
        color_mapper_x = vtk.vtkImageMapToColors()
        color_mapper_x.SetLookupTable(color)
        color_mapper_x.SetInputConnection(mip_reslice_x.GetOutputPort())
        color_mapper_x.Update()

        slice_x_mapper = vtk.vtkImageResliceMapper()
        slice_x_mapper.SetInputConnection(color_mapper_x.GetOutputPort())
        slice_x_mapper.SliceFacesCameraOn()
        slice_x_mapper.BorderOff()

        # set interpolation type



        slice_x_actor = vtk.vtkImageSlice()
        slice_x_actor.SetMapper(slice_x_mapper)
        # slice_x_actor.RotateY(90)
        slice_x_renderer.AddViewProp(slice_x_actor)
        slice_x_renderer.SetBackground(0, 0, 0)
        #first color of the colormap is background
        slice_x_renderer.SetBackground(color_map_file[0][1], color_map_file[0][2], color_map_file[0][3])
        slice_x_renderer.ResetCamera()
        slice_x_renderer.GetActiveCamera().Zoom(1)
        slice_x_renderer.GetActiveCamera().Roll(90)

        center = self.cold_rods[0].center
        print(center)
        center[2] -= 5+2
        center[0] /= self.scale_x
        center[1] /= self.scale_y
        center[2] /= self.scale_z
        mip_reslice_y = self.create_reslice(reader.GetOutputPort(), slab_thickness=20, slab_mode="Min", center=center)
        color_mapper_y = vtk.vtkImageMapToColors()
        color_mapper_y.SetLookupTable(color)
        color_mapper_y.SetInputConnection(mip_reslice_y.GetOutputPort())
        color_mapper_y.Update()

        slice_y_mapper = vtk.vtkImageResliceMapper()
        slice_y_mapper.SetInputConnection(color_mapper_y.GetOutputPort())
        slice_y_mapper.SliceFacesCameraOn()
        slice_y_mapper.SliceAtFocalPointOn()
        slice_y_mapper.BorderOff()

        slice_y_actor = vtk.vtkImageSlice()
        slice_y_actor.SetMapper(slice_y_mapper)

        slice_y_renderer.AddViewProp(slice_y_actor)
        slice_y_renderer.SetBackground(0, 0, 0)
        slice_y_renderer.SetBackground(color_map_file[0][1], color_map_file[0][2], color_map_file[0][3])
        slice_y_renderer.ResetCamera()
        slice_y_renderer.GetActiveCamera().Zoom(1)
        slice_y_renderer.GetActiveCamera().Roll(90)

        # middle slice center of the phantom
        center = [self.scale_x*self.data3D.shape[0]/2, self.scale_y*self.data3D.shape[1]/2, self.scale_z*self.data3D.shape[2]/2]
        print(center)
        center = self.cold_rods[1].center
        center[2] -= 5
        center[0] = center[0]*self.scale_x +self.scale_x*self.data3D.shape[0] -3

        print(center)
        mip_reslice_z = self.create_reslice(reader.GetOutputPort(), slab_thickness=7, slab_mode="Max", rotation_type="Sagittal", center=center)
        color_mapper_z = vtk.vtkImageMapToColors()
        color_mapper_z.SetLookupTable(color)
        color_mapper_z.SetInputConnection(mip_reslice_z.GetOutputPort())
        color_mapper_z.Update()

        slice_z_mapper = vtk.vtkImageResliceMapper()
        slice_z_mapper.SetInputConnection(color_mapper_z.GetOutputPort())
        slice_z_mapper.SliceFacesCameraOn()
        slice_z_mapper.SliceAtFocalPointOn()
        slice_z_mapper.BorderOff()

        slice_z_actor = vtk.vtkImageSlice()
        slice_z_actor.SetMapper(slice_z_mapper)
        slice_z_renderer.AddViewProp(slice_z_actor)
        slice_z_renderer.SetBackground(0, 0, 0)
        slice_z_renderer.SetBackground(color_map_file[0][1], color_map_file[0][2], color_map_file[0][3])
        slice_z_renderer.ResetCamera()
        slice_z_renderer.GetActiveCamera().Zoom(1.5)
        slice_z_renderer.GetActiveCamera().Roll(180)



        return slice_x_renderer, slice_y_renderer, slice_z_renderer, volume_renderer, mip_reslice_x, mip_reslice_y, mip_reslice_z
        # slice_z_renderer.GetActiveCamera().Yaw(90)

        # textActor = vtk.vtkTextActor()
        # textActor.SetTextScaleModeToNone()
        # textActor.GetTextProperty().SetFontSize(24)
        # textActor.GetTextProperty().SetColor(1, 1, 1)
        # textActor.SetInput("NEMA IQ 2008 Phantom")
        # textActor.SetPosition(10, 10)
        # self.renderer.AddActor(textActor)
        # # Add scale
        # scaleActor = vtk.vtkScalarBarActor()
        # scaleActor.SetLookupTable(color)
        # scaleActor.SetTitle("Activity (Bq/ml)")
        # scaleActor.SetNumberOfLabels(5)
        # scaleActor.SetLabelFormat("%6.1f")
        # scaleActor.SetPosition(0.1, 0.1)
        # scaleActor.SetWidth(0.8)
        # scaleActor.SetHeight(0.1)
        # scaleActor.SetOrientationToHorizontal()
        # scaleActor.GetTitleTextProperty().SetColor(1, 1, 1)
        # scaleActor.GetLabelTextProperty().SetColor(1, 1, 1)
        # self.renderer.AddActor(scaleActor)



        # Add save button
        # self.save_button = tk.Button(master, text="Save Screenshot", command=self.save_screenshot)
        # self.save_button.pack()

        # # Save render window as png

        # writer = vtk.vtkPNGWriter()
        # writer.SetFileName("../../outputs/NeMa_renderODRTVF.png")
        # writer.SetInputConnection(windowToImageFilter.GetOutputPort())
        # writer.Write()
    def createMovie(self, filename):
        # perform a 360 degree rotation of the volume
        # create a camera

        self.renderer.SetUseDepthPeeling(1)
        # self.ren_volume.SetBackground(0, 0, 0)  # black

        # self.ren_volume.ResetCamera()

        # self.renderWin = renderWin
        self.interactor.Initialize()
        # self.call_backs = [None] * len(self.detectors)
        # for i in range(len(self.detectors)):
        # for i in range(5):
        camera = self.renderer.GetActiveCamera()
        camera.Azimuth(90)
        camera.Roll(-90)
        # camera.Pitch(3)
        # camera.Elevation(5)
        # camera.Yaw(5)
        camera.SetViewAngle(15)
        # camera.SetFocalDistance()
        # camera.SetClippingRange(0,400)

        windowToImageFilter = vtk.vtkWindowToImageFilter()
        windowToImageFilter.SetInput(self.render_window)
        # windowToImageFilter.SetInputBufferTypeToRGBA()
        # windowToImageFilter.ReadFrontBufferOff()
        windowToImageFilter.Update()

        # writer = vtk.vtkAVIWriter()
        # writer.SetInputConnection(windowToImageFilter.GetOutputPort())
        # writer.SetFileName("test.avi")
        # writer.Start()
        #
        moviewriter = vtk.vtkOggTheoraWriter()
        moviewriter.SetInputConnection(windowToImageFilter.GetOutputPort())
        moviewriter.SetFileName(filename)
        moviewriter.Start()
        ca = vtkTimerCallback(moviewriter, windowToImageFilter)
        ca.camera = camera
        self.interactor.AddObserver('TimerEvent', ca.execute)
        timerId = self.interactor.CreateRepeatingTimer(100)

        # Because nothing will be rendered without any input, we order the first render manually before control is handed over to the main-loop.
        self.renderer.Render()
        self.interactor.Start()
        self.moviewriter = moviewriter
        self.moviewriter.End()


    def save_screenshot(self):
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(self.render_window)
        window_to_image_filter.SetScale(1)  # Image quality
        window_to_image_filter.SetInputBufferTypeToRGBA()
        window_to_image_filter.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName("screenshot.png")
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        writer.Write()

    def prepare_data(self, voxeldata=None, colormap=None, path=None, z_pos=None, data3D=None):

        # voxeldata = np.random.rand(100, 100, 100)  # Example data

        z_pos = voxeldata.shape[2]
        data3D = voxeldata

        # data3D = ndimage.median_filter(data3D, size=3)
        # data3D = ndimage.gaussian_filter(data3D, sigma=0.57)
        # data3D =ndimage.filters.convolve(data3D, np.ones((3, 1, 1)) / 9)
        # wavelet filter on the image
        import pywt
        # Reconstruct the denoised image
        from skimage import data, img_as_float
        from skimage.restoration import denoise_nl_means
        # Adjust the parameters according to your requirements
        patch_size = 1  # Size of patches used for denoising
        patch_distance = 1 # Maximum distance for patches to be considered similar
        h = 0.0005  # Smoothing parameter (higher values give stronger smoothing)

        # Apply non-local means denoising
        # data3D = denoise_nl_means(data3D, patch_size=patch_size, patch_distance=patch_distance, h=h)

        # denoised_image = pywt.idwtn(coeffs_thresh, 'haar')




        xx = (np.tile(np.arange(0, data3D.shape[0]), (data3D.shape[0], 1)) - (data3D.shape[0] - 1) / 2) ** 2
        yy = (np.tile(np.arange(0, data3D.shape[1]), (data3D.shape[1], 1)) - (data3D.shape[1] - 1) / 2) ** 2
        xx = xx.T
        try:
            circle_cut = xx + yy - (data3D.shape[1] * 0.5) ** 2
            circle_cut[circle_cut > 0] = 0
            circle_cut[circle_cut < 0] = 1
            circle_cut = np.tile(circle_cut[:, :, None], (1, 1, data3D.shape[2]))

            data3D = data3D * circle_cut
        except ValueError:
            pass
        data3D[:, :, :14] = 0 # 5
        data3D[:, :, -25 :] = 0 #-5
        # data3D[:, :7, :] = 0  # 5
        # data3D[:, -7:, :] = 0  # -5
        # transforme array for the axis 1 has the same shape as the axis 0
        add_missing = data3D.shape[0] - data3D.shape[1]
        print("add missing", add_missing)
        if add_missing > 0:
            data3D = np.pad(data3D, ((0, 0), (0, add_missing), (0, 0)), 'constant')
        elif add_missing < 0:
            data3D = np.pad(data3D, ((0, -add_missing), (0, 0), (0, 0)), 'constant')
        size = data3D.shape
        w, h, d = size
        stack = np.zeros((w, d, h))
        for j in range(0, z_pos):
            stack[:, j, :] = data3D[:, :, j]

        stack = np.require(stack, dtype=np.float32)

        # data is with the wrong order per slice
        # stack = np.swapaxes(stack, 0, 2)
        #add 200 zeros in axis 0
        #the
        # stack = np.swapaxes(stack, 1, 2)


        return stack

    def create_viewport(self, renderer, viewport):
        renderer.SetViewport(viewport[0], viewport[1], viewport[2], viewport[3])

    def create_reslice(self, input_port, slab_thickness, slab_mode, rotation_type="Coronal", center=[0, 0, 0],
                       azimuthal_angle=0, polar_angle=0):
        newResliceAxes = vtk.vtkMatrix4x4()

        if rotation_type == "Axial":
            newResliceAxes.DeepCopy((1, 0, 0, center[0],
                                     0, 1, 0, center[2],
                                     0, 0, 1, center[1],
                                     0, 0, 0, 1))
        elif rotation_type == "Coronal":
            newResliceAxes.DeepCopy((1, 0, 0, center[0],
                                        0, 0, 1, center[2],
                                        0, -1, 0, center[1],
                                        0, 0, 0, 1))
        elif rotation_type == "Sagittal":
            newResliceAxes.DeepCopy((0, 0, 1, center[0],
                                        0, 1, 0, center[2],
                                        -1, 0, 0, center[1],
                                        0, 0, 0, 1))


        reslice = vtk.vtkImageSlabReslice()
        reslice.SetInputConnection(input_port)
        reslice.SetResliceAxes(newResliceAxes)
        reslice.SetSlabThickness(slab_thickness)
        reslice.SetInterpolationModeToCubic()

        if slab_mode == "Mean":
            reslice.SetBlendModeToMean()
        elif slab_mode == "Max":
            reslice.SetBlendModeToMax()
        elif slab_mode == "Min":
            reslice.SetBlendModeToMin()
        reslice.Update()
        return reslice

    def MouseMoveCallback(self, obj, event):
        (lastX, lastY) = self.interactor.GetLastEventPosition()
        (mouseX, mouseY) = self.interactor.GetEventPosition()
        renderer_entered = self.interactor.FindPokedRenderer(mouseX, mouseY)
        # print('Mouse moving')
        # print(renderer_entered)
        if self._listOfRenderers[0] == renderer_entered:
            # print('3D')
            # pass
            self.interactor.SetInteractorStyle(
                self.interactorStyle3D)
            self.interactorStyle3D.OnMouseMove()

        else:
            # print('Slicers')
            # self.interactorStyleImage = vtk.vtkInteractorStyleImage()
            # self.interactorStyleImage.SetInteractionModeToImageSlicing()
            # self.interactorMainWindow.SetInteractorStyle(self.interactorStyleImage)
            # index_renderer = self.parent.populateMainWindowVTK.listOfRenderers.index(
            #     renderer_entered) - 1  # testing -- put as dict
            # if self.actions["Slicing"] == 1:
            #     deltaY = mouseY - lastY
            #     # self.resliced_main_window[index_renderer].Update()
            #     sliceSpacing = \
            #     self.parent.populateMainWindowVTK.resliced_main_window[index_renderer].GetOutput().GetSpacing()[2]
            #     matrix = self.parent.populateMainWindowVTK.resliced_main_window[index_renderer].GetResliceAxes()
            #     center = matrix.MultiplyPoint((0, 0, sliceSpacing * deltaY, 1))
            #     matrix.SetElement(0, 3, center[0])
            #     matrix.SetElement(1, 3, center[1])
            #     matrix.SetElement(2, 3, center[2])
            #     self.parent.populateMainWindowVTK.vtkWidget.GetRenderWindow().Render()
            #
            # else:
            self.interactorStyleImage.SetInteractionModeToImageSlicing()
            self.interactor.SetInteractorStyle(
                self.interactorStyleImage)

            self.interactorStyleImage.OnMouseMove()

            self.renderer_entered = renderer_entered

    # override scroll event and  update reslice
    def wheelForwardCallback(self, obj, event):
        if event == "MouseWheelForwardEvent":
            increment = 2
        elif event == "MouseWheelBackwardEvent":
            increment = -2
        index_renderer = self._listOfRenderers.index(self.renderer_entered)-1
        # if self.parent.populateMainWindowVTK.resliced_main_window[index_renderer] is None:
        #     return

        sliceSpacing =  self._resliced_main_window[index_renderer].GetOutput().GetSpacing()[2]
        matrix = self._resliced_main_window[index_renderer].GetResliceAxes()
        center = matrix.MultiplyPoint((0, 0, sliceSpacing * increment, 1))
        matrix.SetElement(0, 3, center[0])
        matrix.SetElement(1, 3, center[1])
        matrix.SetElement(2, 3, center[2])
        # set the reslice axes
        self._resliced_main_window[index_renderer].SetResliceAxes(matrix)

        self.render_window.Render()


class vtkTimerCallback():
   def __init__(self, writer=None, imageFilter=None):
       self.timer_count = 0
       self.writer = writer
       self.imageFilter = imageFilter

   def execute(self,obj,event):
       print(self.timer_count)
       # self.actor.SetPosition(self.timer_count, self.timer_count,0)
       if self.timer_count <360:
           focal_position =np.array([1000,1000,500])
           alpha = np.deg2rad(self.timer_count)
           beta = 0
           gamma = 0
           rotation_matrix = np.array([[np.cos(alpha) * np.cos(beta),
                                        np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma),
                                        np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma)],
                                       [np.sin(alpha) * np.cos(beta),
                                        np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma),
                                        np.sin(alpha) * np.sin(beta) * np.sin(gamma) - np.cos(alpha) * np.sin(gamma)],
                                       [-np.sin(beta),
                                        np.cos(beta) * np.sin(gamma),
                                        np.cos(beta) * np.cos(gamma)]], dtype=np.float32)
           focal_position = (np.dot(focal_position, rotation_matrix))

           self.camera.SetPosition(focal_position)
       # elif self.timer_count <370:
       #     self.camera.Zoom((self.timer_count-360)*.1+1)
       #
       # else:
       #     focal_position = np.array([1000, 1000, 0])
       #     alpha = np.deg2rad(self.timer_count)
       #     beta = 0
       #     gamma = 0
       #     rotation_matrix = np.array([[np.cos(alpha) * np.cos(beta),
       #                                  np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma),
       #                                  np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma)],
       #                                 [np.sin(alpha) * np.cos(beta),
       #                                  np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma),
       #                                  np.sin(alpha) * np.sin(beta) * np.sin(gamma) - np.cos(alpha) * np.sin(gamma)],
       #                                 [-np.sin(beta),
       #                                  np.cos(beta) * np.sin(gamma),
       #                                  np.cos(beta) * np.cos(gamma)]], dtype=np.float32)
       #     focal_position = (np.dot(focal_position, rotation_matrix))
       #     self.camera.SetPosition(focal_position)
       # self.camera.SetViewUp(1,-1,0)
       # self.camera.GetViewUp()
       # self.camera.OrthogonalizeViewUp()
       # self.camera.Roll(90)
       iren = obj
       iren.GetRenderWindow().Render()
       self.timer_count += 1
       self.imageFilter.Modified()
       self.writer.Write()


if __name__ == "__main__":
    from src.ImageReader import RawDataSetter
    from src.Quantification import FactorQuantificationFromUniformPhantom
    from src.Geometry import GeometryDesigner
    filepath = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\SimulacoesGATE\\EasyPET3D64\\NEMA-NU-4-2008-IQ\\15-December-2022ListMode\\whole_body\\ID_26 Jan 2022 - 00h 16m 02s_1p80bot_ IMAGE (71, 71, 129).T"
    filepath = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\SimulacoesGATE\\EasyPET3D64\\NEMA-NU-4-2008-IQ\\15-December-2022ListMode\\Data articles\\gpu_article\\IQ\\OD-RT varying FWHM\\iterations\\EasyPETScan_it20_sb0"
    filepath = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\SimulacoesGATE\\EasyPET3D64\\NEMA-NU-4-2008-IQ\\15-December-2022ListMode\\Data articles\\gpu_article\\IQ\\TOR\\iterations\\EasyPETScan_it20_sb0"
    filepath = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\ICNAS\\Acquisitions 2022\\Easypet Scan 16 May 2022 - 10h 16m 52s\\whole_body\\ID_16 May 2022 - 10h 16m 52s_None_ IMAGE (71, 71, 129).T"
    filepath = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\ICNAS\\Acquisitions 2022\\Easypet Scan 16 May 2022 - 10h 16m 52s\\whole_body\\ID_16 May 2022 - 10h 16m 52s_None_ IMAGE (71, 71, 129)_bestODRTVF.T"
    filepath = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\ICNAS\\Acquisitions 2022\\Easypet Scan 16 May 2022 - 10h 16m 52s\\whole_body\\ID_16 May 2022 - 10h 16m 52s_None_ IMAGE (89, 89, 161).T"
    filepath = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\ICNAS\\Acquisitions 2022\\Easypet Scan 16 May 2022 - 10h 16m 52s\\whole_body\\ID_16 May 2022 - 10h 16m 52s_None_ IMAGE (89, 89, 161)_32corrected_ODRTVF.T"
    filepath = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\ICNAS\\Acquisitions 2022\\Easypet Scan 16 May 2022 - 10h 16m 52s\\whole_body\\ID_16 May 2022 - 10h 16m 52s_None_ IMAGE (118, 118, 216).T"
    # filepath = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\ICNAS\\Acquisitions 2022\\Easypet Scan 16 May 2022 - 10h 16m 52s\\whole_body\\ID_16 May 2022 - 10h 16m 52s_None_ IMAGE (89, 89, 161)_32corrected_TOR.T"
    # filepath = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\ICNAS\\Acquisitions 2022\\Easypet Scan 16 May 2022 - 10h 16m 52s\\whole_body\\ID_16 May 2022 - 10h 16m 52s_None_ IMAGE (89, 89, 161)_32corrected_ODRT.T"
    # filepath = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\ICNAS\\Acquisitions 2022\\Easypet Scan 16 May 2022 - 10h 16m 52s\\whole_body\\ID_16 May 2022 - 10h 16m 52s_None_ IMAGE (89, 89, 161).T"
    # filepath ="C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\SimulacoesGATE\EasyPET3D64\\NEMA-NU-4-2008-IQ\\15-December-2022ListMode\\Data articles\\gpu_article\\IQ\\TOR\\iterations\\EasyPETScan_it28_sb0"
    filepath = "C:\\Users\\pedro\\OneDrive\\Ambiente de Trabalho\\DESKTOP Organizar\\GPU article\\ODTR\\Derenzo\\ID_26 Jan 2022 - 00h 16m 02s_GammasBackToBack.npy_ IMAGE (176, 176, 321).T"
    filepath = "C:\\Users\\pedro\\OneDrive\\Ambiente de Trabalho\\DESKTOP Organizar\\GPU article\\TOR\\Derenzo\\ID_26 Jan 2022 - 00h 16m 02s_GammasBackToBack.npy_ IMAGE (176, 176, 321) .T"
    # filepath = "C:\\Users\\pedro\\OneDrive\\Ambiente de Trabalho\\DESKTOP Organizar\\GPU article\\ODTRVF\\Derenzo\\ID_26 Jan 2022 - 00h 16m 02s_GammasBackToBack.npy_ IMAGE (176, 176, 321).T"
    # filepath = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\ICNAS\\Acquisitions 2022\\Easypet Scan 16 May 2022 - 10h 16m 52s\\iterations\\EasyPETScan_it24_sb0"
    # filepath = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\ICNAS\\Acquisitions 2022\\Easypet Scan 16 May 2022 - 10h 16m 52s\\\Data article\\ODRT\\EasyPETScan_it28_sb0"
    # filepath = "C:\\Users\\pedro\\OneDrive\\Ambiente de Trabalho\\DESKTOP Organizar\\GPU article\\ODTRVF\\Mice\\ID_16 May 2022 - 10h 16m 52s_None_ IMAGE (118, 118, 216)_article.T"
    # filepath = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\ICNAS\\Acquisitions 2022\\Easypet Scan 16 May 2022 - 10h 16m 52s\\iterations\\EasyPETScan_it28_sb0"
    # filepath = "C:\\Users\\pedro\\OneDrive\\Ambiente de Trabalho\\Images intro\\Cu64\\MRI_Cu64_7_Dec_00h48"
    # filepath = "C:\\Users\\pedro\\OneDrive\\Ambiente de Trabalho\\Images intro\\FDG_cardiac\\ID_06 Jul 2023 - 12h 29m 51s_Test Slow_ IMAGE (89, 89, 161).T"
    size_file_m = [129, 129, 216]
    size_file_m = [129, 129, 161]
    size_file_m = [118, 118, 216]
    # size_file_m = [89, 89, 161]
    # size_file_m = [200, 400,50]
    # size_file_m = [71, 71, 124]
    # size_file_m = [71, 71, 129]
    size_file_m = [176, 176, 321]
    # size_file_m = [104,104,129]

    # size_file_m = [78,78,129]
    # size_file_m = [53,53,65]
    r = RawDataSetter(filepath, size_file_m=size_file_m)
    # r = RawDataSetter(filepath)
    r.read_files(type_file="float32", big_endian=True)
    voxeldata = r.volume

    # set to zero the pixels in which the neighbor is much lo

    # voxeldata[:, :, -2:] = 0
    # voxeldata[:, :, :2] = 0
    v_lim = np.array([0.02, 0.8])
    scale_x = float(0.2)
    scale_y = float(0.2)
    scale_z = float(0.2 / (0.4 / 0.44))
    #mri
    # scale_x = 0.225
    # scale_y = 0.5
    # scale_z = 0.2
    extent_x_y = [-scale_x * voxeldata.shape[0] / 2, scale_x * voxeldata.shape[0] / 2,
                  -scale_y * voxeldata.shape[1] / 2, scale_y * voxeldata.shape[1] / 2]

    # generate coordinates
    height = r.size_file_m[0]
    width = r.size_file_m[1]
    x = np.arange(0, width)
    y = np.arange(0, width)
    xx, yy = np.meshgrid(x, y)

    quantification = FactorQuantificationFromUniformPhantom(voxel_volume=scale_x * scale_y * scale_z*0.001)
    quantification.load_info()
    quantification_factor = quantification.quantification_factor
    print(quantification_factor)
    quantification_factor=0.011
    acquisition_time = 6586.2*7.2  #F18 half-life 109.77 min and in s 6586.2
    voxel_volume = scale_x * scale_y * scale_z*0.001
    # voxeldata =  voxeldata / (acquisition_time * voxel_volume)
    # voxeldata/=1*(10**6)

    # voxeldata *= quantification_factor
    # actidade Media
    #mice
    # voxeldata = voxeldata/(381*37000/17.33)


    root = tk.Tk()
    #derenzo
    voxeldata = np.flip(voxeldata, axis=0)

    viewer = SliceViewer(root, data3D=voxeldata, scale_x=scale_x, scale_y=scale_y, scale_z=scale_z, colormap="hot.cl")
    #Nema
    # viewer.setColorMapLimits(v_lim=[20, 1080])
    # viewer.setZoomVolume(2)
    # viewer.setVolumeRotation(90, 75, 60)
    # # viewer.setMaxValueForVolume(300*37000/22/(1*10**6))
    # viewer.NemaIQ2008(file_to_save="../../outputs/NemaIQ2008_renderTOR", only_scalar_bar=False)
    # viewer.NemaIQ2008(file_to_save="../../outputs/NemaIQ2008_colorbar", only_scalar_bar=True)

    # mice
    # name_to_record ="../../outputs/MiceNAF_test_{}".format(os.path.basename(os.path.dirname(os.path.dirname(filepath))).split(".")[0].split("_")[-1])
    # only_scalar_bar = False
    # viewer.setColorMapLimits(v_lim=[10,750]) # 40, 750
    # viewer.setZoomVolume(1.4)
    # viewer.setVolumeRotation(0, -80, 10)
    # viewer.setMaxValueForVolume(14)
    # viewer.MiceNaf(filename=name_to_record, only_scalar_bar=only_scalar_bar, render_volume=True, recalculate_max_value=True)
    #
    # # # add masked image in the same window
    # maskFilename = "../../outputs/mask_skull_name.png"
    #
    # # using PIL
    # if only_scalar_bar is not True:
    #     from PIL import Image, ImageDraw
    #     # image_mask = Image.open(maskFilename)
    #     image = Image.open(name_to_record + ".png")
    #     image.convert("RGBA")
    #     # canvas = Image.new('RGBA', image.size, color="#")  # Empty canvas colour (r,g,b,a)
    #     #new canvas with black background
    #     canvas = Image.new('RGB', image.size, color="#000000")
    #     #black background
    #     background = np.zeros((image.size[1], image.size[0], 3), dtype=np.uint8)
    #     maskedImage = Image.open(maskFilename)
    #     maskedImage.convert("RGBA")
    #
    #     canvas.paste(image, mask=image)  # Paste the image onto the canvas, using it's alpha channel as mask
    #     canvas.paste(maskedImage, mask=maskedImage)  # Paste the image onto the canvas, using it's alpha channel as mask
    #     canvas.save(name_to_record + "_mask.png", resolution=300.0)

    # viewer.createMovie(filename=name_to_record+".avi")


    #Derenzo
    viewer.setColorMapLimits(v_lim=[75, 1080])
    viewer.setZoomVolume(2.5)
    view = "37.5"
    # view = "axial"
    if view == "axial":
        viewer.setVolumeRotation(90, -90, 90)
    elif view == "37.5":
        viewer.setVolumeRotation(90, -90, 37.5)

    outputFilename = "../../outputs/Derenzo_{}_view_{}".format(os.path.basename(os.path.dirname(os.path.dirname(filepath))), view)
    print(outputFilename)
    viewer.derenzo(outputFilename=outputFilename)

    root.mainloop()
