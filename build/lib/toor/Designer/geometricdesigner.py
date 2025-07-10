# *******************************************************
# * FILE: geometricdesigner.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

import os
import vtk
import numpy as np
# from scipy import ndimage, misc


class GeometryDesigner:
    def __init__(self, detector_geometry=False, draw_volume=False, geometry_vector=None, pixel_dimensions=[2,10,2], volume = None,top_angle=None):
        self.pixel_dimensions = pixel_dimensions
        self.geometry_vector = geometry_vector
        self.top = top_angle
        self.volume = volume

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

        if detector_geometry:
            self._draw_detectors()

        # With almost everything else ready, its time to initialize the renderer and window, as well as creating a method for exiting the application
        # renderWin = self.vtkWidget.GetRenderWindow()
        # renderInteractor = self.vtkWidget.GetRenderWindow().GetInteractor()

        # Add Renderer to PYVTK window
        # self.vtkWidget.GetRenderWindow().AddRenderer(self.ren_volume)
        # renderInteractor.SetRenderWindow(renderWin)
        # We add the volume to the renderer ...
        # self.ren_volume.AddVolume(volume)
        # self.Volume4VTK = volume
        # self.ren.AddVolume(self.Volume4VTK)

        self.ren.ResetCamera()
        # self.ren_volume.SetBackground(0, 0, 0)  # black

        # self.ren_volume.ResetCamera()

        # self.renderWin = renderWin
        self.renderInteractor.Initialize()

        # Because nothing will be rendered without any input, we order the first render manually before control is handed over to the main-loop.


    def _draw_image_reconstructed(self, map_cut=0.6):
        print('volume')
        direct = os.path.dirname(os.path.dirname(__file__))
        colormap = 'hot.cl'

        path = os.path.join(direct,"ControlUI", "colormap_files")
        z_pos = self.volume.shape[2]
        data3D = (self.volume)
        # data3D = self.volume
        #
        # data3D = ndimage.gaussian_filter(data3D, sigma=0.8)
        # data3D = ndimage.uniform_filter(data3D, size=3)
        # data3D = ndimage.convolve(data3D, np.ones((3, 3, 3)) / 27)
        # data3D = ndimage.gaussian_gradient_magnitude(data3D, sigma=0.6)
        # apply a wavelet filter to de image
        import pywt

        coeffs = pywt.dwtn(data3D, 'haar')  # 'haar' is the wavelet type, you can choose others as well

        # Threshold the wavelet coefficients (this is optional and depends on your application)
        threshold = 0.5  # Adjust this threshold as needed
        coeffs_thresh = {key: pywt.threshold(value, threshold, mode='soft') for key, value in coeffs.items()}
        # data3D = pywt.idwtn(coeffs_thresh, 'haar')
        data3D = ndimage.median_filter(data3D, size=3)
        # Reconstruct the denoised image
        from skimage import data, img_as_float
        from skimage.restoration import denoise_nl_means
        # Adjust the parameters according to your requirements
        patch_size = 3 # Size of patches used for denoising
        patch_distance = 3  # Maximum distance for patches to be considered similar
        h = 0.1  # Smoothing parameter (higher values give stronger smoothing)

        # Apply non-local means denoising
        # data3D = denoise_nl_means(data3D, patch_size=patch_size, patch_distance=patch_distance, h=h)


        # denoised_image = pywt.idwtn(coeffs_thresh, 'haar')
        xx = (np.tile(np.arange(0, data3D.shape[0]), (data3D.shape[0], 1)) - (
                data3D.shape[0] - 1) / 2) ** 2
        yy = (np.tile(np.arange(0, data3D.shape[1]), (data3D.shape[1], 1)) - (
                data3D.shape[1] - 1) / 2) ** 2
        xx = xx.T
        # circle_cut = xx + yy - (self.im_index_x.shape[1] * np.sin(np.radians(easypetdata.half_real_range))*0.5) ** 2
        circle_cut = xx + yy - (data3D.shape[1] * 0.4) ** 2
        # circle_cut = xx + yy - (self.im_index_x.shape[1] * np.sin(np.radians(120/2))*.50) ** 2

        circle_cut[circle_cut > 0] = 0
        circle_cut[circle_cut < 0] = 1
        circle_cut = np.tile(circle_cut[:, :, None], (1, 1, data3D.shape[2]))
        data3D = data3D*circle_cut
        # data3D[data3D < 0.02*data3D.max()] = 0
        data3D[:,:,:10] = 0
        data3D[:,:,-10:] = 0

        # data3D[:,:,-3:] = 0
        #  Volume inversion
        # data3D = np.flip(data3D, axis=2)

        size = data3D.shape
        w = size[0]
        h = size[1]
        d = size[2]
        stack = np.zeros((w, d, h))

        for j in range(0, z_pos):
            stack[:, j, :] = data3D[:, :, j]

        stack = np.require(stack, dtype=np.float32)  # stack = np.require(data3D,dtype=np.uint16)

        normalize_colormap = np.max(data3D)*map_cut
        start_color = 20
        end_color = 1500

        # --------IMPORT DATA-----------------

        dataImporter = vtk.vtkImageImport()
        # The preaviusly created array is converted to a string of chars and imported.
        data_string = stack.tostring()
        dataImporter.CopyImportVoidPointer(data_string, len(data_string))
        # dataImporter.SetDataScalarTypeToUnsignedShort()
        dataImporter.SetDataScalarTypeToFloat()
        # dataImporter.SetDataSpacing(self.pixel_size_reconstruct_file[0], self.pixel_size_reconstruct_file[2],
        #                             self.pixel_size_reconstruct_file[1])

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

        # file_name = '{}{}.cl'.format(path, colormap)
        file_name = os.path.join(path, colormap)
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
                increment_color += len(reader) / abs(start_color - end_color)
                if int(np.round(increment_color, 0)) < len(reader):
                    colorTransferFunction.AddRGBPoint(float(reader[row][0]) * normalize_colormap,
                                                      float(reader[int(np.round(increment_color, 0))][1]),
                                                      float(reader[int(np.round(increment_color, 0))][2]),
                                                      float(reader[int(np.round(increment_color, 0))][3]))
        # #
        # alphaChannelFunc.AddPoint(opacitymap[start_color, 0], 0.03)
        # alphaChannelFunc.AddPoint(opacitymap[end_color, 1], 0.15)
        colorTransferFunction.AddRGBPoint(0,
                                          float(reader[0][1]), float(reader[0][2]), float(reader[0][3]))

        colorTransferFunction.AddRGBPoint(data3D.max(),
                                          float(reader[-1][1]), float(reader[-1][2]), float(reader[-1][3]))


        PEAK = 1
        alphaChannelFunc.AddPoint(0, 0.0)
        # alphaChannelFunc.AddPoint(data3D.max()*0.01, 0.0)
        # alphaChannelFunc.AddPoint(data3D.max() * PEAK-0.1, 0.05)
        # alphaChannelFunc.AddPoint(data3D.max()*PEAK, 0.9)
        # alphaChannelFunc.AddPoint(data3D.max()*PEAK+0.2, 0.05)
        alphaChannelFunc.AddPoint(data3D.max(), 1)
        # alphaChannelFunc.AddPoint(0.296, 0.01)
        # # alphaChannelFunc.AddPoint(0.10*data3D.max(), 0.5)
        # # alphaChannelFunc.AddPoint(0.20*data3D.max(), 0.4)
        # # alphaChannelFunc.AddPoint(0.30*data3D.max(), 0.5)
        # # alphaChannelFunc.AddPoint(0.40*data3D.max(), 0.15)
        # # alphaChannelFunc.AddPoint(0.50*data3D.max(), 0.10)
        # # alphaChannelFunc.AddPoint(data3D.max(), 0.7)
        # alphaChannelFunc.AddPoint(1, 0.2)
        # alphaChannelFunc.AddPoint(1.1, 0.1)
        # alphaChannelFunc.AddPoint(1.127, 0.1)
        # alphaChannelFunc.AddPoint(1.19, 0.01)
        # alphaChannelFunc.AddPoint(float(reader[-1][0]) * normalize_colormap, 1)

        # The preavius two classes stored properties. Because we want to apply these properties to the volume we want to render,
        # we have to store them in a class that stores volume prpoperties.
        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(colorTransferFunction)
        volumeProperty.SetScalarOpacity(alphaChannelFunc)

        # volumeProperty.SetGradientOpacity(volumeGradientOpacity)
        volumeProperty.SetInterpolationType(2)
        volumeProperty.SetAmbient(1)
        # volumeProperty.SetDiffuse(0.1)
        volumeProperty.SetSpecular(0.3)

        # volumeProperty.ShadeOn()

        # This class describes how the volume is rendered (through ray tracing).
        # compositeFunction = vtk.vtkFixedPointVolumeRayCastCompositeFunction()
        # We can finally create our volume. We also have to specify the data for it, as well as how the data will be rendered.
        # smoothing_filter = vtk.vtkImageGaussianSmooth()
        # smoothing_filter.SetDimensionality(3)
        # smoothing_filter.SetInputConnection(dataImporter.GetOutputPort())
        # smoothing_filter.SetStandardDeviations(0.5, 0.5, 0.5)
        # smoothing_filter.SetRadiusFactors(0.25, 0.25, 0.25)
        # #dataImporter = smoothing_filter
        # volumeMapper = vtk.vtkFixedPointVolumeRayCastMapper()
        volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
        volumeMapper.UseJitteringOn()
        volumeMapper.SetSampleDistance(0.2)

        volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

        # The class vtkVolume is used to pair the preaviusly declared volume as well as the properties to be used when rendering that volume.
        volume = vtk.vtkVolume()
        volume.SetMapper(volumeMapper)
        volume.SetProperty(volumeProperty)
        # self.ren.SetUseDepthPeeling(1)

        scalarBar = vtk.vtkScalarBarActor()
        # scalarBar.SetOrientationToHorizontal()
        # scalarBar.SetPosition(0.1, 0.1)
        scalarBar.SetWidth(0.1)
        scalarBar.SetHeight(0.8)
        scalarBar.SetLookupTable(colorTransferFunction)
        # scalarBar.SetTitle("MBq/ml")
        # title vertical rotated
        scalarBar.SetOrientationToVertical()
        scalarBar.SetPosition(0.85, 0.1)
        scalarBar.SetPosition2(0.1, 0.8)

        scalarBar.SetNumberOfLabels(4)

        # self.ren.AddActor2D(scalarB
        self.ren.AddActor2D(scalarBar)

        self.ren.AddVolume(volume)
        self.renderWin.Render()
        self.renderInteractor.Start()

        # self.renderInteractor.Start()
        # volume.SetUserTransform(FOV_trans)

    def _draw_detectors(self):
        number_of_detectors = len(self.geometry_vector[0])
        # number_of_detectors = 8000
        for i in range(number_of_detectors):
            cube = vtk.vtkCubeSource()
            # cube.SetCenter([i * 2, 5, 2])
            cube.SetXLength(self.pixel_dimensions[0])
            cube.SetYLength(self.pixel_dimensions[1])
            cube.SetZLength(self.pixel_dimensions[2])

            cube.Update()
            transL2 = vtk.vtkTransform()
            # transL1.Translate(-10, 150, -10)
            transL2.RotateZ(np.degrees(self.top[i]))

            labelTransform = vtk.vtkTransformPolyDataFilter()
            labelTransform.SetTransform(transL2)
            labelTransform.SetInputConnection(cube.GetOutputPort())

            transL1 = vtk.vtkTransform()
            # transL1.Translate(-10, 150, -10)
            # transL1.RotateZ(np.degrees(self.top[i]))
            transL1.Translate(self.geometry_vector[0][i],self.geometry_vector[1][i], self.geometry_vector[2][i])
            labelTransform_2 = vtk.vtkTransformPolyDataFilter()
            labelTransform_2.SetTransform(transL1)
            labelTransform_2.SetInputConnection(labelTransform.GetOutputPort())



            # transL1.Scale(5, 5, 5)

            cm = vtk.vtkPolyDataMapper()
            cm.SetInputConnection(labelTransform_2.GetOutputPort())
            ca = vtk.vtkActor()
            ca.SetMapper(cm)
            # if i < 1920:
            #     # print('Section A1')
            #     ca.GetProperty().SetColor([1, 0, 0])
            # elif 1920 <= i < 3840:
            #     # print('Section A2')
            #     ca.GetProperty().SetColor([0, 1, 0])
            #
            # elif 3840 <= i < 7296:
            #     # print('Section B1')
            #     ca.GetProperty().SetColor([0, 0, 1])
            #
            # elif 7296 <= i < 7296+3456:
            #     # print('Section B1')
            #     ca.GetProperty().SetColor([0, 0.5, 1])
            #
            # elif 7296+3456<= i < 14912:
            #     ca.GetProperty().SetColor([0.5, 0.5, 1])
            #
            # elif 14912 <= i < 22400:
            #     ca.GetProperty().SetColor([0, 0.5, 0.5])
            #
            # elif 22400+5824 <= i <= 22400+11648:
            #     ca.GetProperty().SetColor([1, 0.5, 0.2])

            # ca.color(red)
            self.ren.AddActor(ca)


        # create a render with 3 viewport

