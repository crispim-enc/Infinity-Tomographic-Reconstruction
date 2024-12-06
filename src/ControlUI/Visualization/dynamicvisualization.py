import os
from PyQt5 import QtCore, QtWidgets
import numpy as np
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class PopulateDynamicViewWindowVTK(QtCore.QObject):
    def __init__(self, planes_distribution="Vertical"):
        super().__init__()
        self.planes_distribution = planes_distribution
        self.layout_vtk_dynamic_window = QtWidgets.QVBoxLayout()
        self.layout_vtk_dynamic_window.setContentsMargins(0, 0, 0, 0)
        self.vtkWidget_dynamic_view = QVTKRenderWindowInteractor(self.dynamic_view_frame)
        self.layout_vtk_dynamic_window.addWidget(self.vtkWidget_dynamic_view)
        self.dynamic_view_frame.setLayout(self.layout_vtk_dynamic_window)

        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        # interatorStyle = self.iren.GetInteractorStyle()
        # self.iren.GetInteractorStyle().SetCurrentStyleToTrackballCamera()
        self.vtkWidget_dynamic_view.show()
        self.iren.Initialize()
        self.vtkWidget_dynamic_view.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.vtkWidget_dynamic_view.GetRenderWindow().Render()
        # PopulateDynamicViewWindowVTK._create_dynamic_view(self, 10)

    def _create_dynamic_view(self, list_of_volumes=[]):
        number_of_views = len(list_of_volumes)
        # area = self.dockWidgetArea(self.dynamic_view_dockWidget)
        # if self.dynamic_view_dockWidget.frameSize().width()>=self.dynamic_view_dockWidget.frameSize().height():
        #     self.dynamic_view_dockWidget.setMaximumSize(QtCore.QSize(524287, 250))
        #     # self.dynamic_view_dockWidget.setMinimumSize(QtCore.QSize(524287,250 ))
        #     width_per_view = self.dynamic_view_frame.width()/number_of_views
        #     xmins = [width_per_view*i/self.dynamic_view_frame.width() for i in range(number_of_views)]
        #     xmaxs = [width_per_view*(i+1)/self.dynamic_view_frame.width() for i in range(number_of_views)]
        #     ymins = [0]*number_of_views
        #     ymaxs = [1]*number_of_views
        # else:
        self.dynamic_view_dockWidget.setMaximumSize(QtCore.QSize(524287, 524287))
        # self.dynamic_view_dockWidget.setMinimumSize(QtCore.QSize(150, 1024))~

        number_in_y_direction = number_of_views / 2
        height_per_view = self.dynamic_view_frame.height() / number_in_y_direction
        ymins = [height_per_view * i / self.dynamic_view_frame.height() for i in range(number_of_views)]
        ymaxs = [height_per_view * (i + 1) / self.dynamic_view_frame.height() for i in range(number_of_views)]
        xmins = [0] * number_of_views
        xmaxs = [1] * number_of_views
        j = 0
        for i in range(number_of_views):
            if i % 2 == 0:
                ymins[i] = height_per_view * j / self.dynamic_view_frame.height()
                ymaxs[i] = height_per_view * (j + 1) / self.dynamic_view_frame.height()
                xmins[i] = 0
                xmaxs[i] = 0.5

            else:
                ymins[i] = height_per_view * (j) / self.dynamic_view_frame.height()
                ymaxs[i] = height_per_view * (j + 1) / self.dynamic_view_frame.height()
                xmins[i] = 0.5
                xmaxs[i] = 1
                j += 1
        # Inverted order of appereance Y

        self.list_dynamic_renderers = [None] * number_of_views
        for i in range(number_of_views):
            self.list_dynamic_renderers[i] = vtk.vtkRenderer()

            # self.vtkWidget.GetRenderWindow().SetAlphaBitPlanes(1)
            # # self.vtkWidget.SetMultiSamples(0)
            # self.renderer.SetUseDepthPeeling(1)
            # self.renderer.SetMaximumNumberOfPeels(100)
            # self.renderer.SetOcclusionRatio(0.1)
            # Create sphere
            self.list_dynamic_renderers[i].SetViewport(xmins[i], ymins[number_of_views - 1 - i], xmaxs[i],
                                                       ymaxs[number_of_views - 1 - i])
            # ViewPortBorder(renderer = self.list_dynamic_renderers[i], color=[0,0,0])
            PopulateDynamicViewWindowVTK._create_time_stamp(self, i, time_stamp=i * 62)
            [dataImporter, colorTransferFunction,
             alphaChannelFunc] = PopulateDynamicViewWindowVTK._applyColorAndOpacityMap(self,
                                                                                           volume=list_of_volumes[i])
            PopulateDynamicViewWindowVTK._add_volume_to_renderer(self, dataImporter, colorTransferFunction,
                                                                 alphaChannelFunc,
                                                                 pixel_size_reconstruct_file=[1, 1, 1], i=i)
            # sphereSource = vtkSphereSource()
            # sphereSource.SetCenter(0.0, 0.0, 0.0)
            # sphereSource.SetRadius(15)
            # sphereSource.Update()
            #
            # mapper = vtk.vtkPolyDataMapper()
            # mapper.SetInputConnection(sphereSource.GetOutputPort())
            # actor = vtkActor()
            # actor.SetMapper(mapper)
            # self.actor = actor
            # self.renderer.AddActor(self.actor)
            #
            # self.renderer.ResetCamera()
            # self.setCentralWidget(self.image_3d_frame)

            self.vtkWidget_dynamic_view.GetRenderWindow().AddRenderer(self.list_dynamic_renderers[i])

    def _create_time_stamp(self, i, time_stamp=62):
        m, s = divmod(time_stamp, 60)
        h, m = divmod(m, 60)

        if time_stamp < 3600:  # Maior que uma hora
            time_stamp = "%02d m %02d s" % (m, s)
        else:
            time_stamp = "%d h %02d m %02d s" % (h, m, s)
        textActor = vtk.vtkTextActor()

        textActor.SetInput(time_stamp)
        # textActor.SetAlignmentPoint(1)
        # textActor.SetPosition2(0.5, 0.5)
        textActor.GetTextProperty().SetFontSize(20)
        textActor.GetTextProperty().BoldOn()
        # textActor.GetTextProperty().SetJustificationToCentered ()
        textActor.GetTextProperty().SetColor(1, 1, 1)
        self.list_dynamic_renderers[i].AddActor2D(textActor)

    def _applyColorAndOpacityMap(self, volume, pixel_size_reconstruct_file=[1, 1, 1]):
        colormap = self.color_map_3D_views[self.colortable_comboBox.currentIndex()]

        dir = os.path.dirname(__file__)
        path = os.path.join(dir, 'colormap_files')
        # path = dir + "/colormap_files/"
        z_pos = volume.shape[2]
        data3D = volume
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
        return dataImporter, colorTransferFunction, alphaChannelFunc

    def _add_volume_to_renderer(self, dataImporter, colorTransferFunction, alphaChannelFunc,
                                pixel_size_reconstruct_file=[1, 1, 1], i=0):
        # The preavius two classes stored properties. Because we want to apply these properties to the volume we want to render,
        # we have to store them in a class that stores volume prpoperties.
        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(colorTransferFunction)
        volumeProperty.SetScalarOpacity(alphaChannelFunc)

        # volumeProperty.SetGradientOpacity(volumeGradientOpacity)
        volumeProperty.SetInterpolationType(2)
        volumeProperty.SetAmbient(1)
        # volumeProperty.SetDiffuse(0.1)
        volumeProperty.SetSpecular(0.2)
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

        # Center volume accordingly with volume radius
        (xMin, xMax, zMin, zMax, yMin, yMax) = dataImporter.GetExecutive().GetWholeExtent(
            dataImporter.GetOutputInformation(0))
        w = xMax
        d = zMax
        h = yMax
        FOV_trans = vtk.vtkTransform()
        FOV_trans.Translate(-w * pixel_size_reconstruct_file[0] / 2, 0,
                            -h * pixel_size_reconstruct_file[1] / 2)

        # planeClipAxial = vtk.vtkPlane()
        # planeClipAxial.SetOrigin(w / 2, d / 2, h / 2)
        # planeClipAxial.SetNormal(0, -1, 0)
        # planeClipCoronal = vtk.vtkPlane()
        # planeClipCoronal.SetOrigin(w, d, h)
        # planeClipCoronal.SetNormal(0.0, 0, -1)
        # planeClipSagittal = vtk.vtkPlane()
        # planeClipSagittal.SetOrigin(w, d, h)
        # planeClipSagittal.SetNormal(-1, 0, 0)
        # self.planeClipAxial = planeClipAxial
        # self.planeClipCoronal = planeClipCoronal
        # self.planeClipSagittal = planeClipSagittal

        # volumeMapper.AddClippingPlane(planeClipAxial)
        # volumeMapper.AddClippingPlane(planeClipCoronal)
        # volumeMapper.AddClippingPlane(planeClipSagittal)

        # The class vtkVolume is used to pair the preaviusly declared volume as well as the properties to be used when rendering that volume.
        volume = vtk.vtkVolume()
        volume.SetMapper(volumeMapper)
        volume.SetProperty(volumeProperty)
        volume.SetUserTransform(FOV_trans)

        # With almost everything else ready, its time to initialize the renderer and window, as well as creating a method for exiting the application
        renderWin = self.vtkWidget_dynamic_view.GetRenderWindow()

        # interactor_style = vtk.vtkInteractorStyleImage()
        renderInteractor = self.vtkWidget_dynamic_view.GetRenderWindow().GetInteractor()
        # interactor_style.SetInteractionModeToImage3D()
        # self.vtkWidget_dynamic_view.GetRenderWindow().GetInteractor().SetInteractorStyle(interactor_style)
        renderInteractor.SetRenderWindow(renderWin)
        # We add the volume to the renderer ...
        self.list_dynamic_renderers[i].AddVolume(volume)
        # self.Volume4VTK = volume
        # self.list_dynamic_renderers[i].SetBackground(0, 0, 0)  # black

        self.list_dynamic_renderers[i].ResetCamera()
        renderInteractor.Initialize()
        # # Because nothing will be rendered without any input, we order the first render manually before control is handed over to the main-loop.
        renderWin.Render()
