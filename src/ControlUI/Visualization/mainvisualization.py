import os
from PyQt5 import QtCore, QtWidgets

import numpy as np
from scipy import ndimage
import math
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from .mouseobserver_main import MouseObserverMainWindow
from .typeofviews import ViewPortBorder, ViewPortSelected


class PopulateMainWindowVTK(object):
    def __init__(self, parent=None):
        self.parent = parent
        self.layout_vtk_main_window = QtWidgets.QGridLayout()
        self.layout_vtk_main_window.setContentsMargins(0, 0, 0, 0)
        self.vtkWidget = QVTKRenderWindowInteractor(self.parent.main_frame)
        self.layout_vtk_main_window.addWidget(self.vtkWidget, 0, 0, 100, 100)
        self.layout_vtk_main_window.setHorizontalSpacing(0)
        self.layout_vtk_main_window.setVerticalSpacing(0)
        self._initializeQtButtonsOverMainWindow()

        self.interactorMainWindow = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.interactorStyle3D = vtk.vtkInteractorStyleTrackballCamera()
        self.interactorStyle3D.AddObserver("MouseMoveEvent",
                                           lambda obj, event: self.parent.mouseObserverMainWindow.MouseMoveCallback(
                                                obj, event))
        # self.interactorMainWindow.SetInteractorStyle(self.interactorStyle3D)
        self.interactorStyleImage = vtk.vtkInteractorStyleImage()
        self.interactorStyleImage.SetInteractionModeToImageSlicing()

        # interactorStyleImage = vtk.vtkInteractorStyleRubberBand3D()
        # self.interactorStyleImage = vtk.vtkInteractorStyleRubberBand2D()
        # interactorStyleImage = vtk.vtkInteractorStyleAreaSelectHover()
        # self.interactorStyleImage = vtk.vtkInteractorStyleDrawPolygon()
        # interactorStyleImage = vtk.vtkInteractorStyleRubberBand3D()
        # interactorStyleImage = vtk.vtkInteractorStyleSwitch()
        # self.vtkWidget.GetRenderWindow().GetInteractor().SetInteractorStyle(interactorStyleImage)
        self.interactorMainWindow.SetInteractorStyle(self.interactorStyleImage)
        self.interactorStyleImage.AddObserver("MouseMoveEvent",
                                              lambda obj, event: self.parent.mouseObserverMainWindow.MouseMoveCallback(obj,
                                                                                                           event))
        self.interactorStyleImage.AddObserver("RightButtonPressEvent",
                                              lambda obj, event: self.parent.mouseObserverMainWindow.ButtonCallback(obj,
                                                                                                        event))
        self.interactorStyleImage.AddObserver("RightButtonReleaseEvent",
                                              lambda obj, event: self.parent.mouseObserverMainWindow.ButtonCallback(obj,
                                                                                                        event))
        self.interactorStyleImage.AddObserver("MouseWheelForwardEvent",
                                              lambda obj, event: self.parent.mouseObserverMainWindow.wheelForwardCallback(
                                                                                                              obj,
                                                                                                              event))
        self.interactorStyleImage.AddObserver("MouseWheelBackwardEvent",
                                              lambda obj, event: self.parent.mouseObserverMainWindow.wheelForwardCallback(
                                                                                                              obj,
                                                                                                              event))
        self.vtkWidget.show()
        self.interactorMainWindow.Initialize()
        self.vtkWidget.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.vtkWidget.GetRenderWindow().Render()

        self.resliced_main_window = [None] * 3
        self.type_view_selected = {"type": "Scanning View",
                                   "orientation": "horizontal",
                                   "active_areas": {},
                                   "number_of_views": 0,
                                   "number_of_active_renderers": 0,
                                   "xmins": [],
                                   "xmaxs": [],
                                   "ymins": [],
                                   "ymaxs": []
                                   }
        if self.parent.actionmain_window3d_and_3planes_vertical.isChecked():
            self.type_view_selected["type"] = "Scanning View"
            self.type_view_selected["orientation"] = "horizontal"

        elif self.parent.actionmain_window3d_and_3planes_horizontal.isChecked():
            self.type_view_selected["type"] = "Scanning View"
            self.type_view_selected["orientation"] = "horizontal"

        elif self.parent.actionall_planes_view.isChecked():
            self.type_view_selected["type"] = "Cutted volume view"
            self.type_view_selected["orientation"] = "horizontal"

        if self.parent.action_one_view.isChecked():
            self.type_view_selected["number_of_views"] = 1
        elif self.parent.action_two_views.isChecked():
            self.type_view_selected["number_of_views"] = 2
        elif self.parent.action_three_views.isChecked():
            self.type_view_selected["number_of_views"] = 3

        self.type_view_selected["active_areas"] = {"area_3D": self.parent.action3D_viewPort.isChecked(),
                                                   "axial": self.parent.action_mice_axial_viewport.isChecked(),
                                                   "coronal": self.parent.action_mice_coronal_viewport.isChecked(),
                                                   "sagittal": self.parent.action_mice_sagittal_viewport.isChecked(),
                                                   }
        self.type_view_selected["number_of_active_renderers"] = 0
        active_areas = 0
        for area in self.type_view_selected["active_areas"]:
            if area:
                active_areas += 1

        if self.type_view_selected["active_areas"]["area_3D"]:
            self.type_view_selected["number_of_active_renderers"] = (active_areas - 1) * self.type_view_selected[
                "number_of_views"] + 1

        if self.type_view_selected["type"] == 'Scanning View':
            if self.type_view_selected["orientation"] == 'vertical':
                if self.parent.action3D_viewPort.isChecked:
                    percentage_of_3D_occupancy = [0.5, 1]
                    xmins = [percentage_of_3D_occupancy[0]] * self.type_view_selected["number_of_active_renderers"]
                    xmins[0] = 0
                    xmaxs = [1] * self.type_view_selected["number_of_active_renderers"]
                    xmaxs[0] = percentage_of_3D_occupancy[0]
                    ymins = [0] * self.type_view_selected["number_of_active_renderers"]
                    ymaxs = [1] * self.type_view_selected["number_of_active_renderers"]
                    x_m = 0
                    y_m = 0
                    for k in range(1, self.type_view_selected["number_of_active_renderers"]):
                        xmins[k] = percentage_of_3D_occupancy[0] + (x_m - percentage_of_3D_occupancy[0]) / \
                                   self.type_view_selected["number_of_views"]
                        xmaxs[k] = percentage_of_3D_occupancy[0] + (x_m - percentage_of_3D_occupancy[0] + 1) / \
                                   self.type_view_selected["number_of_views"]
                        ymins[k] = y_m / (active_areas - 1)
                        ymaxs[k] = (y_m + 1) / (active_areas - 1)

                        y_m += 1
                        if k == active_areas:
                            y_m = 0
                            x_m += 1

                    self.type_view_selected["xmins"] = xmins
                    self.type_view_selected["xmaxs"] = xmaxs
                    self.type_view_selected["ymins"] = ymins
                    self.type_view_selected["ymaxs"] = ymaxs

            elif self.type_view_selected["orientation"] == 'horizontal':
                begin_y = 2 / 5
                ymins = [begin_y, 0, 0, 0, begin_y, 3 / 5]
                ymaxs = [1, begin_y, begin_y, begin_y, 3 / 5, 1]
                xmins = [0, 0, 1 / 3, 2 / 3, 2 / 3, 2 / 3]
                xmaxs = [2 / 3, 1 / 3, 2 / 3, 1, 1, 1]
                self.type_view_selected["xmins"] = xmins
                self.type_view_selected["xmaxs"] = xmaxs
                self.type_view_selected["ymins"] = ymins
                self.type_view_selected["ymaxs"] = ymaxs

        elif self.type_view_selected["type"] == 'Cutted volume view':
            print('Cutted_volumw_view')

        border_colors = [[0.5, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1]]
        self.listOfRenderers =  [vtk.vtkRenderer() for _ in range(len(self.type_view_selected["xmins"]))]
        i = 0
        # self.list_populated_windows = [lambda i: self._initialize_3D_window(self, i),
        #                                lambda i: self._initialize_cuts_volume_port_view(self, i,
        #                                                                                 'axial'),
        #                                lambda i: self._initialize_cuts_volume_port_view(self, i,
        #                                                                                 'coronal'),
        #                                lambda i: self._initialize_cuts_volume_port_view(self, i,
        #                                                                                 'sagital'),
        #                                lambda i: self._initialize_easypetbed_camera(self, i),
        #                                lambda i: self._initialize_easypetbed_camera(self, i)]

        for i in range(len(self.type_view_selected["xmins"])):
            # if self.info_type_of_views[self.type_view_selected["type"]]["renderer_area_3D"] is None:
            #     self.info_type_of_views[self.type_view_selected["type"]]["renderer_area_3D"] = vtk.vtkRenderer()
            # self.listOfRenderers[i] = vtk.vtkRenderer()

            self.listOfRenderers[i].SetViewport(self.type_view_selected["xmins"][i],
                                                self.type_view_selected["ymins"][i],
                                                self.type_view_selected["xmaxs"][i],
                                                self.type_view_selected["ymaxs"][i])

            v=ViewPortBorder(renderer=self.listOfRenderers[i], color=border_colors[i],
                           points_coordinates=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], last=False)
            v.updateViewPortBorder()

            self.listOfRenderers[i].ResetCamera()
            self.vtkWidget.GetRenderWindow().AddRenderer(self.listOfRenderers[i])

        # PopulateMainWindowVTK._initialize_plot_vital_signals(self,2)

        [self.axes_widget, self.axes_3D] = PopulateMainWindowVTK.design_axes(self)
        # PopulateMainWindowVTK._changeto_slicedviews(self)
        PopulateMainWindowVTK.camera_initial_position(self)

        number_max_acceptted_volumes = 20
        my_list = list(range(number_max_acceptted_volumes))
        self.volumesInMain = {item: {"Scanning_name": None,
                                     "PET": None,
                                     "Attenuation_volume": None,
                                     "Detector_sensibility": None,

                                     } for item in my_list}
        self.actorInMain = {item: {"Scanning_name": None,
                                   "PET_axial": None,
                                   "PET_coronal": None,
                                   "PET_sagittal": None,

                                   } for item in my_list}
        # self.listOfRenderers[0].AddActor(self.axes_3D)
        # self.vtkWidget.GetRenderWindow().GetInteractor().SetRenderWindow(self.vtkWidget.GetRenderWindow())
        # self.vtkWidget.GetRenderWindow().Render()

    def _initializeQtButtonsOverMainWindow(self):
        self.gridlayout_camerabuttons = QtWidgets.QGridLayout()
        self.gridlayout_camerabuttons.addWidget(self.parent.changecamera2bottom_toolbutton, 2, 1, 1, 1,
                                                QtCore.Qt.AlignHCenter)
        self.gridlayout_camerabuttons.addWidget(self.parent.changecamera2right_toolbutton, 1, 2, 1, 1)
        self.gridlayout_camerabuttons.addWidget(self.parent.changecamera2left_toolbutton, 1, 0, 1, 1)
        self.gridlayout_camerabuttons.addWidget(self.parent.changecamera2top_toolbutton, 0, 1, 1, 1,
                                                QtCore.Qt.AlignHCenter)
        self.gridlayout_camerabuttons.addWidget(self.parent.resetcamera_toolButton, 2, 0, 1, 1)
        self.gridlayout_cameramiddlebuttons = QtWidgets.QGridLayout()
        self.gridlayout_cameramiddlebuttons.addWidget(self.parent.changecamera2back_toolbutton, 0, 2, 1, 1)
        self.gridlayout_cameramiddlebuttons.addWidget(self.parent.changecamera2front_toolbutton, 0, 1, 1, 1)
        self.gridlayout_camerabuttons.addLayout(self.gridlayout_cameramiddlebuttons, 1, 1, 1, 1)
        self.layout_vtk_main_window.addLayout(self.gridlayout_camerabuttons, 1, 1, 4, 4)

        self.gridlayout_bedbuttons = QtWidgets.QGridLayout()
        self.gridlayout_bedbuttons.addWidget(self.parent.movebed_to_begin_toolbutton, 0, 0, 1, 1)
        self.gridlayout_bedbuttons.addWidget(self.parent.movebed_backward_toolbutton, 0, 1, 1, 1)
        self.gridlayout_bedbuttons.addWidget(self.parent.movebed_foward_toolbutton, 0, 2, 1, 1)
        self.gridlayout_bedbuttons.addWidget(self.parent.movebed_to_end_toolbutton, 0, 3, 1, 1)
        self.gridlayout_bedbuttons.addWidget(self.parent.emergency_stop_bed_tool_button, 0, 4, 1, 1)
        self.layout_vtk_main_window.addLayout(self.gridlayout_bedbuttons, 50, 50, 1, 5)

        self.layout_vtk_main_window.addWidget(self.parent.turn_on_camera_toolbutton, 50, 98, 1, 1)
        self.parent.main_frame.setLayout(self.layout_vtk_main_window)

    def _initialize_3D_window(self, number_renderer):
        x_comum = -(356.44 - 347 + 347 / 2)
        y_comum = -(55.33 + 347 / 2)
        z_comum = -(370 / 2 + 51.07 + 165)  # (center_z+offset_solid+realdistance
        # [self.axes_widget, self.axes_3D] = self.design_axes(self, renderInteractor)
        # [self.scalar_bar_widget, self.scalar_bar_actor] = self.design_scalar_bar(self,
        #                                                                                      colorTransferFunction,
        #                                                                                      renderInteractor)
        self.bed_inside_part_actor = PopulateMainWindowVTK.design_stl_file(self, 'bed_inside',
                                                                           translate_x=x_comum + 160,
                                                                           translate_y=y_comum + 205,
                                                                           translate_z=z_comum + 240,
                                                                           rotate_x=90, rotate_y=0, rotate_z=0,
                                                                           color_actor=[0.4, 0.4, 0.4],
                                                                           opacity_actor=1)
        self.easypet_box_device_actor = PopulateMainWindowVTK.design_stl_file(self, 'Assem - frente_box-1',
                                                                              translate_x=x_comum,
                                                                              translate_y=y_comum,
                                                                              translate_z=z_comum,
                                                                              rotate_x=90, rotate_y=0, rotate_z=0,
                                                                              color_actor=[0.9, 0.9, 0.9],
                                                                              opacity_actor=0.5)
        self.connection2wall_device_actor = PopulateMainWindowVTK.design_stl_file(self, 'Assem - connection2wall-1',
                                                                                  translate_x=x_comum,
                                                                                  translate_y=y_comum,
                                                                                  translate_z=z_comum,
                                                                                  rotate_x=90, rotate_y=0,
                                                                                  rotate_z=0,
                                                                                  color_actor=[0.95, 0.9, 0.95],
                                                                                  opacity_actor=0.6)

        # self.foots_stl = [None] * 4
        # file_name_foot = ['Assem - foot_easypet-1', 'Assem - foot_easypet-2', 'Assem - foot_easypet-3',
        #                   'Assem - foot_easypet-4']
        # self.phantom_parts = [None] * 3
        # phantom_parts_name = ['Phantom_final - PhantomNEMA-1', 'Phantom_final - tampaNEMA_Top-1',
        #                       'Phantom_final - tampaNEMA-1']
        # for i in range(0, len(self.foots_stl)):
        #     self.foots_stl[i] = PopulateMainWindowVTK.design_stl_file(self, file_name_foot[i], translate_x=x_comum,
        #                                                          translate_y=y_comum, translate_z=z_comum,
        #                                                          rotate_x=90, rotate_y=0, rotate_z=0,
        #                                                          color_actor=[0.4, 0.4, 0.4], opacity_actor=0.5)
        #     if i < 3:
        #         self.phantom_parts[i] = PopulateMainWindowVTK.design_stl_file(self, phantom_parts_name[i],
        #                                                                  translate_x=x_comum + 165,
        #                                                                  translate_y=y_comum + 214,
        #                                                                  translate_z=z_comum + 370 / 2 + 51.70 + 165 - 85,
        #                                                                  rotate_x=90, rotate_y=0, rotate_z=0,
        #                                                                  color_actor=[0.95, 0.9, 0.95],
        #                                                                  opacity_actor=0.6)
        #         self.listOfRenderers[i].AddActor(self.phantom_parts[i])
        #     self.listOfRenderers[i].AddActor(self.foots_stl[i])

        self.listOfRenderers[number_renderer].AddActor(self.bed_inside_part_actor)
        self.listOfRenderers[number_renderer].AddActor(self.easypet_box_device_actor)
        self.listOfRenderers[number_renderer].AddActor(self.connection2wall_device_actor)

    def _initialize_plot_vital_signals(self, number_renderer):
        view = vtk.vtkContextView()

        # view.SetRenderWindow(self.vtkWidget.GetRenderWindow())
        # view.SetInteractor(self.vtkWidget.GetRenderWindow().GetInteractor())
        # self.vtkWidget.SetRenderWindow(view.GetRenderWindow())

        #

        self.listOfRenderers[number_renderer] = view.GetRenderer()

        # self.vtkWidget.GetRenderWindow().AddRenderer(view.GetRenderer())
        # .SetBackground(1.0, 1.0, 1.0)

        # view.GetRenderWindow().SetSize(400, 300)
        # view.SetInteractor(self.vtkWidget.GetRenderWindow().GetInteractor())
        chart = vtk.vtkChartXY()

        # self.listOfRenderers[1].Add(view)
        # vtk.vtkRenderer().AddActor(chart)
        # view=self.vtkWidget.GetRenderWindow().GetGenericContext()
        # view.GAddItem(chart)
        view.GetScene().AddItem(chart)
        chart.SetShowLegend(True)

        table = vtk.vtkTable()

        arrX = vtk.vtkFloatArray()
        arrX.SetName('X Axis')

        arrC = vtk.vtkFloatArray()
        arrC.SetName('Cosine')

        arrS = vtk.vtkFloatArray()
        arrS.SetName('Sine')

        arrT = vtk.vtkFloatArray()
        arrT.SetName('Sine-Cosine')

        table.AddColumn(arrC)
        table.AddColumn(arrS)
        table.AddColumn(arrX)
        table.AddColumn(arrT)

        numPoints = 40

        inc = 7.5 / (numPoints - 1)
        table.SetNumberOfRows(numPoints)
        for i in range(numPoints):
            table.SetValue(i, 0, i * inc)
            table.SetValue(i, 1, math.cos(i * inc))
            table.SetValue(i, 2, math.sin(i * inc))
            table.SetValue(i, 3, math.sin(i * inc) - math.cos(i * inc))

        points = chart.AddPlot(vtk.vtkChart.POINTS)
        points.SetInputData(table, 0, 1)
        points.SetColor(0, 0, 0, 255)
        points.SetWidth(1.0)
        points.SetMarkerStyle(vtk.vtkPlotPoints.CROSS)

        points = chart.AddPlot(vtk.vtkChart.POINTS)
        points.SetInputData(table, 0, 2)
        points.SetColor(0, 0, 0, 255)
        points.SetWidth(1.0)
        points.SetMarkerStyle(vtk.vtkPlotPoints.PLUS)

        points = chart.AddPlot(vtk.vtkChart.POINTS)
        points.SetInputData(table, 0, 3)
        points.SetColor(0, 0, 255, 255)
        points.SetWidth(1.0)
        points.SetMarkerStyle(vtk.vtkPlotPoints.CIRCLE)
        # view.GetInteractor().Initialize()
        # view.GetInteractor().Start()
        #
        # tt=view.GetRenderWindow()
        # self.vtkWidget.setRenderWindow(tt)
        # self.vtkWidget.GetRenderWindow().AddRenderer(view.GetRenderer())
        # self.vtkWidget.GetRenderWindow().Render()
        #

    def addNewDataToCurrentView(self, dicomReader=None):
        # Defining colormap and opacity distribution
        # volume = ndimage.median_filter(volume, 5)

        [colorTransferFunction, alphaChannelFunc] = self._applyColorAndOpacityMap(dicomReader=dicomReader)
        # put conditions to the type of the current view

        # maximum = int(np.max(data3D))

        self.addNewVolumeVtkWindow(colorTransferFunction, alphaChannelFunc, dicomReader)

        self.addNewCutPlanesVtkWindow(cutted_plane="axial",
                                      colorTransferFunction=colorTransferFunction,
                                      dicomReader=dicomReader)
        self.addNewCutPlanesVtkWindow(cutted_plane="coronal",
                                      colorTransferFunction=colorTransferFunction,
                                      dicomReader=dicomReader)
        self.addNewCutPlanesVtkWindow(cutted_plane="sagittal",
                                      colorTransferFunction=colorTransferFunction,
                                      dicomReader=dicomReader)
        renderWin = self.vtkWidget.GetRenderWindow()
        renderInteractor = self.vtkWidget.GetRenderWindow().GetInteractor()
        renderInteractor.Initialize()
        # # Because nothing will be rendered without any input, we order the first render manually before control is handed over to the main-loop.
        renderWin.Render()

    def _removeDataToCurrentView(self, volume=False):
        # Defining colormap a   nd opacity distribution
        self.remove_volume_pet_vtk_window()
        self.remove_data_cut_planes_vtk_window()

    def _applyColorAndOpacityMap(self, dicomReader=None):
        print('LOADING COLOR AND OPACITY MAP')
        colormap = self.parent.color_map_3D_views[self.parent.colortable_comboBox.currentIndex()]

        directory = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(directory, 'colormap_files')

        volume = vtk.vtkImageData()
        volume.ShallowCopy(dicomReader.GetOutput())
        # path = dir + "/colormap_files/"
        # z_pos = volume.shape[2]
        # data3D = volume
        # #  Volume inversion
        # data3D = np.flip(data3D, axis=2)
        # # try:
        # #     self.ren_volume.RemoveVolume(self.Volume4VTK)
        # # except AttributeError:
        # #     print('no volume yet')
        # # if init is False:
        # #     self.ren_volume.RemoveActor(self.axial_line)
        # #     self.ren_volume.RemoveActor(self.coronal_line)
        # #     self.ren_volume.RemoveActor(self.sagittal_line)
        # #     self.ren_volume.RemoveActor(self.sagittal)
        # #     self.ren_volume.RemoveActor(self.coronal)
        # #     self.ren_volume.RemoveActor(self.axial)
        #
        # size = data3D.shape
        # w = size[0]
        # h = size[1]
        # d = size[2]
        # image_shape=size
        # stack = np.zeros((w, d, h))
        #
        # for j in range(0, z_pos):
        #     stack[:, j, :] = data3D[:, :, j]
        #
        # stack = np.require(stack, dtype=np.float32)  # stack = np.require(data3D,dtype=np.uint16)
        # normalize_colormap = np.max(volume)
        # # start_color = int(abs(self.rs2.max() - self.rs2.end()) * 10.24)
        # # end_color = int(abs(self.rs2.max() - self.rs2.start()) * 10.24)
        #
        start_color = int(abs(0) * 10.24)
        end_color = int(abs(100) * 10.24)
        #
        # # --------IMPORT DATA-----------------
        #
        # dataImporter = vtk.vtkImageImport()
        # # The preaviusly created array is converted to a string of chars and imported.
        # data_string = stack.tostring()
        # dataImporter.CopyImportVoidPointer(data_string, len(data_string))
        # # dataImporter.SetDataScalarTypeToUnsignedShort()
        # dataImporter.SetDataScalarTypeToFloat()
        # dataImporter.SetDataSpacing(pixel_size_reconstruct_file[0], pixel_size_reconstruct_file[2],
        #                             pixel_size_reconstruct_file[1])
        # print('pixel_size{}'.format(dataImporter.GetDataSpacing()))
        # # Because the data that is imported only contains an intensity value (it isnt RGB-coded or someting similar), the importer
        # # must be told this is the case.
        # dataImporter.SetNumberOfScalarComponents(1)
        # # ---------------STORE DATA-------------------------------
        # dataImporter.SetDataExtent(0, w - 1, 0, d - 1, 0, h - 1)
        # dataImporter.SetWholeExtent(0, w - 1, 0, d - 1, 0, h - 1)
        # # dataImporter.SetTransform(transL1)
        #
        # # -----------------------Scalar range-------------
        # dataImporter.Update()

        # -----------------------------------------------------------
        # The following class is used to store transparencyv-values for later retrival. In our case, we want the value 0 to be
        # completly opaque whereas the three different cubes are given different transperancy-values to show how it works.
        alphaChannelFunc = vtk.vtkPiecewiseFunction()
        colorTransferFunction = vtk.vtkColorTransferFunction()
        colorTransferFunction.SetColorSpaceToRGB()

        # ----------------OPACITYMAP-----------------------------

        # http://www.kennethmoreland.com/color-advice/
        scalarRange = volume.GetScalarRange()
        normalize_colormap = scalarRange[1]

        file_name = os.path.join(path, '{}.cl'.format(colormap))
        reader = np.loadtxt(file_name)
        # bins, res = np.histogram(stack.ravel(), len(reader), (stack.min(), stack.max()))
        # res2 = np.interp(res, [stack.min(), stack.max()], [0, 1])
        # opacitymap = np.vstack((res, res2)).T
        # opacitymap = opacitymap.astype('float32')

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

        alphaChannelFunc.AddPoint(scalarRange[0], 0)
        alphaChannelFunc.AddPoint(scalarRange[1], 1)
        print('END LOADING COLOR AND OPACITY MAP')
        return colorTransferFunction, alphaChannelFunc

    def remove_volume_pet_vtk_window(self):
        volume = self.volumesInMain[0]['PET']
        self.listOfRenderers[0].RemoveVolume(volume)

    def remove_data_cut_planes_vtk_window(self):
        actor_axial = self.actorInMain[0]['PET_axial']
        actor_coronal = self.actorInMain[0]['PET_coronal']
        actor_sagittal = self.actorInMain[0]['PET_sagittal']
        self.listOfRenderers[1].RemoveActor(actor_axial)
        self.listOfRenderers[2].RemoveActor(actor_coronal)
        self.listOfRenderers[3].RemoveActor(actor_sagittal)

    def addNewVolumeVtkWindow(self, colorTransferFunction, alphaChannelFunc, vtkImageReader,
                              pixel_size_reconstruct_file=[0.5, 0.5, 0.5]):
        # dataImporter.pixel
        print("NEW Volume")
        # dataImporter.GetScalarRange()
        # pixel_size_reconstruct_file =dataImporter.GetPixelSpacing()
        # w = image_shape[0]
        # h = image_shape[1]
        # d = image_shape[2]
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

        volumeMapper.SetInputConnection(vtkImageReader.GetOutputPort())

        # Center volume accordingly with volume radius
        FOV_trans = vtk.vtkTransform()
        # FOV_trans.Translate(-w * pixel_size_reconstruct_file[0] / 2, 0,
        #                     -h * pixel_size_reconstruct_file[1] / 2)
        FOV_trans.Translate(-vtkImageReader.GetWidth() / 2, 0,
                            -vtkImageReader.GetHeight() / 2)

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
        renderWin = self.vtkWidget.GetRenderWindow()
        renderInteractor = self.vtkWidget.GetRenderWindow().GetInteractor()

        # Add Renderer to PYVTK window
        # self.vtkWidget.GetRenderWindow().AddRenderer(self.listOfRenderers[2])
        renderInteractor.SetRenderWindow(renderWin)
        # We add the volume to the renderer ...
        self.listOfRenderers[0].AddVolume(volume)
        self.volumesInMain[0]['Scanning_name'] = self.parent.selected_study
        self.volumesInMain[0]['PET'] = volume
        # self.Volume4VTK = volume
        self.listOfRenderers[0].SetBackground(0, 0, 0)  # black

        self.listOfRenderers[0].ResetCamera()
        [self.scalar_bar_widget, self.scalar_bar_actor] = PopulateMainWindowVTK.design_scalar_bar(self,
                                                                                                  colorTransferFunction,
                                                                                                  renderInteractor)

        # # Create the planes
        #
        # display_axial = [[0, w], [int(d / 2), int(d / 2)], [0, h]]
        # [self.axial, self.axialplaneColors] = image_3D_handler.design_planes(self, dataImporter, colorTransferFunction,
        #                                                                      display_axial, FOV_trans)
        #
        # display_coronal = [[0, w], [0, d], [int(h / 2), int(h / 2)]]
        # [self.coronal, self.coronalplaneColors] = image_3D_handler.design_planes(self, dataImporter,
        #                                                                          colorTransferFunction, display_coronal,
        #                                                                          FOV_trans)
        #
        # display_sagittal = [[int(w / 2), int(w / 2)], [0, d], [0, h]]
        # [self.sagittal, self.sagittalplaneColors] = image_3D_handler.design_planes(self, dataImporter,
        #                                                                            colorTransferFunction,
        #                                                                            display_sagittal, FOV_trans)
        #
        # # Design borders of the planes
        # w = w * self.pixel_size_reconstruct_file[0]
        # h = h * self.pixel_size_reconstruct_file[1]
        # d = d * self.pixel_size_reconstruct_file[2]
        #
        # axial_coordinates = [[-w / 2, 0, -h / 2], [-w / 2, 0, h / 2], [w / 2, 0, h / 2],
        #                      [w / 2, 0, -h / 2 - 1]]
        # axial_color = [1, 0, 0]
        #
        # coronal_coordinates = [[-w / 2, 0, h / 2], [w / 2, 0, h / 2], [w / 2, d, h / 2],
        #                        [-w / 2, d, h / 2]]
        #
        # coronal_color = [0, 1, 0]
        #
        # sagittal_coordinates = [[0, 0, -h / 2], [0, d, -h / 2], [0, d, h / 2],
        #                         [0, 0, h / 2]]
        # sagittal_color = [1, 1, 0]
        # self.axial_line = image_3D_handler.design_plane_lines(self, axial_color, len(axial_coordinates),
        #                                                       axial_coordinates)
        # self.coronal_line = image_3D_handler.design_plane_lines(self, coronal_color, len(axial_coordinates),
        #                                                         coronal_coordinates)
        # self.sagittal_line = image_3D_handler.design_plane_lines(self, sagittal_color, len(sagittal_coordinates),
        #                                                          sagittal_coordinates)
        #
        # # Add actors to render window
        #
        # self.ren_volume.AddActor(self.axial_line)
        # self.ren_volume.AddActor(self.coronal_line)
        # self.ren_volume.AddActor(self.sagittal_line)
        # self.ren_volume.AddActor(self.sagittal)
        # self.ren_volume.AddActor(self.coronal)
        # self.ren_volume.AddActor(self.axial)
        #
        #
        # self.Volume4VTK.VisibilityOn()
        # self.sagittal.VisibilityOn()
        # self.coronal.VisibilityOn()
        # self.axial.VisibilityOn()
        # self.scalar_bar_actor.SetLookupTable(colorTransferFunction)
        #
        # # self.ren_volume.GetActiveCamera().Dolly(10)
        #
        # list_to_hide = [self.actionShowVolume, self.actionShowPlanes, self.actionShowAxial,
        #                 self.actionShowCoronal, self.actionShowSagittal, self.actionShowOrientationWidget,
        #                 self.actionShowLines, self.actionShowScalarBar, self.actionShow_beds_inside,
        #                 ]
        #
        # hide_show_element = [[self.Volume4VTK], [self.axial, self.coronal, self.sagittal], [self.axial], [self.coronal],
        #                      [self.sagittal], [self.axes_3D], [self.axial_line, self.coronal_line, self.sagittal_line],
        #                      [self.scalar_bar_actor], [self.bed_inside_part_actor]]
        # for action in range(len(list_to_hide)):
        #     if not list_to_hide[action].isChecked():
        #         for i in range(len(hide_show_element[action])):
        #             hide_show_element[action][i].VisibilityOff()
        #
        # self.renderWin = renderWin
        # PopulateMainWindowVTK.camera_initial_position(self)

        # renderInteractor.Initialize()
        # # Because nothing will be rendered without any input, we order the first render manually before control is handed over to the main-loop.
        # renderWin.Render()

    def addNewCutPlanesVtkWindow(self, cutted_plane="axial", colorTransferFunction=None,
                                 dicomReader=None):
        # if colorTransferFunction or dataImporter is None:
        #     return
        print("NEW cut")
        # (xMin, xMax, zMin, zMax, yMin, yMax) = dataImporter.GetExecutive().GetWholeExtent(
        #     dataImporter.GetOutputInformation(0))
        # (xSpacing, ySpacing, zSpacing) = dataImporter.GetOutput().GetSpacing()
        # (x0, y0, z0) = dataImporter.GetOutput().GetOrigin()
        # volume = vtk.vtkImageData()
        # volume.ShallowCopy(dicomReader.GetOutput())

        (xMin, xMax, zMin, zMax, yMin, yMax) = dicomReader.GetOutput().GetExtent()
        (xSpacing, ySpacing, zSpacing) = dicomReader.GetOutput().GetSpacing()
        (x0, y0, z0) = dicomReader.GetOutput().GetOrigin()

        center = [x0 + xSpacing * 0.5 * (xMin + xMax),
                  y0 + ySpacing * 0.5 * (yMin + yMax),
                  z0 + zSpacing * 0.5 * (zMin + zMax)]
        # sliceSpacing = dataImporter.GetOutput().GetSpacing()[2]

        # center = center_input
        # if view = "3_views":

        if cutted_plane == "axial":
            index = 0
            transform = vtk.vtkMatrix4x4()
            transform.DeepCopy((1, 0, 0, center[0],
                                0, 0, 1, center[1],
                                0, 1, 0, center[2],
                                0, 0, 0, 1))

        if cutted_plane == "coronal":
            index = 1
            transform = vtk.vtkMatrix4x4()
            transform.DeepCopy((1, 0, 0, center[0],
                                0, 1, 0, center[1],
                                0, 0, 1, center[2],
                                0, 0, 0, 1))

        if cutted_plane == "sagittal":
            index = 2
            transform = vtk.vtkMatrix4x4()
            transform.DeepCopy((0, 0, 1, center[0],
                                1, 0, 0, center[1],
                                0, 1, 0, center[2],
                                0, 0, 0, 1))

        # for i in range(3):
        median = vtk.vtkImageMedian3D()
        median.SetInputConnection(dicomReader.GetOutputPort())
        median.SetKernelSize(7, 7, 1)

        dicomReader = median
        self.resliced_main_window[index] = vtk.vtkImageReslice()
        self.resliced_main_window[index].SetInputConnection(median.GetOutputPort())
        self.resliced_main_window[index].SetOutputDimensionality(2)
        self.resliced_main_window[index].SetResliceAxes(transform)
        # reslice.AutoCropOutputOn ()
        # reslice.WrapOn()
        # reslice.SetOutputExtent(xMin,xMax,zMin,zMax,yMin,yMax)
        # print(self.listOfRenderers[i].GetDisplayPoint())
        # reslice.SetInterpolationModeToLinear()
        self.resliced_main_window[index].Update()
        # matrix = reslice.GetResliceAxes()
        # new_center = matrix.MultiplyPoint((0, 0,sliceSpacing * (i), 1))
        # print(sliceSpacing * (i))
        # print(center)
        # matrix.SetElement(0, 3, new_center[0])
        # matrix.SetElement(1, 3, new_center[1])
        # matrix.SetElement(2, 3, new_center[2])

        # table = vtk.vtkLookupTable()
        # table.SetRange(0, maximum)  # image intensity range
        # # table.SetHueRange(0.666667, 0.0)
        # # table.SetSaturationRange(0.8, 0.8)
        # # table.SetValueRange(1.0, 1.0)
        # # table.SetAlphaRange(0.5, 1.0)
        # table.SetNumberOfColors(1025)
        # table.Build()
        #
        # # Map the image through the lookup table

        planeColors = vtk.vtkImageMapToColors()
        planeColors.SetLookupTable(colorTransferFunction)
        planeColors.SetInputConnection(self.resliced_main_window[index].GetOutputPort())
        planeColors.Update()

        # Display the image
        actor = vtk.vtkImageActor()
        actor.GetMapper().SetInputConnection(planeColors.GetOutputPort())

        print(actor.GetDisplayExtent())
        # actor.SetScale(0.5, 0.5, 0.5)
        # actor.SetDisplayExtent(xMin,xMax,zMin,zMax,yMin,yMax)
        if cutted_plane == "axial":
            self.listOfRenderers[1].AddActor(actor)
            self.listOfRenderers[1].ResetCamera()
            self.listOfRenderers[1].GetActiveCamera().ParallelProjectionOn()
            self.listOfRenderers[1].GetActiveCamera().SetParallelScale(center[0])
            self.actorInMain[0]['PET_axial'] = actor

            # distancetofocal = camera.GetDistance()
            # new_coordinates = np.array([-0.2, -1, 1]) * distancetofocal
            # camera.SetPosition(new_coordinates)
            # # center_volume = self.Volume4VTK.GetCenter()
            # # print(center_volume)
            # center_volume = [0, 0, 0]
            # camera.SetFocalPoint(center_volume)

            # self.listOfRenderers[1].GetActiveCamera().SetClippingRange()
            print(self.listOfRenderers[1].GetActiveCamera().GetClippingRange())

        if cutted_plane == "coronal":
            self.listOfRenderers[2].AddActor(actor)
            self.listOfRenderers[2].ResetCamera()
            self.listOfRenderers[2].GetActiveCamera().ParallelProjectionOn()
            self.listOfRenderers[2].GetActiveCamera().SetParallelScale(center[2])
            self.actorInMain[0]['PET_coronal'] = actor

        if cutted_plane == "sagittal":
            self.listOfRenderers[3].AddActor(actor)
            self.listOfRenderers[3].ResetCamera()
            self.listOfRenderers[3].GetActiveCamera().ParallelProjectionOn()
            self.listOfRenderers[3].GetActiveCamera().SetParallelScale(center[1] - 1)
            self.actorInMain[0]['PET_sagittal'] = actor

        # self.listOfRenderers[i].AddActor(actor)
        # self.listOfRenderers[i].ResetCamera()
        # cam = self.listOfRenderers[i].GetActiveCamera()
        # transform_camera = vtk.vtkTransform()
        # transform_camera.Scale(2, 2, 2)

        # cam.SetModelTransformMatrix(transform_camera.GetMatrix())
        # self.listOfRenderers[i].ResetCameraClippingRange (xMin,xMax,zMin,zMax,yMin,yMax)
        # print(self.listOfRenderers[i].GetDistance ())

        # self.listOfRenderers[i].ResetCamera()
        # actor.SetUserTransform(FOV_trans)
        # actor.RotateWXYZ(90, 0, 0, 1)
        # (xMin, xMax, zMin, zMax, yMin, yMax)=dataImporter.GetExecutive().GetWholeExtent(dataImporter.GetOutputInformation(0))
        # w=xMax
        # d=zMax
        # h=yMax
        # if cutted_plane == "axial":
        #     displayExtent=[[0, w], [int(d / 2), int(d / 2)], [0, h]]
        # #displayExtent=[[0, w], [0, d], [int(h / 2), int(h / 2)]]
        # if cutted_plane == "coronal":
        #     displayExtent=[[0, w], [0, d], [int(h / 2), int(h / 2)]]
        # if cutted_plane == "sagittal":
        #     displayExtent = [[0, w], [0, d], [int(h / 2), int(h / 2)]]
        #     actor.RotateWXYZ(90, 0, 0, 1)
        #
        # # displayExtent=[[30,30], [0, d], [0, h]]
        # actor.SetDisplayExtent(displayExtent[0][0], displayExtent[0][1], displayExtent[1][0], displayExtent[1][1],
        #                  displayExtent[2][0], displayExtent[2][1])
        # if cutted_plane == "axial":

        # matrix= [None]*36
        #
        # # move the center point that we are slicing through
        #
        #
        # for i in range(36):
        #
        #     reslice.Update()
        #
        #     sliceSpacing = reslice.GetOutput().GetSpacing()[2]
        #     matrix = reslice.GetResliceAxes()
        #     center = matrix.MultiplyPoint((0, 0, sliceSpacing * (i-18), 1))
        #     matrix.SetElement(0, 3, center[0])
        #     matrix.SetElement(1, 3, center[2])
        #     matrix.SetElement(2, 3, center[1])
        #     reslice.Update()
        # self.vtkWidget.GetRenderWindow().Render()

        # interactorStyleImage = vtk.vtkInteractorStyleImage()
        # interactorStyleImage.SetInteractionModeToImageSlicing()

        # interactorStyleImage = vtk.vtkInteractorStyleRubberBand3D()
        # interactorStyleImage = vtk.vtkInteractorStyleRubberBand2D()
        # interactorStyleImage = vtk.vtkInteractorStyleAreaSelectHover()
        # interactorStyleImage = vtk.vtkInteractorStyleDrawPolygon()
        # interactorStyleImage = vtk.vtkInteractorStyleRubberBand3D()
        # interactorStyleImage = vtk.vtkInteractorStyleSwitch()
        # self.vtkWidget.GetRenderWindow().GetInteractor().SetInteractorStyle(interactorStyleImage)

    def design_scalar_bar(self, colorTransferFunction, renderInteractor):
        label = vtk.vtkTextProperty()
        label.SetFontSize(10)

        scalar_bar = vtk.vtkScalarBarActor()
        # scalar_bar.SetOrientationToHorizontal()
        scalar_bar.SetLookupTable(colorTransferFunction)
        scalar_bar.SetNumberOfLabels(3)
        scalar_bar.SetLabelFormat('%.1f')
        # scalar_bar.SetLabelTextProperty(label)

        scalar_bar_widget = vtk.vtkScalarBarWidget()
        scalar_bar_widget.SetScalarBarActor(scalar_bar)
        scalar_bar_widget.SetInteractor(renderInteractor)
        scalar_bar_widget.SetCurrentRenderer(self.listOfRenderers[0])
        scalarBarRep = scalar_bar_widget.GetRepresentation()
        scalarBarRep.SetOrientation(1)  # 0 = Horizontal, 1 = Vertical
        scalarBarRep.SetPosition(0.87, 0.7)
        scalarBarRep.SetPosition2(0.1, 0.25)
        scalar_bar_widget.On()

        return scalar_bar_widget, scalar_bar

    def design_axes(self, viewport=[]):
        # renderInteractor=self.vtkWidget.GetRenderWindow().GetInteractor()#
        renderInteractor = self.interactorMainWindow
        axes = vtk.vtkAxesActor()
        axes.SetXAxisLabelText('R')
        axes.SetYAxisLabelText('I')
        axes.SetZAxisLabelText('A')
        # axes.GetXAxisShaftProperty().SetColor(1, 1, 0)
        # axes.GetXAxisTipProperty().SetColor(1, 1, 0)
        # axes.GetYAxisShaftProperty().SetColor(1, 1, 0)
        # axes.GetYAxisTipProperty().SetColor(1, 1, 0)
        axes.GetZAxisShaftProperty().SetColor(1, 1, 0)
        axes.GetZAxisTipProperty().SetColor(1, 1, 0)
        widget = vtk.vtkOrientationMarkerWidget()

        widget.SetOrientationMarker(axes)
        widget.SetInteractor(renderInteractor)
        widget.SetCurrentRenderer(self.listOfRenderers[0])
        widget.SetViewport(-.3, 0, 0.20, 0.30)
        widget.SetEnabled(1)
        widget.InteractiveOff()

        return widget, axes

    def camera_initial_position(self):
        camera = self.listOfRenderers[0].GetActiveCamera()
        camera.Zoom(0.3)

        # camera.SetClippingRange(21.9464, 30.0179)

        # camera.SetFocalPoint(3.49221, 2.28844, -0.970866)
        distancetofocal = camera.GetDistance()
        new_coordinates = np.array([-0.2, -1, 1]) * distancetofocal
        camera.SetPosition(new_coordinates)
        # center_volume = self.Volume4VTK.GetCenter()
        # print(center_volume)
        center_volume = [0, 0, 0]
        camera.SetFocalPoint(center_volume)

        camera.SetViewUp(0, 1, 0)
        camera.Azimuth(-60)
        camera.Yaw(45)
        camera.Roll(45)
        camera.OrthogonalizeViewUp()
        self.listOfRenderers[0].ResetCamera()

        renderWin = self.vtkWidget.GetRenderWindow()
        renderWin.Render()  # camera.SetViewAngle(30)

        print(camera.GetPosition())

        # self.ren_volume.GetActiveCamera().Yaw(90)
        # self.ren_volume.GetActiveCamera().Azimuth(90)

    def camera_rotation(self):
        sender = self.window_t.sender()

        camera_rotation_button_list = [self.changecamera2left_toolbutton, self.changecamera2right_toolbutton,
                                       self.changecamera2bottom_toolbutton,
                                       self.changecamera2top_toolbutton, self.changecamera2front_toolbutton,
                                       self.changecamera2back_toolbutton]
        camera_rotation_button = camera_rotation_button_list.index(sender)
        camera = self.listOfRenderers[0].GetActiveCamera()

        distancetofocal = camera.GetDistance()
        camera.SetPosition(0, 0, 0)
        if camera_rotation_button == 0:

            new_coordinates = np.array([-1, 0, 0]) * distancetofocal
            view_up_coordinates = np.array([0, 0, 1]) * distancetofocal
            rotation = 90



        elif camera_rotation_button == 1:
            new_coordinates = np.array([1, 0, 0]) * distancetofocal
            view_up_coordinates = np.array([0, 0, 1]) * distancetofocal
            rotation = -90


        elif camera_rotation_button == 2:
            new_coordinates = np.array([0, 0, -1]) * distancetofocal
            view_up_coordinates = np.array([0, 1, 0]) * distancetofocal
            rotation = 180

        elif camera_rotation_button == 3:
            new_coordinates = np.array([0, 0, 1]) * distancetofocal
            view_up_coordinates = np.array([0, 1, 0]) * distancetofocal
            rotation = 0

        elif camera_rotation_button == 4:
            new_coordinates = np.array([0, -1, 0]) * distancetofocal
            view_up_coordinates = np.array([1, 0, 0]) * distancetofocal
            rotation = 0

        elif camera_rotation_button == 5:
            new_coordinates = np.array([0, 1, 0]) * distancetofocal
            view_up_coordinates = np.array([1, 0, 0]) * distancetofocal
            rotation = 180

        camera.SetPosition(new_coordinates)
        camera.SetFocalPoint(new_coordinates / distancetofocal)
        camera.SetViewUp(view_up_coordinates)
        camera.SetViewAngle(0)
        camera.SetRoll(rotation)
        # camera.Azimuth(0)
        # camera.Yaw(0)
        # camera.OrthogonalizeViewUp()
        # camera.ParallelProjectionOn()

        #
        self.listOfRenderers[0].ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

        # ##### FUTURE to movei rotation #####
        # focal_point = camera.GetFocalPoint()
        # view_up = camera.GetViewUp()
        # position = camera.GetPosition()
        #
        # axis = [0, 0, 0]
        # axis[0] = -1 * camera.GetViewTransformMatrix().GetElement(0, 0)
        # axis[1] = -1 * camera.GetViewTransformMatrix().GetElement(0, 1)
        # axis[2] = -1 * camera.GetViewTransformMatrix().GetElement(0, 2)
        #
        # print(position, focal_point, view_up,)
        #
        # print(camera.GetViewTransformMatrix())
        # print(camera.GetViewTransformMatrix().GetElement(0, 0))
        # print(camera.GetViewTransformMatrix().GetElement(0, 1))
        # print(camera.GetViewTransformMatrix().GetElement(0, 2))
        # for n, q in enumerate([10] * 90):
        #     transform = vtk.vtkTransform()
        #     transform.Identity()
        #
        #     transform.Translate(*center)
        #     transform.RotateWXYZ(q, view_up)
        #     transform.RotateWXYZ(0, axis)
        #     transform.Translate(*[-1 * x for x in center])
        #
        #     new_position = [0, 0, 0]
        #     new_focal_point = [0, 0, 0]
        #     transform.TransformPoint(position, new_position)
        #     transform.TransformPoint(focal_point, new_focal_point)
        #
        #     camera.SetPosition(new_position)
        #     camera.SetFocalPoint(new_focal_point)
        #
        #     focal_point = camera.GetFocalPoint()
        #     view_up = camera.GetViewUp()
        #     position = camera.GetPosition()
        #
        #     camera.OrthogonalizeViewUp()

    def design_stl_file(self, filename_stl, translate_x, translate_y, translate_z, rotate_x, rotate_y, rotate_z,
                        color_actor, opacity_actor):
        path_stl = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(path_stl, 'bin', "{}.stl".format(filename_stl))

        reader = vtk.vtkSTLReader()
        reader.SetFileName(filename)

        transL1 = vtk.vtkTransform()
        # transL1.Translate(-10, 150, -10)
        transL1.RotateX(rotate_x)
        transL1.Translate(translate_x, translate_y, translate_z)
        # transL1.Scale(5, 5, 5)

        # Move the label to a new position.
        labelTransform = vtk.vtkTransformPolyDataFilter()
        labelTransform.SetTransform(transL1)
        labelTransform.SetInputConnection(reader.GetOutputPort())

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(labelTransform.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color_actor)
        actor.GetProperty().SetOpacity(opacity_actor)

        return actor

    # def volume_update(self):
    #     # Updates data of the volume if dimensions are the same
    #
    #     volumeMapper = self.Volume4VTK.GetMapper()
    #     data3D = self.volume
    #     data3D = np.flip(data3D, axis=2)
    #     size = data3D.shape
    #     updatedata_image = volumeMapper.GetDataSetInput()
    #
    #     w = size[0]
    #     h = size[1]
    #     d = size[2]
    #     stack = np.zeros((w, d, h))
    #
    #     for j in range(0, d):
    #         stack[:, j, :] = data3D[:, :, j]
    #
    #     stack = np.require(stack, dtype=np.float32)
    #     dataImporter = vtk.vtkImageImport()
    #     # The preaviusly created array is converted to a string of chars and imported.
    #     data_string = stack.tostring()
    #     dataImporter.CopyImportVoidPointer(data_string, len(data_string))
    #     #dataImporter.SetDataScalarTypeToUnsignedShort()
    #     dataImporter.SetDataScalarTypeToFloat()
    #     dataImporter.SetDataSpacing(.pixel_size_reconstruct_file[0], self.pixel_size_reconstruct_file[2],
    #                                 self.pixel_size_reconstruct_file[1])
    #     # Because the data that is imported only contains an intensity value (it isnt RGB-coded or someting similar), the importer
    #     # must be told this is the case.
    #     dataImporter.SetNumberOfScalarComponents(1)
    #     # ---------------STORE DATA-------------------------------
    #     dataImporter.SetDataExtent(0, w - 1, 0, d - 1, 0, h - 1)
    #     dataImporter.SetWholeExtent(0, w - 1, 0, d - 1, 0, h - 1)
    #
    #     # -----------------------Scalar range-------------
    #     #dataImporter.Modified()
    #     dataImporter.Update()
    #     new_data = dataImporter.GetOutput().GetPointData().GetScalars()
    #     updatedata_image.GetPointData().SetScalars(new_data)
    #     #dataImporter.SetInputConnection(dataImporter.GetOutputPort())
    #     #updatedata_image.Modified()
    #     #updatedata_image.Update()
    #
    #     #volumeMapper.SetInputConnection(dataImporter.GetOutputPort())
    #     #volumeMapper.Modified()
    #     #volumeMapper.Update()
    #     #self.Volume4VTK.SetMapper(volumeMapper)
    #     #image_3D_handler.update_volume_colors(self)
    #     self.Volume4VTK.Update()
    #     self.renderWin.Render()
    #
    # # Control of which volume is selected

    def hide_qt_buttons(self, hide_camera_buttons=False, hide_bed_buttons=False):
        camera_rotation_button_list = [self.changecamera2left_toolbutton, self.changecamera2right_toolbutton,
                                       self.changecamera2bottom_toolbutton,
                                       self.changecamera2top_toolbutton, self.changecamera2front_toolbutton,
                                       self.changecamera2back_toolbutton, self.resetcamera_toolButton]
        bed_buttons_list = [self.movebed_to_begin_toolbutton, self.movebed_backward_toolbutton,
                            self.movebed_foward_toolbutton,
                            self.movebed_to_end_toolbutton, self.emergency_stop_bed_tool_button]

        if hide_camera_buttons:

            for button in camera_rotation_button_list:
                button.hide()
        else:
            for button in camera_rotation_button_list:
                button.show()

        if hide_bed_buttons:
            for button in bed_buttons_list:
                button.hide()
        else:
            for button in bed_buttons_list:
                button.show()
