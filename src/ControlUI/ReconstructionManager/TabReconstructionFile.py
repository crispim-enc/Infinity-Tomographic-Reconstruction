import logging
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, QObject, pyqtSlot
from PyQt5.QtGui import  QPixmap
# from PyQt5.QtCore import (QDate, QDateTime, QRegExp, QSortFilterProxyModel, Qt, QTimer, QSize)
from PyQt5.QtWidgets import QFileDialog

import sys
import vtk
from vtk.util import numpy_support
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import os
import numpy as np


class TabReconstruction(QtCore.QObject):
    def __init__(self):
        # super().__init__()
        TabReconstruction._update_variables_tab(self)
        TabReconstruction.easypet_data_show_options(self)
        TabReconstruction.updateVisible_algorithm_options(self)
        TabReconstruction._protect_options(self)
        print("Tab init")

    def _protect_options(self):
        self.iterative_reconstruct_combobox_algorithm_type.model().item(1).setEnabled(False)
        for i in range(3,8):
            self.iterative_reconstruct_combobox_algorithm_type.model().item(i).setEnabled(False)

        for i in range(1,4):
            self.geometric_project_correction_combobox.model().item(i).setEnabled(False)

        for i in range(2,5):
            self.normalization_combobox.model().item(i).setEnabled(False)

        self.iterative_reconstruct_push_button_easypet_data_original.setEnabled(False)
        self.progressBar_reconstruction_total.hide()
        # self.gpu_design_multiple_kernel_push_button.setEnabled(False)

    # def _set_exclusive_groups(self):
    #
    #     self.iterative_reconstruct_all_detectors.setChecked(True)
    #     self.slow_acquisition_pushButton.setAutoExclusive(False)
    #     self.slow_acquisition_pushButton.setChecked(False)
    #     self.slow_acquisition_pushButton.setAutoExclusive(True)

    def _update_variables_tab(self, acquisitionInfo=None, otherinfo=None):
        if self.iterative_reconstruct_push_button_easypet_data_original.isChecked():
            self.data_choosen = ".easypetoriginal"
        else:
            self.data_choosen = ".easypet"


        self.parameters_2D_cut = [self.iterative_reconstruc_spin_box_sideA_parameterA.value(),
                                  self.iterative_reconstruc_spin_box_sideA_parameterB.value(),
                                  self.iterative_reconstruc_spin_box_sideB_parameterA.value(),
                                  self.iterative_reconstruc_spin_box_sideB_parameterB.value()]

        self.ratio_threshold = self.iterative_reconstruct_doublespinbox_ratiothreshold.value()
        adc_value = self.iterative_reconstruct_doublespinbox_ratiothreshold.value()

        self.ADCs_threshold_cut =[adc_value]*4
        self.top_correction_angle = self.angleCorrection_doubleSpinBox.value()
        self.top_acceleration_cut = self.iterative_reconstruct_doublespinbox_acceleration_ramp.value()/100
        self.energy_pushbuttons = [self.energywindow_pushButton_430_620.isChecked(),
                                   self.energywindow_pushButton_320_720.isChecked(),
                                   self.energywindow_pushButton_0_1024.isChecked()]
        energy_window_set = [[430, 620], [320, 720], [0, 1024]]
        self.energy_window = []
        for i in range(len(self.energy_pushbuttons)):
            if self.energy_pushbuttons[i]:
                self.energy_window = energy_window_set[i]

        self.remove_first_row_bool = self.iterative_reconstruct_remove_first_row.isChecked()
        self.remove_last_row_bool = self.iterative_reconstruct_remove_first_row.isChecked()
        self.remove_peripheral_crystals = self.iterative_reconstruct_remove_peripheral.isChecked()
        self.only_left_detectors_bool = self.iterative_reconstruct_only_left_detectors.isChecked()
        self.only_right_detectors_bool = self.iterative_reconstruct_only_right_detectors.isChecked()
        self.all_detectors_bool = self.iterative_reconstruct_all_detectors.isChecked()
        self.cut_initial_frames = self.iterative_reconstruct_spinBox_cut_initial_frames.value()
        self.cut_end_frames = self.iterative_reconstruct_spinBox_cut_end_frames.value()
        self.remove_incomplete_turn = self.iterative_reconstruct_remove_incomplete_turn.isChecked()

        # PAGE 2
        self.algorithm = self.iterative_reconstruct_combobox_algorithm_type.currentText()
        self.include_pet_info_in_map = self.iterative_reconstruct_map_pet_bool.isChecked()
        self.algorithm_pet_used_to_map = self.iterative_reconstruct_map_pet_combobox.currentText()
        self.include_ct_info_in_map = self.iterative_reconstruct_map_ct_bool.isChecked()
        self.beta_value_penalized_algorithms = self.iterative_reconstruct_beta_double_spin_box.value()
        self.mrp_kernel_size = self.iterative_reconstruct_MRP_double_spin_box.value()
        self.filter_fbp_type = self.filtertype_comboBox.currentText()
        self.regularization_fbp_filter = self.regularization_histogram_comboBox.currentText()
        self.number_of_subsets = self.number_of_subs_spinBox.value()
        self.number_of_iterations = self.number_iterations_spinBox.value()
        self.single_kernel_bool = self.gpu_design_single_kernel_push_button.isChecked()
        self.multiple_kernel_bool = self.gpu_design_multiple_kernel_push_button.isChecked()

        # geometry
        self.geometry_2D_bool = self.iterative_reconstruct_push_button_2D.isChecked()
        self.geometry_3D_bool = self.iterative_reconstruct_push_button_3D.isChecked()
        self.fbp_fronttofront = self.iterative_reconstruct_push_button_front_to_front.isChecked()
        self.fbp_axial_rebinned = self.iterative_reconstruct_push_button_axial_rebinned.isChecked()
        self.axial_neighbours = self.neighbours_comboBox.currentText()
        pixel_size = self.pixelsize_comboBox.currentText()
        pixel_size = pixel_size.split(' ')
        self.pixel_size = float(pixel_size[0])
        # geometriccorrection
        self.geometric_projector = self.geometric_project_correction_combobox.currentText()
        self.normalization_type = self.normalization_combobox.currentText()
        self.attenuation_type = self.attenuation_combobox.currentText()
        self.decay_correction_bool = self.iterative_reconstruct_push_button_decay.isChecked()
        self.scatter_correction_bool = self.iterative_reconstruct_push_button_scatter_angle.isChecked()
        self.dead_time_bool = self.iterative_reconstruct_push_button_scatter_angle.isChecked()
        self.respiratory_movement_correction_bool = self.iterative_reconstruct_push_button_respiratory_motion.isChecked()
        self.doi_correction_bool = self.iterative_reconstruct_push_button_doi.isChecked()
        self.randoms_correction_bool = self.iterative_reconstruct_push_button_randoms.isChecked()
        self.heart_movement_correction_bool = self.iterative_reconstruct_push_button_heart_motion.isChecked()

        # Page3
        self.reconstruct_whole_body = self.iterative_reconstruct_push_button_whole_body.isChecked()
        self.reconstruct_static = self.iterative_reconstruct_push_button_static.isChecked()
        self.reconstruct_dynamic = self.iterative_reconstruct_push_button_dynamic.isChecked()
        self.reconstruct_gated = self.iterative_reconstruct_push_button_gated.isChecked()
        TabReconstruction.set_temporal_cuts_values(self,acquisitionInfo, otherinfo)

    def updateVisible_buttons(self):
        if self.iterative_reconstruct_push_button_easypet_data_original.isChecked():
            TabReconstruction.original_data_show_options(self)

        elif self.iterative_reconstruct_push_button_easypet_data.isChecked():
            TabReconstruction.easypet_data_show_options(self)

    def updateVisible_algorithm_options(self):
        widgets_to_hide = []
        widgets_to_show = []
        print(self.iterative_reconstruct_combobox_algorithm_type.currentText() )
        if self.iterative_reconstruct_combobox_algorithm_type.currentText() == "LM-MLEM" or \
                self.iterative_reconstruct_combobox_algorithm_type.currentText() == "MLEM":
            widgets_to_show = [self.label_iterations, self.number_iterations_spinBox,
                               self.gpu_design_single_kernel_push_button, self.gpu_design_multiple_kernel_push_button,
                               self.label_gpu_design]

            widgets_to_hide= [self.label_filter_type,
                               self.filtertype_comboBox, self.label_axial,
                              self.iterative_reconstruct_push_button_front_to_front,
                              self.iterative_reconstruct_push_button_axial_rebinned,
                              self.label_axial, self.label_regularization, self.regularization_histogram_comboBox,
                              self.iterative_reconstruct_map_pet_bool, self.iterative_reconstruct_map_pet_combobox,
                              self.iterative_reconstruct_map_ct_bool, self.iterative_reconstruct_map_ct_combobox,
                              self.label_map_generation, self.iterative_reconstruct_beta_double_spin_box,
                              self.label_beta, self.iterative_reconstruct_MRP_double_spin_box,
                              self.label_mrp_kernel_size, self.label_subsets, self.number_of_subs_spinBox,]

            # self.iterative_reconstruct_push_button_3D.setAutoExclusive(False)
            self.iterative_reconstruct_push_button_3D.setChecked(True)
            self.iterative_reconstruct_push_button_2D.setChecked(False)
            self.iterative_reconstruct_push_button_3D.setEnabled(True)
            self.iterative_reconstruct_push_button_2D.setAutoExclusive(True)
            self.iterative_reconstruct_push_button_3D.setAutoExclusive(True)

            self.neighbours_comboBox.setCurrentIndex(0)

        elif self.iterative_reconstruct_combobox_algorithm_type.currentText() == "AnalyticalReconstruction-2D":
            widgets_to_hide = [self.iterative_reconstruct_map_pet_bool, self.iterative_reconstruct_map_pet_combobox,
                               self.iterative_reconstruct_map_ct_bool, self.iterative_reconstruct_map_ct_combobox,
                               self.label_map_generation, self.iterative_reconstruct_beta_double_spin_box,
                               self.label_beta, self.iterative_reconstruct_MRP_double_spin_box,
                               self.label_mrp_kernel_size, self.label_subsets, self.number_of_subs_spinBox,
                               self.label_iterations, self.number_iterations_spinBox,
                               self.gpu_design_single_kernel_push_button, self.gpu_design_multiple_kernel_push_button,
                               self.label_gpu_design]

            widgets_to_show = [self.label_filter_type,
                               self.label_regularization,
                               self.regularization_histogram_comboBox,
                               self.filtertype_comboBox, self.label_axial,
                               self.iterative_reconstruct_push_button_front_to_front,
                               self.iterative_reconstruct_push_button_axial_rebinned]

            self.iterative_reconstruct_push_button_2D.setChecked(True)
            self.iterative_reconstruct_push_button_3D.setEnabled(False)
            self.iterative_reconstruct_push_button_axial_rebinned.setChecked(True)
            self.iterative_reconstruct_push_button_2D.setAutoExclusive(True)
            self.iterative_reconstruct_push_button_3D.setAutoExclusive(True)
            self.neighbours_comboBox.setCurrentIndex(1)


        elif self.iterative_reconstruct_combobox_algorithm_type.currentText() == "LM-MRP":
            widgets_to_show = [self.iterative_reconstruct_beta_double_spin_box,
                               self.label_beta, self.iterative_reconstruct_MRP_double_spin_box,
                               self.label_mrp_kernel_size,
                               self.label_iterations, self.number_iterations_spinBox,
                               self.gpu_design_single_kernel_push_button, self.gpu_design_multiple_kernel_push_button,
                               self.label_gpu_design]

            widgets_to_hide = [self.label_map_generation, self.iterative_reconstruct_map_pet_bool,
                               self.iterative_reconstruct_map_pet_combobox, self.iterative_reconstruct_map_ct_bool,
                               self.iterative_reconstruct_map_ct_combobox, self.filtertype_comboBox,
                               self.label_filter_type, self.label_regularization, self.regularization_histogram_comboBox,
                               self.number_of_subs_spinBox, self.label_subsets, self.label_axial,
                               self.iterative_reconstruct_push_button_front_to_front,
                               self.iterative_reconstruct_push_button_axial_rebinned]

            self.label_beta.setText("Beta MRP")
            self.neighbours_comboBox.setCurrentIndex(0)


            self.iterative_reconstruct_push_button_3D.setChecked(True)
            self.iterative_reconstruct_push_button_2D.setChecked(False)
            self.iterative_reconstruct_push_button_2D.setAutoExclusive(True)
            self.iterative_reconstruct_push_button_3D.setAutoExclusive(True)

        elif self.iterative_reconstruct_combobox_algorithm_type.currentText() == "LM-OSEM" or \
                self.iterative_reconstruct_combobox_algorithm_type.currentText() == "OSEM":
            widgets_to_show = [self.label_iterations, self.number_iterations_spinBox,
                               self.gpu_design_single_kernel_push_button, self.gpu_design_multiple_kernel_push_button,
                               self.label_gpu_design,self.label_subsets, self.number_of_subs_spinBox, ]

            widgets_to_hide = [self.label_filter_type,
                               self.filtertype_comboBox, self.label_axial,
                               self.iterative_reconstruct_push_button_front_to_front,
                               self.iterative_reconstruct_push_button_axial_rebinned,
                               self.label_axial, self.label_regularization, self.regularization_histogram_comboBox,
                               self.iterative_reconstruct_map_pet_bool, self.iterative_reconstruct_map_pet_combobox,
                               self.iterative_reconstruct_map_ct_bool, self.iterative_reconstruct_map_ct_combobox,
                               self.label_map_generation, self.iterative_reconstruct_beta_double_spin_box,
                               self.label_beta, self.iterative_reconstruct_MRP_double_spin_box,
                               self.label_mrp_kernel_size, ]



            self.iterative_reconstruct_push_button_3D.setChecked(True)
            self.iterative_reconstruct_push_button_2D.setChecked(False)
            self.iterative_reconstruct_push_button_2D.setAutoExclusive(True)
            self.iterative_reconstruct_push_button_3D.setAutoExclusive(True)

            self.iterative_reconstruct_push_button_3D.setEnabled(True)
            self.neighbours_comboBox.setCurrentIndex(0)

        elif self.iterative_reconstruct_combobox_algorithm_type.currentText() == "MAP":
            widgets_to_show = [self.iterative_reconstruct_beta_double_spin_box,
                               self.label_beta, self.iterative_reconstruct_MRP_double_spin_box,
                               self.label_map_generation, self.iterative_reconstruct_map_pet_bool,
                               self.iterative_reconstruct_map_pet_combobox, self.iterative_reconstruct_map_ct_bool,
                               self.iterative_reconstruct_map_ct_combobox,

                               self.label_iterations, self.number_iterations_spinBox,
                               self.gpu_design_single_kernel_push_button, self.gpu_design_multiple_kernel_push_button]

            widgets_to_hide = [ self.filtertype_comboBox,
                               self.label_filter_type, self.label_regularization,
                               self.regularization_histogram_comboBox,
                               self.number_of_subs_spinBox, self.label_subsets, self.label_axial,
                               self.iterative_reconstruct_push_button_front_to_front,
                               self.iterative_reconstruct_push_button_axial_rebinned,
                               self.label_mrp_kernel_size]

            self.label_beta.setText("Beta MAP")
            self.neighbours_comboBox.setCurrentIndex(0)
            # self.iterative_reconstruct_push_button_3D.setAutoExclusive(True)

            self.iterative_reconstruct_push_button_3D.setChecked(True)
            self.iterative_reconstruct_push_button_2D.setChecked(False)
            self.iterative_reconstruct_push_button_2D.setAutoExclusive(True)
            self.iterative_reconstruct_push_button_3D.setAutoExclusive(True)


        if widgets_to_hide is not None:
            for widget in widgets_to_hide:
                widget.hide()

        if widgets_to_hide is not None:
            for widget in widgets_to_show:
                widget.show()

    def original_data_show_options(self):
        list_options_to_show = [self.iterative_reconstruc_spin_box_sideA_parameterA,
                                self.iterative_reconstruc_spin_box_sideA_parameterB,
                                self.iterative_reconstruc_spin_box_sideB_parameterA,
                                self.iterative_reconstruc_spin_box_sideB_parameterB,
                                self.iterative_reconstruct_doublespinbox_ratiothreshold,
                                self.iterative_reconstruct_doublespinbox_adc_threshold,
                                self.label_sideA, self.label_sideB, self.label_ratio,
                                self.label_adcthreshold, self.label_accelerationramp]
        list_options_to_hide = [self.energywindow_pushButton_430_620, self.energywindow_pushButton_320_720, self.energywindow_pushButton_0_1024]

        for element in list_options_to_show:
            element.show()

        for element in list_options_to_hide:
            element.hide()

    def easypet_data_show_options(self):
        list_options_to_hide = [self.iterative_reconstruc_spin_box_sideA_parameterA,
                                self.iterative_reconstruc_spin_box_sideA_parameterB,
                                self.iterative_reconstruc_spin_box_sideB_parameterA,
                                self.iterative_reconstruc_spin_box_sideB_parameterB,
                                self.iterative_reconstruct_doublespinbox_ratiothreshold,
                                self.iterative_reconstruct_doublespinbox_adc_threshold,
                                self.label_sideA, self.label_sideB, self.label_ratio,
                                self.label_adcthreshold]

        list_options_to_show = [self.energywindow_pushButton_430_620, self.energywindow_pushButton_320_720,
                                self.energywindow_pushButton_0_1024]

        for element in list_options_to_show:
                element.show()

        for element in list_options_to_hide:
            element.hide()

    def set_temporal_cuts_values(self, acquisitionInfo, otherinfo):
        if acquisitionInfo is None:
            number_of_turns = 2
        else:
            index_end_turn = acquisitionInfo['Turn end index'].split(' ')
            number_of_turns = len(index_end_turn)

        if otherinfo is None:
            aborted_acquisition=False
        else:
            aborted_acquisition = otherinfo[0].split(':')
            aborted_acquisition = aborted_acquisition[1]
        end_maximum = number_of_turns
        if aborted_acquisition:
            end_maximum = number_of_turns - 1
        initial_maximum = end_maximum - 1

        self.iterative_reconstruct_spinBox_cut_initial_frames.setMaximum(initial_maximum)
        self.iterative_reconstruct_spinBox_cut_initial_frames.setValue(0)
        self.iterative_reconstruct_spinBox_cut_end_frames.setMaximum(end_maximum)
        self.iterative_reconstruct_spinBox_cut_end_frames.setValue(end_maximum)


class EnergyWindowVTKWidget(QtCore.QObject):
    def __init__(self, default_view="2D"):
        self.default_view = default_view
        self.layout_vtk_window_selection_hist= QtWidgets.QVBoxLayout()
        self.layout_vtk_window_selection_hist.setContentsMargins(0, 0, 0, 0)
        self.vtkWidget_window_selection = vtkWidget_window_selection = QVTKRenderWindowInteractor(self.window_selection_hist)
        self.layout_vtk_window_selection_hist.addWidget(self.vtkWidget_window_selection)
        self.interactor_window_selection_hist = interactor_window_selection_hist = self.vtkWidget_window_selection.GetRenderWindow().GetInteractor()

        # rend.setBackground(backgroundColor.GetData())
        #
        # backgroundColor = colors.GetColor3d("SlateGray")
        # EnergyWindow._add_3d_histogram(self)
        # EnergyWindow._add_2d_histogram(self)

        self.vtkWidget_window_selection.show()
        self.interactor_window_selection_hist.Initialize()
        self.vtkWidget_window_selection.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.vtkWidget_window_selection.GetRenderWindow().Render()
        self.window_selection_hist.setLayout(self.layout_vtk_window_selection_hist)
        self.view = vtk.vtkContextView()
        self.view.SetRenderWindow(self.vtkWidget_window_selection.GetRenderWindow())
        self.barchart = None
        EnergyWindowVTKWidget._energy_histogram(self)
        self.view.GetScene().RemoveItem(self.barchart)


    # def _update_histogram_area

    def _add_3d_histogram(self):

        ###########################################################
        # CREATE ARRAY VALUES
        ###########################################################
        # Just create some fancy looking values for z.
        n = 100
        m = 50
        xmin = -1
        xmax = 1
        ymin = -1
        ymax = 1
        x = np.linspace(xmin, xmax, n)
        y = np.linspace(ymin, ymax, m)
        x, y = np.meshgrid(x, y)
        x, y = x.flatten(), y.flatten()
        z = (x + y) * np.exp(-3.0 * (x ** 2 + y ** 2))

        ###########################################################
        # CREATE PLANE
        ###########################################################
        # Create a planar mesh of quadriliterals with nxm points.
        # (SetOrigin and SetPointX only required if the extent
        # of the plane should be the same. For the mapping
        # of the scalar values, this is not required.)
        plane = vtk.vtkPlaneSource()
        plane.SetResolution(n - 1, m - 1)
        plane.SetOrigin([xmin, ymin, 0])  # Lower left corner
        plane.SetPoint1([xmax, ymin, 0])
        plane.SetPoint2([xmin, ymax, 0])
        plane.Update()

        # Map the values to the planar mesh.
        # Assumption: same index i for scalars z[i] and mesh points
        nPoints = plane.GetOutput().GetNumberOfPoints()
        assert (nPoints == len(z))
        # VTK has its own array format. Convert the input
        # array (z) to a vtkFloatArray.
        scalars = vtk.vtkFloatArray()
        scalars.SetNumberOfValues(nPoints)
        for i in range(nPoints):
            scalars.SetValue(i, z[i])
        # Assign the scalar array.
        plane.GetOutput().GetPointData().SetScalars(scalars)

        ###########################################################
        # WRITE DATA
        ###########################################################
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName('output.vtp')
        writer.SetInputConnection(plane.GetOutputPort())
        writer.Write()  # => Use for example ParaView to see scalars

        ###########################################################
        # VISUALIZATION
        ###########################################################
        # This is a bit annoying: ensure a proper color-lookup.
        colorSeries = vtk.vtkColorSeries()
        colorSeries.SetColorScheme(vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_10)
        lut = vtk.vtkColorTransferFunction()
        lut.SetColorSpaceToHSV()
        nColors = colorSeries.GetNumberOfColors()
        zMin = np.min(z)
        zMax = np.max(z)
        for i in range(0, nColors):
            color = colorSeries.GetColor(i)
            color = [c / 255.0 for c in color]
            t = zMin + float(zMax - zMin) / (nColors - 1) * i
            lut.AddRGBPoint(t, color[0], color[1], color[2])

        # Mapper.
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(plane.GetOutputPort())
        mapper.ScalarVisibilityOn()
        mapper.SetScalarModeToUsePointData()
        mapper.SetLookupTable(lut)
        mapper.SetColorModeToMapScalars()
        # Actor.
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        # Renderer.
        renderer = vtk.vtkRenderer()
        renderer.SetBackground([0.5] * 3)
        # Render window and interactor.
        # renderWindow = vtk.vtkRenderWindow()
        # renderWindow.SetWindowName('Demo')
        self.vtkWidget_window_selection.GetRenderWindow().Render()
        self.vtkWidget_window_selection.GetRenderWindow().AddRenderer(renderer)
        renderer.AddActor(actor)
        # interactor = vtk.vtkRenderWindowInteractor()
        # interactor.SetRenderWindow(renderWindow)
        # renderWindow.Render()
        # interactor.Start()

    def _add_2d_histogram(self,listMode_original= None):
        # colors=vtk.vtkColor3d()
        # backgroundColor = colors.GetColor3d("SlateGray")
        #
        # titleColor = colors.GetColor3d("Orange");
        #
        # axisTitleColor = colors.GetColor3d("Orange");
        #
        # axisLabelColor = colors.GetColor3d("Beige");
        #
        # legendBackgroundColor = colors.GetColor4ub("Tomato");
        listMode = listMode_original
        spectrumA = listMode[:, 0] + listMode[:, 1]
        spectrumB = listMode[:, 2] + listMode[:, 3]
        ratioA = (listMode[spectrumA != 0, 0] - listMode[spectrumA != 0, 1]) / spectrumA[spectrumA != 0]
        ratioB = (listMode[spectrumB != 0, 3] - listMode[spectrumB != 0, 2]) / spectrumB[spectrumB != 0]
        spectrumA = spectrumA[spectrumA != 0]
        spectrumB = spectrumB[spectrumB != 0]
        # Z_A, X, Y = np.histogram2d(ratioA, spectrumA, 750, range=[[np.nanmin(ratioA), np.nanmax(ratioB)], [np.nanmin(spectrumA), np.nanmax(spectrumA)]])
        number_of_bins=750
        ratio_lim=[-0.95,0.95]
        energy_lim = [200,6000]
        range_ratio = np.abs(ratio_lim[1]-ratio_lim[0])
        range_energy = np.abs(energy_lim[1]-energy_lim[0])
        Z_A, X, Y = np.histogram2d(ratioA, spectrumA, 750, range=[ratio_lim, energy_lim])
        Z_B, X, Y = np.histogram2d(ratioB, spectrumB, 750, range=[ratio_lim, energy_lim])

        size = 400
        view = vtk.vtkContextView()

        view.SetRenderWindow(self.vtkWidget_window_selection.GetRenderWindow())

        renderer = view.GetRenderer()
        # renderer= vtk.vtkRenderer()
        self.vtkWidget_window_selection.GetRenderWindow().AddRenderer(renderer)
        # renwin.AddRenderer(renderer)
        renderer.SetBackground([0] * 3)


#         vtkNew < vtkContextView > view;
#         view->GetRenderWindow()->SetSize(512, 512);
#         view->GetRenderer()->SetBackground(backgroundColor.GetData());
#
#         // Define
#         a
#         chart
#         vtkNew < vtkChartHistogram2D > chart;
#
#         // Chart
#         Title
#         chart->SetTitle("2D Histogram");
#         chart->GetTitleProperties()
#         ->SetFontSize(36);
#
#     chart->GetTitleProperties()
#     ->SetColor(titleColor.GetData());
#
# // Chart
        chart = vtk.vtkChartHistogram2D()
        # axis= chart.GetAxis(0).GetTitleProperties()
        # barChart = vtk.vtkChartXY()
        xAxis = chart.GetAxis(vtk.vtkAxis.BOTTOM)
        xAxis.SetTitle("Ratio")
        xAxis.GetTitleProperties().SetColor(1, 1, 1)
        # xAxis.GetTitleProperties().SetFontSize(16)
        # xAxis.GetTitleProperties().ItalicOn()
        xAxis.GetLabelProperties().SetColor(1, 1, 1)
        xAxis.GetPen().SetColor(255, 255, 255, 255)
        # xAxis.SetTicksVisible(False)
        xAxis.SetGridVisible(False)
        xAxis.SetRangeLabelsVisible(True)
        xAxis.SetRange(0, 1000)
        # xAxis.SetOffset(0)
        #
        # xAxis.SetUnscaledMinimum(200)

        # barChart = vtk.vtkChartXY()
        yAxis = chart.GetAxis(vtk.vtkAxis.LEFT)
        yAxis.SetTitle("ADC Channel")
        yAxis.GetTitleProperties().SetColor(1, 1, 1)
        # xAxis.GetTitleProperties().SetFontSize(16)
        # xAxis.GetTitleProperties().ItalicOn()
        yAxis.GetLabelProperties().SetColor(1, 1, 1)
        yAxis.GetPen().SetColor(255, 255, 255, 255)
        # xAxis.SetTicksVisible(False)
        yAxis.SetRangeLabelsVisible(True)
        yAxis.SetGridVisible(False)
        view.GetScene().AddItem(chart)

        yAxis_twin = chart.GetAxis(vtk.vtkAxis.TOP)
        yAxis_twin.GetTitleProperties().SetColor(1, 1, 1)
        yAxis_twin.GetLabelProperties().SetColor(1, 1, 1)
        yAxis_twin.SetAxisVisible(True)
# Axes
# chart->GetAxis(0)->GetTitleProperties()
# ->SetFontSize(24);
# chart->GetAxis(0)->GetTitleProperties()
# ->SetColor(axisTitleColor.GetData());
# chart->GetAxis(0)->GetLabelProperties()
# ->SetColor(axisLabelColor.GetData());
# chart->GetAxis(0)->GetLabelProperties()
# ->SetFontSize(18);
#
# chart->GetAxis(1)->GetTitleProperties()
# ->SetFontSize(24);
# chart->GetAxis(1)->GetTitleProperties()
# ->SetColor(colors->GetColor3d("orange").GetData());
# chart->GetAxis(1)->GetLabelProperties()
# ->SetColor(colors->GetColor3d("beige").GetData());
# chart->GetAxis(1)->GetLabelProperties()
# ->SetFontSize(18);
#
# // Chart
# Legend
# dynamic_cast < vtkColorLegend * > (chart->GetLegend())->DrawBorderOn();
# chart->GetLegend()->GetBrush()
# ->SetColor(legendBackgroundColor);
#
# // Add
# the
# chart
# to
# the
# view
# view->GetScene()->AddItem(chart);
#
        # Z_A =Z_A.T
        # Z_A=np.flip(Z_A, axis=1)
        Z_A=np.rot90(Z_A)
        data = EnergyWindow.numpy_array_as_vtk_image_data(self, Z_A)
        data.SetSpacing(range_ratio/number_of_bins, range_energy/number_of_bins, 1.0)
        data.SetOrigin(-1, 0, 0.0)
        # data = vtk.vtkImageData()
        # data.SetExtent(0,size-1,0,size-1,0,0)
        # data.AllocateScalars(vtk.VTK_DOUBLE,2)
        # data.SetOrigin(100.0,-100.0,0.0)
        # data.SetSpacing(2.0, 1.0, 1.0)
        # array = vtk.vtkDoubleArray()
        #
        # # dPtr = data.GetScalarPointer(0,0,0)
        # # dPtr = np.zeros((size*size))
        # for i in range(0,size*size):
        #     # for j in range(0,size):
        #     array.InsertNextValue(i)
        #         # dPtr[i * size + j] = np.random.randint(0,100)
        # data.GetPointData().AddArray(array)
# vtkNew < vtkImageData > data;
# data->SetExtent(0, size - 1, 0, size - 1, 0, 0);
# data->AllocateScalars(VTK_DOUBLE, 1);
#
# data->SetOrigin(100.0, -100.0, 0.0);
# data->SetSpacing(2.0, 1.0, 1.0);
#
# double * dPtr = static_cast < double * > (data->GetScalarPointer(0, 0, 0));
# for (int i = 0; i < size; ++i)
#     {
#     for (int j = 0; j < size; ++j)
#     {
#         dPtr[i * size + j] =
#     std::
#         sin(vtkMath::RadiansFromDegrees(double(2 * i))) *
#     std::cos(vtkMath::RadiansFromDegrees(double(j)));
#     }
#     }
#     chart->SetInputData(data);
        chart.SetInputData(data)
        transferFunction = vtk.vtkColorTransferFunction()
        transferFunction.AddHSVSegment(0, 0, 0, 0,
                                    np.max(Z_A),0.497,0.741, 1)
        # transferFunction.AddHSVSegment(0.3333, 0.3333, 1.0, 1.0,
        #                             0.6666, 0.6666, 1.0, 1.0)
        # transferFunction.AddHSVSegment(0.6666, 0.6666, 1.0, 1.0,
        #                             1.0, 0.2, 1.0, 0.3)
        transferFunction.Build()
        chart.SetTransferFunction(transferFunction)
#     vtkNew < vtkColorTransferFunction > transferFunction;
#     transferFunction->AddHSVSegment(0.0, 0.0, 1.0, 1.0,
#                                     0.3333, 0.3333, 1.0, 1.0);
#     transferFunction->AddHSVSegment(0.3333, 0.3333, 1.0, 1.0,
#                                     0.6666, 0.6666, 1.0, 1.0);
#     transferFunction->AddHSVSegment(0.6666, 0.6666, 1.0, 1.0,
#                                     1.0, 0.2, 1.0, 0.3);
#     transferFunction->Build();
#     chart->SetTransferFunction(transferFunction);
#
#     view->GetRenderWindow()->Render();
#     view->GetInteractor()->Start();

    def numpy_array_as_vtk_image_data(self, source_numpy_array):
        """
        :param source_numpy_array: source array with 2-3 dimensions. If used, the third dimension represents the channel count.
        Note: Channels are flipped, i.e. source is assumed to be BGR instead of RGB (which works if you're using cv2.imread function to read three-channel images)
        Note: Assumes array value at [0,0] represents the upper-left pixel.
        :type source_numpy_array: np.ndarray
        :return: vtk-compatible image, if conversion is successful. Raises exception otherwise
        :rtype vtk.vtkImageData
        """

        if len(source_numpy_array.shape) > 2:
            channel_count = source_numpy_array.shape[2]
        else:
            channel_count = 1

        output_vtk_image = vtk.vtkImageData()
        output_vtk_image.SetDimensions(source_numpy_array.shape[1], source_numpy_array.shape[0], channel_count)

        vtk_type_by_numpy_type = {
            np.uint8: vtk.VTK_UNSIGNED_CHAR,
            np.uint16: vtk.VTK_UNSIGNED_SHORT,
            np.uint32: vtk.VTK_UNSIGNED_INT,
            np.uint64: vtk.VTK_UNSIGNED_LONG if vtk.VTK_SIZEOF_LONG == 64 else vtk.VTK_UNSIGNED_LONG_LONG,
            np.int8: vtk.VTK_CHAR,
            np.int16: vtk.VTK_SHORT,
            np.int32: vtk.VTK_INT,
            np.int64: vtk.VTK_LONG if vtk.VTK_SIZEOF_LONG == 64 else vtk.VTK_LONG_LONG,
            np.float32: vtk.VTK_FLOAT,
            np.float64: vtk.VTK_DOUBLE
        }
        vtk_datatype = vtk_type_by_numpy_type[source_numpy_array.dtype.type]

        source_numpy_array = np.flipud(source_numpy_array)

        # Note: don't flip (take out next two lines) if input is RGB.
        # Likewise, BGRA->RGBA would require a different reordering here.
        # if channel_count > 1:
        #     source_numpy_array = np.flip(source_numpy_array, 2)

        depth_array = numpy_support.numpy_to_vtk(source_numpy_array.ravel(), deep=True, array_type=vtk_datatype)
        depth_array.SetNumberOfComponents(channel_count)
        output_vtk_image.SetSpacing([1, 1, 1])
        output_vtk_image.SetOrigin([-1, -1, -1])
        output_vtk_image.GetPointData().SetScalars(depth_array)

        output_vtk_image.Modified()
        return output_vtk_image

    def _energy_histogram(self, listMode=None):

        # double        red[3] = {1, 0, 0};
        #
        if self.barchart is not None:
            self.view.GetScene().RemoveItem(self.barchart)
        if listMode is None:
            listMode= np.ones((100,2))
        energy = np.zeros((len(listMode[:,0])*2))
        energy[0:len(listMode[:,0])] = listMode[:,0]
        energy[len(listMode[:,0]):] = listMode[:,1]

        # NumPy_data=np.random.randint(0,100,size=(256))
        b= np.histogram(energy,100,(1,1001))
        number_of_bins = len(b[0])

        # self.renderer = vtk.vtkRende8rer()
        # self.vtkWidget_window_selection.GetRenderWindow().AddRenderer(self.renderer)


        barChart = vtk.vtkChartXY()
        xAxis = barChart.GetAxis(vtk.vtkAxis.BOTTOM)
        xAxis.SetTitle("Energy keV")
        xAxis.GetTitleProperties().SetColor(1, 1, 1)
        # xAxis.GetTitleProperties().SetFontSize(16)
        # xAxis.GetTitleProperties().ItalicOn()
        xAxis.GetLabelProperties().SetColor(1, 1, 1)
        xAxis.GetPen().SetColor(255, 255, 255, 255)
        # xAxis.SetTicksVisible(False)
        xAxis.SetGridVisible(False)
        xAxis.SetRangeLabelsVisible(True)
        xAxis.SetRange(0,1000)
        # xAxis.SetOffset(0)
        #
        # xAxis.SetUnscaledMinimum(200)

        # barChart = vtk.vtkChartXY()
        yAxis = barChart.GetAxis(vtk.vtkAxis.LEFT)
        yAxis.SetTitle("Counts")
        yAxis.GetTitleProperties().SetColor(1, 1, 1)
        # xAxis.GetTitleProperties().SetFontSize(16)
        # xAxis.GetTitleProperties().ItalicOn()
        yAxis.GetLabelProperties().SetColor(1, 1, 1)
        yAxis.GetPen().SetColor(255, 255, 255, 255)
        # xAxis.SetTicksVisible(False)
        yAxis.SetGridVisible(False)
        # vtkAxis * yAxis = chart->GetAxis(vtkAxis::LEFT);
        # yAxis->SetTitle("Circulation");
        # yAxis->GetTitleProperties()->SetColor(axisColor.GetData());
        # yAxis->GetTitleProperties()->SetFontSize(16);
        # yAxis->GetTitleProperties()->ItalicOn();
        # yAxis->GetLabelProperties()->SetColor(axisColor.GetData());


        # renwin = self.vtkWidget_window_selection.GetRenderWindow()
        #

        self.barchart =barChart
        self.view.GetScene().AddItem(self.barchart)

        renderer = self.view.GetRenderer()
        # renderer= vtk.vtkRenderer()
        self.vtkWidget_window_selection.GetRenderWindow().AddRenderer(renderer)
        # renwin.AddRenderer(renderer)
        renderer.SetBackground([0] * 3)
        # view.SetInteractor(self.interactor_window_selection_hist)
        # self.view=view
        # renderer.SetViewport(-.3, 0, 0.20, 0.30)


        # # output = histogram.GetOutput().GetScalarPointer()
        # frequencies = vtk.vtkIntArray()
        # frequencies.SetNumberOfComponents(1)
        # frequencies.SetNumberOfTuples(number_of_bins)
        # for j in range(0,number_of_bins):
        #     frequencies.SetTuple1(j, b[0][j])


        table = vtk.vtkTable()
        arrX = vtk.vtkFloatArray()
        arrX.SetName('X Axis')
        #
        arrC = vtk.vtkIntArray()
        arrC.SetName('Cosine')
        #
        # arrS = vtk.vtkFloatArray()
        # arrS.SetName('Sine')
        #
        # arrT = vtk.vtkFloatArray()
        # arrT.SetName('Sine-Cosine')
        #
        table.AddColumn(arrX)
        table.AddColumn(arrC)
        table.SetNumberOfRows(number_of_bins)
        for j in range(0,number_of_bins):
            table.SetValue(j,0,np.float(b[1][j]))
            table.SetValue(j,1,np.int(b[0][j]))
        #
        # # dataObject = vtk.vtkDataObject()
        # # dataObject.GetFieldData().AddArray(frequencies)
        #
        line=barChart.AddPlot(vtk.vtkChart.BAR)
        line.SetInputData(table,0,1)
        line.SetColor(95, 186, 189, 235)

        # barChart = vtk.vtkBarChartActor()
        # barChart.SetInput(dataObject)
        # # barChart.SetTitle("Histogram")
        # barChart.SetYTitle("Counts")
        #
        # textLabel=barChart.GetLabelTextProperty()
        # textLabel.SetFontSize(24)
        # # axis = barChart.GetAxis(1)
        # # axis.SetTitle("Month")
        # # barChart.SetBarLabel(0,"Energy (keV)")
        # barChart.GetPositionCoordinate().SetValue(0.15, 0, 0.0)
        # barChart.GetPosition2Coordinate().SetValue(1, 1, 0)
        # barChart.GetProperty().SetColor(1, 1, 1)
        # barChart.GetLegendActor().SetNumberOfEntries(dataObject.GetFieldData().GetArray(0).GetNumberOfTuples())
        # # barChart.LegendVisibilityOff()
        # # barChart.LabelVisibilityOff()
        # count = 0
        # for j in range(0, number_of_bins):
        #     barChart.SetBarColor(j, (0.372,0.73,0.74))


        # Render window and interactor.
        # renderWindow = vtk.vtkRenderWindow()
        # renderWindow.SetWindowName('Demo')
        # renderer.AddActor(barChart)

        # renderer.GetScene().AddItem(barChart)
        self.vtkWidget_window_selection.GetRenderWindow().Render()
