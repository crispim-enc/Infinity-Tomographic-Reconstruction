import datetime
import os
import sys
from PyQt5 import QtCore, QtWidgets
import pandas as pd
from src.ControlUI.UiFiles.preclinic_mainwindow import Ui_MainWindow
from multiprocessing import Event
from src.EasyPETLinkInitializer.Preprocessing import PrepareEasyPETdata
from src.EasyPETLinkInitializer import ReconstructionInitializer
from ReconstructionManager import TabReconstruction, EnergyWindowVTKWidget
from src.ControlUI.Toolbar.programaticaly_widgets import MainToolBar, Objects3DToolBar
from src.ControlUI.Signals import SignalsConnectionFromEasyPetWindow
from src.ControlUI.Visualization import MouseObserverMainWindow, PopulateMainWindowVTK
from src.ControlUI.Utils import OpenNewCaseStudy, CustomTableModel
from src.ControlUI.SegmentationManager import SinogramVtk

directory = os.path.dirname(os.path.abspath(__file__))
time_stamp_debug_error = datetime.datetime.now()
time_stamp_debug_error = time_stamp_debug_error.strftime('%d %b %Y - %Hh %Mm %Ss')
# logging.basicConfig(
#    level=logging.DEBUG,
#    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
#    filename=directory+"//debug_errors//logfile " +time_stamp_debug_error+".log",
# )
# logging.debug('This message should go to the log file') # teste to see if logging is working


class InfiniToRWindow(Ui_MainWindow):
    def __init__(self):
        Ui_MainWindow.__init__(self)
        mainwindow = InterfaceOverride()
        self.setupUi(mainwindow)
        self.mainwindow = mainwindow
        # self.window_t = window_t
        # StatusBar.__init__(self)
        TabReconstruction.__init__(self)
        MainToolBar.__init__(self)
        # Objects3DToolBar.__init__(self)
        # HiddenUnfinishFeatures.__init__(self)
        OpenNewCaseStudy.__init__(self)
        SignalsConnectionFromEasyPetWindow.__init__(self)
        self.mouseObserverMainWindow = MouseObserverMainWindow(parent=self)
        self.populateMainWindowVTK = PopulateMainWindowVTK(parent=self)
        # # # PopulateDynamicViewWindowVTK.__init__(self)
        # #
        # # SinogramVtk.__init__(self)
        # #
        # self.mainwindow.vtkWidget_window_selection(self.vtkWidget_sinogram)
        # self.mainwindow.vtkWidget(self.populateMainWindowVTK.vtkWidget)
        # EnergyWindowVTKWidget.__init__(self)

        # ReconstructOpenfile.__init__(self)
        # self.list_open_studies
        self.gridlayout_video_frame = QtWidgets.QGridLayout()

        # l = QtGui.QVBoxLayout(self)
        # cdf = self.get_data_frame()
        # self._tm = CustomTableModel(cdf)
        # self._tm.update(cdf)
        # self._tv = self.tableView_info
        # self.tableView_info.setModel(self._tm)
        # for row in range(0, self._tm.rowCount()):
        #     self.tableView_info.openPersistentEditor(self._tm.index(row, 0))
        # l.addWidget(self._tv)
        # self.video_frame.addLayout(self.gridlayout_video_frame, 50, 50, 1, 5)
        # self.capture = QtCapture(self.video_frame, 0)
        # self.capture.start()
        # Rearranje Dockwidgets  to layout default
        # window_t.tabifyDockWidget(self.dockWidget_Scanning, self.dockWidget_preferences)
        # window_t.tabifyDockWidget(self.dockWidget_preferences,self.dockWidget_Database_admin)
        # window_t.tabifyDockWidget(self.dockWidget_analysis, self.volume_dockWidget)

        mainwindow.tabifyDockWidget(self.energy_window_dockwidget, self.volume_dockWidget)
        mainwindow.tabifyDockWidget(self.volume_dockWidget, self.dockWidget_analysis)
        mainwindow.tabifyDockWidget(self.dockWidget_analysis, self.volume_dockWidget)

        # window_t.resizeDocks({self.volume_dockWidget, self.dockWidget_live_tracer}, {1000, 400}, QtCore.Qt.Horizontal)

        self.list_open_studies = []
        self.color_map_3D_views = ['hot', 'afmhot', 'gist_heat', 'copper', 'jet', 'cool', 'gray', 'gist_gray', 'ocean',
                                   'gist_earth',
                                   'terrain', 'gist_stern',
                                   'gnuplot', 'gnuplot2', 'viridis', 'plasma', 'inferno', 'hsv',
                                   'gist_rainbow', 'rainbow', 'nipy_spectral', 'gist_ncar']
    def get_data_frame(self):
        df = pd.DataFrame({'Name': ['a', 'b', 'c', 'd'],
                           'Values': [2.3, 5.4, 3.1, 7.7], 'Last': [23.4, 11.2, 65.3, 88.8], 'Class': [1, 1, 2, 1],
                           'Valid': [True, True, True, False]})
        return df
        # self.gridlayout_video_frame.addWidget(self.turn_on_camera_toolbutton_2, 0, 0, 1, 1)


class InterfaceOverride(QtWidgets.QMainWindow):
    # def __init__(self, parent=None):
    #     # super().__init__(parent)
    #     self.parent = parent

    def closeEvent(self, event):
        self._vtkWidget_window_selection.Finalize()
        self._vtkWidget.Finalize()
        event.accept()

    def vtkWidget_window_selection(self, vtk_widget):
        self._vtkWidget_window_selection = vtk_widget

    def vtkWidget(self, vtk_widget):
        self._vtkWidget = vtk_widget


class ReconstructOpenfile:
    def __init__(self, list_open_studies=None):
        print(list_open_studies)

        # self.number_of_subsets = 1
        # self.voxel_size = 0.3
        # self.detector_sensibility_correction = True
        # self.detector_sensibility_data = "Simulation"
        # self.attenuation_correction = True
        # self.attenuation_data = "External"
        # self.decay_correction = False
        # self.positron_range_correction = False
        # self.doi_correction = False
        # self.scatter_correction = False
        # self.random_correction = False
        # self.respiratory_movement_correction = False
        # self.heart_movement_correction = False
        crystals_geometry = [16, 2]

        # reconstruction_data_type = ".easypet"
        TabReconstruction._update_variables_tab(self)

        prepareEasypetdata = PrepareEasyPETdata(study_file=list_open_studies[0],
                                                reconstruction_data_type=self.data_choosen,
                                                energy_window=self.energy_window,
                                                remove_first_crystals_row=self.remove_first_row_bool,
                                                remove_last_crystal_row=self.remove_last_row_bool,
                                                remove_peripheral_crystal_on_resistive_chain=self.remove_peripheral_crystals,
                                                only_left_side_crystals=self.only_left_detectors_bool,
                                                only_right_side_crystals=self.only_right_detectors_bool,
                                                parameters_2D_cut=self.parameters_2D_cut, adc_cuts=[50, 50, 50, 50],
                                                threshold_ratio=self.ratio_threshold,
                                                top_correction_angle=self.top_correction_angle,
                                                cut_acc_ramp=self.top_acceleration_cut, swap_sideAtoB=True,
                                                crystal_geometry=crystals_geometry, save_validation_data=False,
                                                save_spectrum_file=False,
                                                remove_incomplete_turn=self.remove_incomplete_turn)

        algorithm_function = ReconstructionInitializer(Easypetdata=prepareEasypetdata, algorithm=self.algorithm,
                                                       crystals_geometry=crystals_geometry,
                                                       type_of_reconstruction=[self.reconstruct_whole_body,
                                                                               self.reconstruct_static,
                                                                               self.reconstruct_dynamic,
                                                                               self.reconstruct_gated],
                                                       number_of_iterations=self.number_of_iterations,
                                                       number_of_subsets=self.number_of_subsets,
                                                       pixelSizeXY=self.pixel_size,
                                                       pixelSizeXYZ=self.pixel_size,
                                                       fov=prepareEasypetdata.fov_real,
                                                       type_of_projector=self.geometric_projector,
                                                       recon2D=self.geometry_2D_bool,
                                                       number_of_neighbours=self.axial_neighbours,
                                                       study_path=os.path.dirname(list_open_studies[0]),
                                                       multiple_kernel=self.multiple_kernel_bool,
                                                       map_precedent=self.algorithm_pet_used_to_map,
                                                       beta=self.beta_value_penalized_algorithms,
                                                       local_median_v_number=self.mrp_kernel_size,
                                                       override_geometry_file=True,
                                                       detector_normalization_correction=False,
                                                       detector_normalization_data="Calculated",
                                                       detector_normalization_read_from_memory=False,
                                                       attenuation_correction=False,
                                                       attenuation_data="Calculated Uniform", decay_correction=True,
                                                       positron_range=False,
                                                       scatter_angle_correction=False, dead_time_correction=False,
                                                       random_correction=False, doi_correction=False,
                                                       respiratory_movement_correction=False,
                                                       heart_movement_correction=False, save_numpy=True,
                                                       save_raw_data=True)
        volume_static = algorithm_function.im

        PopulateMainWindowVTK._removeDataToCurrentView(self)
        PopulateMainWindowVTK.addNewDataToCurrentView(self, volume=volume_static)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # window_t = InterfaceOverride()
    # window_t=QtWidgets.QMainWindow()
    w = InfiniToRWindow()
    w.mainwindow.showMaximized()
    # window_t.show()
    # window_t.showMaximized()
    stop_event = Event()

    sys.exit(app.exec_())
