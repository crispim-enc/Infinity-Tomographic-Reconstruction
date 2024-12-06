from src.ControlUI.Toolbar import MainToolBar
from src.ControlUI.Visualization import  PopulateMainWindowVTK
from src.ControlUI.Utils import OpenNewCaseStudy


class SignalsConnectionFromEasyPetWindow:
    def __init__(self):
        self.actiondockScanning.triggered.connect(lambda: MainToolBar.change_between_dock_stations(self))
        self.actionVisualization.triggered.connect(lambda: MainToolBar.change_between_dock_stations(self))
        self.actionDock_DataAnalysis.triggered.connect(lambda: MainToolBar.change_between_dock_stations(self))
        self.actionDock_Preferences.triggered.connect(lambda: MainToolBar.change_between_dock_stations(self))
        self.OpenStudyAction.triggered.connect(lambda: OpenNewCaseStudy.updateList(self))
        # Camera 3D buttons connection
        self.changecamera2left_toolbutton.clicked.connect(lambda: PopulateMainWindowVTK.camera_rotation(self))
        self.changecamera2right_toolbutton.clicked.connect(lambda: PopulateMainWindowVTK.camera_rotation(self))
        self.changecamera2bottom_toolbutton.clicked.connect(lambda: PopulateMainWindowVTK.camera_rotation(self))
        self.changecamera2top_toolbutton.clicked.connect(lambda: PopulateMainWindowVTK.camera_rotation(self))
        self.changecamera2front_toolbutton.clicked.connect(lambda: PopulateMainWindowVTK.camera_rotation(self))
        self.changecamera2back_toolbutton.clicked.connect(lambda: PopulateMainWindowVTK.camera_rotation(self))
        self.resetcamera_toolButton.clicked.connect(lambda: PopulateMainWindowVTK.camera_initial_position(self))
        # self.iterative_reconstruct_push_button.clicked.connect(lambda: ReconstructOpenfile.__init__(self, self.list_open_studies))

        # self.iterative_reconstruct_push_button_easypet_data_original.clicked.connect(lambda: TabReconstruction.updateVisible_buttons(self))
        # self.iterative_reconstruct_push_button_easypet_data.clicked.connect(lambda:TabReconstruction.updateVisible_buttons(self))
        # self.iterative_reconstruct_combobox_algorithm_type.currentIndexChanged.connect(lambda: TabReconstruction.updateVisible_algorithm_options(self))

        # self.dynamic_view_dockWidget.dockLocationChanged.connect(
        #     lambda: PopulateDynamicViewWindowVTK._create_dynamic_view(self, 10))
        print('Singals Connection')
