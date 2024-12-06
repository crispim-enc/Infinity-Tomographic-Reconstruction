from PyQt5 import QtCore, QtGui, QtWidgets
from src.ControlUI.UiFiles.radialbar import RadialBar


class StatusBar:
    def __init__(self):
        # STATUS TOOL BAR
        self.empty_space_toolbar_1 = QtWidgets.QLabel()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        self.empty_space_toolbar_1.setSizePolicy(sizePolicy)

        self.status_toolBar.addWidget(self.empty_space_toolbar_1)
        # self.status_toolBar.addWidget(self.status_label)
        # self.status_toolBar.addWidget(self.widget_progressbar_tool_bar)
        # self.status_toolBar.addWidget(self.label_status_bar_counts_string)
        # self.status_toolBar.addWidget(self.CorrectCounts_label)
        self.empty_space_toolbar_2 = QtWidgets.QLabel()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        self.empty_space_toolbar_2.setSizePolicy(sizePolicy)
        self.status_toolBar.addWidget(self.empty_space_toolbar_2)
        self.status_toolBar.addWidget(self.label_status_bar_database_string)
        self.status_toolBar.addWidget(self.connectionlabel_database_value)
        self.status_toolBar.addWidget(self.database_recconect_toolButton)
        self.status_toolBar.addWidget(self.label_status_bar_usb_connection_string)
        self.status_toolBar.addWidget(self.label_status_bar_usb_connection_value)
        self.status_toolBar.addWidget(self.usb_recconect_toolButton)
        # spacerItem4 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        # self.horizontalLayout_9.addItem(spacerItem4)
        # self.status_toolBar.addWidget(spacerItem4)

        self.status_toolBar.addWidget(self.label_status_bar_temperature_string)
        self.status_toolBar.addWidget(self.temperature_label)


class MainToolBar:
    def __init__(self):
        # Creates excllusivity between buttons of the maintoolBAr (Green one at left)
        self.exclusive_mainbar_buttons = QtWidgets.QActionGroup(self.main_toolBar)
        self.exclusive_mainbar_buttons.addAction(self.actionVisualization)
        self.exclusive_mainbar_buttons.addAction(self.actiondockScanning)
        self.exclusive_mainbar_buttons.addAction(self.actionDock_DataAnalysis)
        self.exclusive_mainbar_buttons.addAction(self.actionDock_Preferences)

    def change_between_dock_stations(self):
        sender = self.window_t.sender()
        list_docks = [self.actionVisualization, self.actiondockScanning, self.actionDock_DataAnalysis, self.actionDock_Preferences]
        dock_choosed = list_docks.index(sender)
        list_action = [self.volume_dockWidget,self.dockWidget_Scanning, self.dockWidget_Database_admin, self.dockWidget_preferences]
        list_action[dock_choosed].raise_()


class Objects3DToolBar:
    def __init__(self):
        icon_planes_3D = QtGui.QIcon()
        icon_planes_3D.addPixmap(QtGui.QPixmap("../img/planes.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)

        self.planes_active_3D = QtWidgets.QToolButton()
        self.planes_active_3D.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        self.planes_active_3D.setMenu(QtWidgets.QMenu())
        self.planes_active_3D.setIcon(icon_planes_3D)
        self.objects3D_toolBar.addWidget(self.planes_active_3D)
        # self.planes_active_3D.menu().toolButtonStyle(ToolButtonIconOnly)
        self.planes_active_3D.menu().addAction(self.actionaxial_plane_in_3D_volume)
        self.planes_active_3D.menu().setStyleSheet("QMenu\n"
                                                    "{\n"
                                                    " font-size: 16px ; \n"
                                                    "color:white; \n"
                                                    "icon-size: 40px 40px;"
                                                    " background-color:qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.946, fx:0.744, fy:0.159, stop:0 rgba(140, 150, 165, 255), stop:1 rgba(82, 98, 112, 255));\n"
                                                    "    \n"
                                                    "}\n"
                                                    "\n"
                                                    "\n"
                                                    "QMenu:hover\n"
                                                    "{\n"
                                                    "    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #565656, stop: 0.1 #525252, stop: 0.5 #4e4e4e, stop: 0.9 #4a4a4a, stop: 1 #464646);\n"
                                                    "    border-width: 0px 0px 3px 0px;\n"
                                                    "    border-color: rgb(42, 252, 253);\n"
                                                    "    border-style: solid;\n"
                                                    "    font-weight:bold;\n"
                                                    "}\n"
                                                    "\n")

        # self.Show_volume_toolButton.setDefaultAction(self.actionShowVolume)
        # self.Show_planes_toolButton.setDefaultAction(self.actionShowPlanes)
        # self.Show_axial_toolButton.setDefaultAction(self.actionaxial_plane_in_3D_volume)
        # self.Show_coronal_toolButton.setDefaultAction(self.actionShowCoronal)
        # self.Show_sagittal_toolButton.setDefaultAction(self.actionShowSagittal)
        # self.Show_axis_toolButton.setDefaultAction(self.actionShowOrientationWidget)
        # self.Show_lines_toolButton.setDefaultAction(self.actionShowLines)
        # self.Show_colorbar_toolButton.setDefaultAction(self.actionShowScalarBar)
        # self.Show_inside_bed_toolButton.setDefaultAction(self.actionShow_beds_inside)
        # self.cut_volume_tool_button.setDefaultAction(self.actionClipping_volume)
        # self.Show_easypetbox_toolbutton.setDefaultAction(self.actionShow_EasyPET_structure)
        # self.Show_phantom_human_tool_button.setDefaultAction(self.actionShow_human_phantom)


class VitalSignalsWidgets:
    def __init__(self):
        # self.widget_oximeter
        self.oximeter_radial_widget = RadialBar()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(10)
        self.oximeter_radial_widget.backgroundColor = QtGui.QColor("transparent")

        self.oximeter_radial_widget.foregroundColor = QtGui.QColor("#191a2f")
        self.oximeter_radial_widget.dialWidth = 15
        self.oximeter_radial_widget.spanAngle = 280
        self.oximeter_radial_widget.textColor = QtGui.QColor("black")
        self.oximeter_radial_widget.penStyle = QtCore.Qt.RoundCap
        # w.dialType = RadialBar.DialType.FullDial
        self.oximeter_radial_widget.dialType = RadialBar.DialType.MinToMax
        self.oximeter_radial_widget.suffixText = "%"
        self.oximeter_radial_widget.textFont.setBold(True)
        self.oximeter_radial_widget.textFont.setPointSize(20)
        self.oximeter_radial_widget.progressColor = QtGui.QColor("white")
        # self.widget_oximeter.setMinimumSize(QtCore.QSize(100, 300))
        self.lay_oximeter = QtWidgets.QVBoxLayout(self.widget_oximeter)
        self.lay_oximeter.setContentsMargins(0, 0, 0, 0)
        self.lay_oximeter.setAlignment(QtCore.Qt.AlignHCenter)
        self.lay_oximeter.addWidget(self.oximeter_radial_widget, 0)

        self.pulsation_widget = RadialBar()
        self.gridLayout_39.addWidget(self.pulsation_widget,2,0)
        # self.lay_oximeter.addWidget(self.oximeter_radial_widget, 0, 0)
        # self.oximeter_radial_widget.show()