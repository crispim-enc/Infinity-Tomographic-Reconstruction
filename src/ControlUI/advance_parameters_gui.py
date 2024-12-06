# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\github_easypet\easyPETtraining\EasyPET training versions\gui_client\advanced_parameters.ui'
#
# Created by: PyQt5 UI code generator 5.10
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_advanced_parameters_Dialog(object):
    def setupUi(self, advanced_parameters_Dialog):
        advanced_parameters_Dialog.setObjectName("advanced_parameters_Dialog")
        advanced_parameters_Dialog.resize(700, 670)
        advanced_parameters_Dialog.setStyleSheet("QGroupBox{\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                      stop: 0 #E0E0E0, stop: 1 #FFFFFF);\n"
"    border: 2px solid gray;\n"
"    border-radius: 5px;\n"
"    margin-top: 2ex; /* leave space at the top for the title */\n"
"   font: bold 16px;\n"
"\n"
"}\n"
"\n"
"QGroupBox::title {\n"
"    subcontrol-origin: margin;\n"
"    subcontrol-position: top center; /* position at the top center */\n"
"    padding: 0 5px;\n"
"    \n"
"    \n"
"}\n"
"\n"
"QLabel{\n"
" font: bold 11px;\n"
"}\n"
"\n"
"QLineEdit {\n"
"\n"
"qproperty-alignment: AlignCenter;\n"
"  height: 31px;\n"
" border:2px solid  rgb(95,186,188);\n"
" font: bold 11px;\n"
"}\n"
"QLine Edi:disabled{\n"
"background-color: rgb(227, 227, 227);\n"
"    color: rgb(79, 79, 79)\n"
"}\n"
"\n"
"QLineEdit:focus{\n"
"border:3px solid   rgb(95,186,188);\n"
"}\n"
"\n"
"\n"
"\n"
"QSpinBox:editable {\n"
"     background-color:qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.946, fx:0.744, fy:0.159, stop:0 rgba(140, 150, 165, 255), stop:1 rgba(82, 98, 112, 255));\n"
"font-weight:bold;\n"
"}\n"
"\n"
"QSpinBox {\n"
"\n"
"    border: 1px solid gray;\n"
"    border-radius: 3px;\n"
"    padding: 1px 0px 1px 3px;\n"
"    color:white;\n"
"    font:10px;\n"
"background-color:qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.946, fx:0.744, fy:0.159, stop:0 rgba(140, 150, 165, 255), stop:1 rgba(82, 98, 112, 255));\n"
"}\n"
"\n"
"QSpinBox::up-button {\n"
"    subcontrol-origin: border;\n"
"    subcontrol-position: right; /* position at the top right corner */\n"
"\n"
"    width: 60px; /* 16 + 2*1px border-width = 15px padding + 3px parent border */\n"
"    height: 20px;\n"
"    border-color:rgb(172, 172, 172);;\n"
"    background-color: rgb(172, 172, 172);\n"
"   \n"
"    border-width: 2px;\n"
"}\n"
"\n"
"QSpinBox::up-arrow {\n"
"    border-image: url(Resources/plus - white.png) 1;\n"
"    width: 16px;\n"
"    height: 16px;\n"
"}\n"
"\n"
"QSpinBox::up-button:hover {\n"
"    \n"
"    border-color:rgb(190, 190, 190);\n"
"    background-color:rgb(190, 190, 190);\n"
"   \n"
"  \n"
"}\n"
"\n"
"QSpinBox::up-button:pressed {\n"
"    \n"
"    border-color:rgb(160, 160, 160);\n"
"    background-color:rgb(160, 160, 160);   \n"
"  \n"
"}\n"
"\n"
"QSpinBox::down-button {\n"
"    subcontrol-origin: border;\n"
"    subcontrol-position: left; /* position at bottom right corner */\n"
"\n"
"  \n"
"    width: 60px; /* 16 + 2*1px border-width = 15px padding + 3px parent border */\n"
"    height: 20px;\n"
"    border-color:rgb(172, 172, 172);;\n"
"    background-color: rgb(172, 172, 172);\n"
"   \n"
"    border-width: 2px;\n"
"}\n"
"\n"
"QSpinBox::down-button:hover {    \n"
"    border-color:rgb(190, 190, 190);\n"
"    background-color:rgb(190, 190, 190);   \n"
"\n"
"}\n"
"\n"
"QSpinBox::down-button:pressed {\n"
"    \n"
"    border-color:rgb(160, 160, 160);\n"
"    background-color:rgb(160, 160, 160);\n"
"     \n"
"}\n"
"\n"
"QSpinBox::down-arrow {\n"
"    border-image: url(Resources/minus.png) 1;\n"
"    width: 16px;\n"
"    height: 2px;\n"
"}\n"
"\n"
"QSpinBox::disabled {\n"
"    \n"
"background-color:rgb(216, 216, 216);\n"
"font-weight:bold;\n"
"}\n"
"\n"
"\n"
"QDoubleSpinBox:editable {\n"
"     background-color:qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.946, fx:0.744, fy:0.159, stop:0 rgba(140, 150, 165, 255), stop:1 rgba(82, 98, 112, 255));\n"
"font-weight:bold;\n"
"}\n"
"\n"
"QDoubleSpinBox {\n"
"\n"
"    border: 1px solid gray;\n"
"    border-radius: 3px;\n"
"    padding: 1px 0px 1px 3px;\n"
"    color:white;\n"
"    font:11px;\n"
"    font-weight:bold;\n"
"qproperty-alignment: AlignCenter;\n"
"background-color:qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.946, fx:0.744, fy:0.159, stop:0 rgba(140, 150, 165, 255), stop:1 rgba(82, 98, 112, 255));\n"
"}\n"
"\n"
"QDoubleSpinBox::up-button {\n"
"    subcontrol-origin: border;\n"
"    subcontrol-position: right; /* position at the top right corner */\n"
"\n"
"    width: 40px; /* 16 + 2*1px border-width = 15px padding + 3px parent border */\n"
"    height: 20px;\n"
"    border-color:rgb(172, 172, 172);;\n"
"    background-color: rgb(172, 172, 172);\n"
"   \n"
"    border-width: 2px;\n"
"}\n"
"\n"
"QDoubleSpinBox::up-arrow {\n"
"    border-image: url(Resources/plus - white.png) 1;\n"
"    width: 16px;\n"
"    height: 16px;\n"
"}\n"
"\n"
"QDoubleSpinBox::up-button:hover {\n"
"    \n"
"    border-color:rgb(190, 190, 190);\n"
"    background-color:rgb(190, 190, 190);\n"
"   \n"
"  \n"
"}\n"
"\n"
"QDoubleSpinBox::up-button:pressed {\n"
"    \n"
"    border-color:rgb(160, 160, 160);\n"
"    background-color:rgb(160, 160, 160);\n"
"   \n"
"  \n"
"}\n"
"\n"
"QDoubleSpinBox::down-button {\n"
"    subcontrol-origin: border;\n"
"    subcontrol-position: left; /* position at bottom right corner */\n"
"\n"
"  \n"
"    width: 40px; /* 16 + 2*1px border-width = 15px padding + 3px parent border */\n"
"    height: 20px;\n"
"    border-color:rgb(172, 172, 172);;\n"
"    background-color: rgb(172, 172, 172);\n"
"   \n"
"    border-width: 2px;\n"
"}\n"
"\n"
"QDoubleSpinBox::down-button:hover {\n"
"    \n"
"    border-color:rgb(190, 190, 190);\n"
"    background-color:rgb(190, 190, 190);\n"
"   \n"
"  \n"
"}\n"
"\n"
"QDoubleSpinBox::down-button:pressed {\n"
"    \n"
"    border-color:rgb(160, 160, 160);\n"
"    background-color:rgb(160, 160, 160);\n"
"   \n"
"  \n"
"}\n"
"\n"
"QDoubleSpinBox::down-arrow {\n"
"    border-image: url(Resources/minus.png) 1;\n"
"    width: 16px;\n"
"    height: 2px;\n"
"}\n"
"\n"
"\n"
"QDoubleSpinBox::disabled {\n"
"    \n"
"background-color:rgb(216, 216, 216);\n"
"font-weight:bold;\n"
"}\n"
"\n"
"QComboBox {\n"
"    border: 1px solid gray;\n"
"    border-radius: 3px;\n"
"    padding: 1px 0px 1px 3px;\n"
"    color:white;\n"
"    font:14px;\n"
"background-color:qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.946, fx:0.744, fy:0.159, stop:0 rgba(140, 150, 165, 255), stop:1 rgba(82, 98, 112, 255));\n"
"font-weight:bold;\n"
"\n"
"}\n"
"\n"
"QComboBox::disabled {\n"
"    \n"
"background-color:rgb(216, 216, 216);\n"
"font-weight:bold;\n"
" border: 0px solid gray;\n"
"}\n"
"\n"
"/* \n"
"QComboBox::drop-down {\n"
"     background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                 stop: 0 #E1E1E1, stop: 0.4 #DDDDDD,\n"
"                                 stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3);\n"
"}\n"
"\n"
"QComboBox gets the \"on\" state when the popup is open \n"
"\n"
"\n"
"QComboBox:on { /* shift the text when the popup opens \n"
"    padding-top: 3px;\n"
"    padding-left: 4px;\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                stop: 0 #D3D3D3, stop: 0.4 #D8D8D8,\n"
"                                stop: 0.5 #DDDDDD, stop: 1.0 #E1E1E1);\n"
"}\n"
"*/\n"
" QComboBox::drop-down:on {\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                stop: 0 #D3D3D3, stop: 0.4 #D8D8D8,\n"
"                                stop: 0.5 #DDDDDD, stop: 1.0 #E1E1E1);\n"
"}\n"
"\n"
"QComboBox::drop-down {\n"
"    subcontrol-origin: padding;\n"
"    subcontrol-position: top right;\n"
"    width: 30px;  \n"
"    font-weight:normal;\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                 stop: 0 #E1E1E1, stop: 0.4 #DDDDDD,\n"
"                                 stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3);    \n"
"\n"
"}\n"
"\n"
"QComboBox::drop-down:disabled {\n"
"    \n"
"background-color:rgb(216, 216, 216);\n"
"font-weight:bold;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"    image: url(Resources/if_angle-down_1608507.png);\n"
"    width:30px;\n"
"height:35px;\n"
"    background-color:qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.946, fx:0.744, fy:0.159, stop:0 rgba(140, 150, 165, 255), stop:1 rgba(82, 98, 112, 255));\n"
"    \n"
"}\n"
"\n"
"QComboBox::down-arrow:disabled {\n"
"    \n"
"background-color:rgb(216, 216, 216);\n"
"font-weight:bold;\n"
"\n"
"}\n"
"\n"
"QComboBox::down-arrow:on { /* shift the arrow when popup is open */\n"
"    top: 1px;\n"
"    left: 1px;\n"
"}\n"
"\n"
"\n"
"QComboBox QAbstractItemView {\n"
"    border: 2px solid darkgray;\n"
"    selection-background-color: rgb(95, 186, 189,235);\n"
"    background-color:lightgray;\n"
"    font-weight: bold;\n"
"}\n"
"")
        self.gridLayout = QtWidgets.QGridLayout(advanced_parameters_Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.apply_pushButton = QtWidgets.QPushButton(advanced_parameters_Dialog)
        self.apply_pushButton.setStyleSheet("QPushButton {/* background-color:  rgb(55, 97, 102); border: none ; color: rgb(255, 210, 73)\n"
"*/}\n"
"QPushButton\n"
"{\n"
"    color: rgb(240, 240, 240);   \n"
" background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.946, fx:0.744, fy:0.159, stop:0 rgba(140, 150, 165, 255), stop:1 rgba(82, 98, 112, 255));\n"
"    border-width: 2px;\n"
"    border-color: #1e1e1e;\n"
"    border-style: solid;\n"
"    border-radius: 8px;\n"
"    padding: 3px;\n"
"    font-size: 16px;\n"
"    padding-left: 5px;\n"
"    padding-right: 5px;\n"
"}\n"
"\n"
"\n"
"\n"
"QPushButton:hover\n"
"{\n"
"    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #565656, stop: 0.1 #525252, stop: 0.5 #4e4e4e, stop: 0.9 #4a4a4a, stop: 1 #464646);\n"
"    border-width: 3px;\n"
"    border-color: rgb(42, 252, 253);\n"
"    border-style: solid;\n"
"    font-weight:bold;\n"
"}\n"
"\n"
"QPushButton:open\n"
"{\n"
"    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #2d2d2d, stop: 0.1 #2b2b2b, stop: 0.5 #292929, stop: 0.9 #282828, stop: 1 #252525);\n"
"    border-width: 3px;\n"
"    border-color: rgb(42, 252, 253);\n"
"    border-style: solid;\n"
"    font-weight:bold;\n"
"}")
        self.apply_pushButton.setObjectName("apply_pushButton")
        self.gridLayout.addWidget(self.apply_pushButton, 3, 4, 1, 1)
        self.ok_pushButton = QtWidgets.QPushButton(advanced_parameters_Dialog)
        self.ok_pushButton.setStyleSheet("QPushButton {/* background-color:  rgb(55, 97, 102); border: none ; color: rgb(255, 210, 73)\n"
"*/}\n"
"QPushButton\n"
"{\n"
"    color: rgb(240, 240, 240);   \n"
" background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.946, fx:0.744, fy:0.159, stop:0 rgba(140, 150, 165, 255), stop:1 rgba(82, 98, 112, 255));\n"
"    border-width: 2px;\n"
"    border-color: #1e1e1e;\n"
"    border-style: solid;\n"
"    border-radius: 8px;\n"
"    padding: 3px;\n"
"    font-size: 16px;\n"
"    padding-left: 5px;\n"
"    padding-right: 5px;\n"
"}\n"
"\n"
"\n"
"\n"
"QPushButton:hover\n"
"{\n"
"    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #565656, stop: 0.1 #525252, stop: 0.5 #4e4e4e, stop: 0.9 #4a4a4a, stop: 1 #464646);\n"
"    border-width: 3px;\n"
"    border-color: rgb(42, 252, 253);\n"
"    border-style: solid;\n"
"    font-weight:bold;\n"
"}\n"
"\n"
"QPushButton:open\n"
"{\n"
"    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #2d2d2d, stop: 0.1 #2b2b2b, stop: 0.5 #292929, stop: 0.9 #282828, stop: 1 #252525);\n"
"    border-width: 3px;\n"
"    border-color: rgb(42, 252, 253);\n"
"    border-style: solid;\n"
"    font-weight:bold;\n"
"}")
        self.ok_pushButton.setObjectName("ok_pushButton")
        self.gridLayout.addWidget(self.ok_pushButton, 3, 1, 1, 1)
        self.default_values_pushButton = QtWidgets.QPushButton(advanced_parameters_Dialog)
        self.default_values_pushButton.setStyleSheet("QPushButton {/* background-color:  rgb(55, 97, 102); border: none ; color: rgb(255, 210, 73)\n"
"*/}\n"
"QPushButton\n"
"{\n"
"    color: rgb(240, 240, 240);   \n"
" background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.946, fx:0.744, fy:0.159, stop:0 rgba(140, 150, 165, 255), stop:1 rgba(82, 98, 112, 255));\n"
"    border-width: 2px;\n"
"    border-color: #1e1e1e;\n"
"    border-style: solid;\n"
"    border-radius: 8px;\n"
"    padding: 3px;\n"
"    font-size: 16px;\n"
"    padding-left: 5px;\n"
"    padding-right: 5px;\n"
"}\n"
"\n"
"\n"
"\n"
"QPushButton:hover\n"
"{\n"
"    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #565656, stop: 0.1 #525252, stop: 0.5 #4e4e4e, stop: 0.9 #4a4a4a, stop: 1 #464646);\n"
"    border-width: 3px;\n"
"    border-color: rgb(42, 252, 253);\n"
"    border-style: solid;\n"
"    font-weight:bold;\n"
"}\n"
"\n"
"QPushButton:open\n"
"{\n"
"    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #2d2d2d, stop: 0.1 #2b2b2b, stop: 0.5 #292929, stop: 0.9 #282828, stop: 1 #252525);\n"
"    border-width: 3px;\n"
"    border-color: rgb(42, 252, 253);\n"
"    border-style: solid;\n"
"    font-weight:bold;\n"
"}")
        self.default_values_pushButton.setObjectName("default_values_pushButton")
        self.gridLayout.addWidget(self.default_values_pushButton, 3, 2, 1, 1)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.fast_acquisition_pushButton = QtWidgets.QPushButton(advanced_parameters_Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fast_acquisition_pushButton.sizePolicy().hasHeightForWidth())
        self.fast_acquisition_pushButton.setSizePolicy(sizePolicy)
        self.fast_acquisition_pushButton.setMinimumSize(QtCore.QSize(80, 30))
        self.fast_acquisition_pushButton.setMaximumSize(QtCore.QSize(16777215, 100))
        font = QtGui.QFont()
        font.setPointSize(-1)
        self.fast_acquisition_pushButton.setFont(font)
        self.fast_acquisition_pushButton.setStyleSheet("QPushButton {/* background-color:  rgb(55, 97, 102); border: none ; color: rgb(255, 210, 73)\n"
"*/}\n"
"QPushButton\n"
"{\n"
"    color: rgb(240, 240, 240);   \n"
" background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.946, fx:0.744, fy:0.159, stop:0 rgba(140, 150, 165, 255), stop:1 rgba(82, 98, 112, 255));\n"
"    border-width: 2px;\n"
"    border-color: #1e1e1e;\n"
"    border-style: solid;\n"
"    border-radius: 8px;\n"
"    padding: 3px;\n"
"    font-size: 16px;\n"
"    padding-left: 5px;\n"
"    padding-right: 5px;\n"
"}\n"
"\n"
"\n"
"\n"
"QPushButton:hover\n"
"{\n"
"    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #565656, stop: 0.1 #525252, stop: 0.5 #4e4e4e, stop: 0.9 #4a4a4a, stop: 1 #464646);\n"
"    border-width: 3px;\n"
"    border-color: rgb(42, 252, 253);\n"
"    border-style: solid;\n"
"    font-weight:bold;\n"
"}\n"
"\n"
"QPushButton:open\n"
"{\n"
"    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #2d2d2d, stop: 0.1 #2b2b2b, stop: 0.5 #292929, stop: 0.9 #282828, stop: 1 #252525);\n"
"    border-width: 3px;\n"
"    border-color: rgb(42, 252, 253);\n"
"    border-style: solid;\n"
"    font-weight:bold;\n"
"}")
        self.fast_acquisition_pushButton.setCheckable(True)
        self.fast_acquisition_pushButton.setChecked(True)
        self.fast_acquisition_pushButton.setAutoExclusive(True)
        self.fast_acquisition_pushButton.setObjectName("fast_acquisition_pushButton")
        self.horizontalLayout_4.addWidget(self.fast_acquisition_pushButton)
        self.medium_acquisition_pushButton = QtWidgets.QPushButton(advanced_parameters_Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.medium_acquisition_pushButton.sizePolicy().hasHeightForWidth())
        self.medium_acquisition_pushButton.setSizePolicy(sizePolicy)
        self.medium_acquisition_pushButton.setMinimumSize(QtCore.QSize(80, 30))
        self.medium_acquisition_pushButton.setMaximumSize(QtCore.QSize(16777215, 100))
        font = QtGui.QFont()
        font.setPointSize(-1)
        self.medium_acquisition_pushButton.setFont(font)
        self.medium_acquisition_pushButton.setStyleSheet("QPushButton {/* background-color:  rgb(55, 97, 102); border: none ; color: rgb(255, 210, 73)\n"
"*/}\n"
"QPushButton\n"
"{\n"
"    color: rgb(240, 240, 240);   \n"
" background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.946, fx:0.744, fy:0.159, stop:0 rgba(140, 150, 165, 255), stop:1 rgba(82, 98, 112, 255));\n"
"    border-width: 2px;\n"
"    border-color: #1e1e1e;\n"
"    border-style: solid;\n"
"    border-radius: 8px;\n"
"    padding: 3px;\n"
"    font-size: 16px;\n"
"    padding-left: 5px;\n"
"    padding-right: 5px;\n"
"}\n"
"\n"
"\n"
"\n"
"QPushButton:hover\n"
"{\n"
"    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #565656, stop: 0.1 #525252, stop: 0.5 #4e4e4e, stop: 0.9 #4a4a4a, stop: 1 #464646);\n"
"    border-width: 3px;\n"
"    border-color: rgb(42, 252, 253);\n"
"    border-style: solid;\n"
"    font-weight:bold;\n"
"}\n"
"\n"
"QPushButton:open\n"
"{\n"
"    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #2d2d2d, stop: 0.1 #2b2b2b, stop: 0.5 #292929, stop: 0.9 #282828, stop: 1 #252525);\n"
"    border-width: 3px;\n"
"    border-color: rgb(42, 252, 253);\n"
"    border-style: solid;\n"
"    font-weight:bold;\n"
"}")
        self.medium_acquisition_pushButton.setCheckable(True)
        self.medium_acquisition_pushButton.setAutoExclusive(True)
        self.medium_acquisition_pushButton.setObjectName("medium_acquisition_pushButton")
        self.horizontalLayout_4.addWidget(self.medium_acquisition_pushButton)
        self.slow_acquisition_pushButton = QtWidgets.QPushButton(advanced_parameters_Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.slow_acquisition_pushButton.sizePolicy().hasHeightForWidth())
        self.slow_acquisition_pushButton.setSizePolicy(sizePolicy)
        self.slow_acquisition_pushButton.setMinimumSize(QtCore.QSize(80, 30))
        self.slow_acquisition_pushButton.setMaximumSize(QtCore.QSize(16777215, 100))
        font = QtGui.QFont()
        font.setPointSize(-1)
        self.slow_acquisition_pushButton.setFont(font)
        self.slow_acquisition_pushButton.setStyleSheet("QPushButton {/* background-color:  rgb(55, 97, 102); border: none ; color: rgb(255, 210, 73)\n"
"*/}\n"
"QPushButton\n"
"{\n"
"    color: rgb(240, 240, 240);   \n"
" background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.946, fx:0.744, fy:0.159, stop:0 rgba(140, 150, 165, 255), stop:1 rgba(82, 98, 112, 255));\n"
"    border-width: 2px;\n"
"    border-color: #1e1e1e;\n"
"    border-style: solid;\n"
"    border-radius: 8px;\n"
"    padding: 3px;\n"
"    font-size: 16px;\n"
"    padding-left: 5px;\n"
"    padding-right: 5px;\n"
"}\n"
"\n"
"\n"
"\n"
"QPushButton:hover\n"
"{\n"
"    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #565656, stop: 0.1 #525252, stop: 0.5 #4e4e4e, stop: 0.9 #4a4a4a, stop: 1 #464646);\n"
"    border-width: 3px;\n"
"    border-color: rgb(42, 252, 253);\n"
"    border-style: solid;\n"
"    font-weight:bold;\n"
"}\n"
"\n"
"QPushButton:open\n"
"{\n"
"    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #2d2d2d, stop: 0.1 #2b2b2b, stop: 0.5 #292929, stop: 0.9 #282828, stop: 1 #252525);\n"
"    border-width: 3px;\n"
"    border-color: rgb(42, 252, 253);\n"
"    border-style: solid;\n"
"    font-weight:bold;\n"
"}")
        self.slow_acquisition_pushButton.setCheckable(True)
        self.slow_acquisition_pushButton.setChecked(False)
        self.slow_acquisition_pushButton.setAutoExclusive(True)
        self.slow_acquisition_pushButton.setObjectName("slow_acquisition_pushButton")
        self.horizontalLayout_4.addWidget(self.slow_acquisition_pushButton)
        self.pushButton = QtWidgets.QPushButton(advanced_parameters_Dialog)
        self.pushButton.setStyleSheet("QPushButton {/* background-color:  rgb(55, 97, 102); border: none ; color: rgb(255, 210, 73)\n"
"*/}\n"
"QPushButton\n"
"{\n"
"    color: rgb(240, 240, 240);   \n"
" background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.946, fx:0.744, fy:0.159, stop:0 rgba(140, 150, 165, 255), stop:1 rgba(82, 98, 112, 255));\n"
"    border-width: 2px;\n"
"    border-color: #1e1e1e;\n"
"    border-style: solid;\n"
"    border-radius: 8px;\n"
"    padding: 3px;\n"
"    font-size: 16px;\n"
"    padding-left: 5px;\n"
"    padding-right: 5px;\n"
"}\n"
"\n"
"\n"
"\n"
"QPushButton:hover\n"
"{\n"
"    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #565656, stop: 0.1 #525252, stop: 0.5 #4e4e4e, stop: 0.9 #4a4a4a, stop: 1 #464646);\n"
"    border-width: 3px;\n"
"    border-color: rgb(42, 252, 253);\n"
"    border-style: solid;\n"
"    font-weight:bold;\n"
"}\n"
"\n"
"QPushButton:open\n"
"{\n"
"    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #2d2d2d, stop: 0.1 #2b2b2b, stop: 0.5 #292929, stop: 0.9 #282828, stop: 1 #252525);\n"
"    border-width: 3px;\n"
"    border-color: rgb(42, 252, 253);\n"
"    border-style: solid;\n"
"    font-weight:bold;\n"
"}")
        self.pushButton.setAutoExclusive(True)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_4.addWidget(self.pushButton)
        self.gridLayout.addLayout(self.horizontalLayout_4, 0, 0, 1, 5)
        self.tabWidget = QtWidgets.QTabWidget(advanced_parameters_Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.tabWidget.setStyleSheet("QTabbar\n"
" {\n"
"}\n"
"QTabBar::tab\n"
"{\n"
"    background:qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.946, fx:0.744, fy:0.159, stop:0 rgba(140, 150, 165, 255), stop:1 rgba(82, 98, 112, 255));\n"
"     color:rgb(240, 240, 240);\n"
"font: 12px;\n"
"font-weight: bold;\n"
"height: 40ex;\n"
"    /*padding: 12px 5px 12px 4px;\n"
"    */\n"
"\n"
"}\n"
"\n"
"\n"
"\n"
"QTabBar::tab:selected\n"
"{    \n"
"    background:  rgba(76, 151, 152, 255);\n"
"    /*padding: 12px 4px 12px 5px;\n"
"    width: 40ex;*/\n"
"    \n"
"}\n"
"\n"
"QTabWidget::pane { \n"
"    border-right: 3px solid rgba(76, 151, 152, 255);\n"
"    \n"
"}\n"
"/*\n"
"QTabBar::tab:!selected {\n"
"    margin-top: 2px; make non-selected tabs look smaller\n"
"}*/\n"
" ")
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.East)
        self.tabWidget.setIconSize(QtCore.QSize(28, 28))
        self.tabWidget.setDocumentMode(False)
        self.tabWidget.setTabBarAutoHide(False)
        self.tabWidget.setObjectName("tabWidget")
        self.crystals_number_tabcontrols = QtWidgets.QWidget()
        self.crystals_number_tabcontrols.setObjectName("crystals_number_tabcontrols")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.crystals_number_tabcontrols)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.groupBox_5 = QtWidgets.QGroupBox(self.crystals_number_tabcontrols)
        self.groupBox_5.setObjectName("groupBox_5")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_5)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.label_21 = QtWidgets.QLabel(self.groupBox_5)
        self.label_21.setObjectName("label_21")
        self.gridLayout_5.addWidget(self.label_21, 2, 1, 1, 1)
        self.label_22 = QtWidgets.QLabel(self.groupBox_5)
        self.label_22.setObjectName("label_22")
        self.gridLayout_5.addWidget(self.label_22, 3, 1, 1, 1)
        self.ref2_lineedit = QtWidgets.QLineEdit(self.groupBox_5)
        self.ref2_lineedit.setObjectName("ref2_lineedit")
        self.gridLayout_5.addWidget(self.ref2_lineedit, 3, 2, 1, 1)
        self.ref1_lineedit = QtWidgets.QLineEdit(self.groupBox_5)
        self.ref1_lineedit.setObjectName("ref1_lineedit")
        self.gridLayout_5.addWidget(self.ref1_lineedit, 2, 2, 1, 1)
        self.label_73 = QtWidgets.QLabel(self.groupBox_5)
        self.label_73.setObjectName("label_73")
        self.gridLayout_5.addWidget(self.label_73, 2, 3, 1, 1)
        self.label_74 = QtWidgets.QLabel(self.groupBox_5)
        self.label_74.setObjectName("label_74")
        self.gridLayout_5.addWidget(self.label_74, 3, 3, 1, 1)
        self.gridLayout_6.addWidget(self.groupBox_5, 1, 2, 1, 1)
        self.number_turns_lineedit = QtWidgets.QLineEdit(self.crystals_number_tabcontrols)
        self.number_turns_lineedit.setObjectName("number_turns_lineedit")
        self.gridLayout_6.addWidget(self.number_turns_lineedit, 0, 1, 1, 2)
        self.groupBox_9 = QtWidgets.QGroupBox(self.crystals_number_tabcontrols)
        self.groupBox_9.setObjectName("groupBox_9")
        self.gridLayout_13 = QtWidgets.QGridLayout(self.groupBox_9)
        self.gridLayout_13.setObjectName("gridLayout_13")
        self.ref4_lineedit = QtWidgets.QLineEdit(self.groupBox_9)
        self.ref4_lineedit.setObjectName("ref4_lineedit")
        self.gridLayout_13.addWidget(self.ref4_lineedit, 0, 1, 1, 1)
        self.label_23 = QtWidgets.QLabel(self.groupBox_9)
        self.label_23.setObjectName("label_23")
        self.gridLayout_13.addWidget(self.label_23, 0, 0, 1, 1)
        self.label_24 = QtWidgets.QLabel(self.groupBox_9)
        self.label_24.setObjectName("label_24")
        self.gridLayout_13.addWidget(self.label_24, 3, 0, 1, 1)
        self.label_75 = QtWidgets.QLabel(self.groupBox_9)
        self.label_75.setObjectName("label_75")
        self.gridLayout_13.addWidget(self.label_75, 0, 2, 1, 1)
        self.ref3_lineedit = QtWidgets.QLineEdit(self.groupBox_9)
        self.ref3_lineedit.setObjectName("ref3_lineedit")
        self.gridLayout_13.addWidget(self.ref3_lineedit, 3, 1, 1, 1)
        self.label_76 = QtWidgets.QLabel(self.groupBox_9)
        self.label_76.setObjectName("label_76")
        self.gridLayout_13.addWidget(self.label_76, 3, 2, 1, 1)
        self.gridLayout_6.addWidget(self.groupBox_9, 2, 2, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_6.addItem(spacerItem, 5, 2, 1, 1)
        self.label_27 = QtWidgets.QLabel(self.crystals_number_tabcontrols)
        self.label_27.setObjectName("label_27")
        self.gridLayout_6.addWidget(self.label_27, 0, 0, 1, 1)
        self.groupBox_4 = QtWidgets.QGroupBox(self.crystals_number_tabcontrols)
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.label_25 = QtWidgets.QLabel(self.groupBox_4)
        self.label_25.setObjectName("label_25")
        self.gridLayout_7.addWidget(self.label_25, 0, 0, 1, 1)
        self.mmpc_voltageA_lineedit = QtWidgets.QLineEdit(self.groupBox_4)
        self.mmpc_voltageA_lineedit.setObjectName("mmpc_voltageA_lineedit")
        self.gridLayout_7.addWidget(self.mmpc_voltageA_lineedit, 0, 1, 1, 1)
        self.label_26 = QtWidgets.QLabel(self.groupBox_4)
        self.label_26.setObjectName("label_26")
        self.gridLayout_7.addWidget(self.label_26, 1, 0, 1, 1)
        self.mmpc_voltageB_lineedit = QtWidgets.QLineEdit(self.groupBox_4)
        self.mmpc_voltageB_lineedit.setObjectName("mmpc_voltageB_lineedit")
        self.gridLayout_7.addWidget(self.mmpc_voltageB_lineedit, 1, 1, 1, 1)
        self.gridLayout_6.addWidget(self.groupBox_4, 1, 0, 2, 2)
        self.tabWidget.addTab(self.crystals_number_tabcontrols, "")
        self.crystals_time_tabcontrols = QtWidgets.QWidget()
        self.crystals_time_tabcontrols.setObjectName("crystals_time_tabcontrols")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.crystals_time_tabcontrols)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.groupBox = QtWidgets.QGroupBox(self.crystals_time_tabcontrols)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.max_velocity_top_spinbox = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.max_velocity_top_spinbox.setDecimals(2)
        self.max_velocity_top_spinbox.setMinimum(15.25)
        self.max_velocity_top_spinbox.setMaximum(15625.0)
        self.max_velocity_top_spinbox.setSingleStep(15.25)
        self.max_velocity_top_spinbox.setObjectName("max_velocity_top_spinbox")
        self.gridLayout_3.addWidget(self.max_velocity_top_spinbox, 5, 1, 1, 1)
        self.label_realstep = QtWidgets.QLabel(self.groupBox)
        self.label_realstep.setObjectName("label_realstep")
        self.gridLayout_3.addWidget(self.label_realstep, 0, 2, 2, 1)
        self.label_19 = QtWidgets.QLabel(self.groupBox)
        self.label_19.setObjectName("label_19")
        self.gridLayout_3.addWidget(self.label_19, 5, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 1, 0, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.groupBox)
        self.label_13.setObjectName("label_13")
        self.gridLayout_3.addWidget(self.label_13, 7, 0, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.groupBox)
        self.label_16.setObjectName("label_16")
        self.gridLayout_3.addWidget(self.label_16, 6, 2, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.groupBox)
        self.label_9.setObjectName("label_9")
        self.gridLayout_3.addWidget(self.label_9, 6, 0, 1, 1)
        self.min_velocity_top_spinbox = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.min_velocity_top_spinbox.setDecimals(2)
        self.min_velocity_top_spinbox.setMaximum(976.3)
        self.min_velocity_top_spinbox.setSingleStep(0.238)
        self.min_velocity_top_spinbox.setObjectName("min_velocity_top_spinbox")
        self.gridLayout_3.addWidget(self.min_velocity_top_spinbox, 4, 1, 1, 1)
        self.label_18 = QtWidgets.QLabel(self.groupBox)
        self.label_18.setObjectName("label_18")
        self.gridLayout_3.addWidget(self.label_18, 4, 2, 1, 1)
        self.Full_step_top_spinbox = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.Full_step_top_spinbox.setDecimals(1)
        self.Full_step_top_spinbox.setMaximum(360.0)
        self.Full_step_top_spinbox.setSingleStep(1.8)
        self.Full_step_top_spinbox.setObjectName("Full_step_top_spinbox")
        self.gridLayout_3.addWidget(self.Full_step_top_spinbox, 0, 1, 1, 1)
        self.dec_top_spinbox = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.dec_top_spinbox.setMinimum(14.55)
        self.dec_top_spinbox.setMaximum(59590.0)
        self.dec_top_spinbox.setSingleStep(14.55)
        self.dec_top_spinbox.setProperty("value", 14.55)
        self.dec_top_spinbox.setObjectName("dec_top_spinbox")
        self.gridLayout_3.addWidget(self.dec_top_spinbox, 7, 1, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.groupBox)
        self.label_15.setObjectName("label_15")
        self.gridLayout_3.addWidget(self.label_15, 2, 2, 1, 1)
        self.microstepping_top_combobox = QtWidgets.QComboBox(self.groupBox)
        self.microstepping_top_combobox.setStyleSheet("QComboBox {\n"
"    border: 1px solid gray;\n"
"    border-radius: 3px;\n"
"    padding: 1px 0px 1px 3px;\n"
"    color:white;\n"
"    font:14px;\n"
"background-color:qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.946, fx:0.744, fy:0.159, stop:0 rgba(140, 150, 165, 255), stop:1 rgba(82, 98, 112, 255));\n"
"font-weight:bold;\n"
"}\n"
"\n"
"\n"
"/* \n"
"QComboBox::drop-down {\n"
"     background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                 stop: 0 #E1E1E1, stop: 0.4 #DDDDDD,\n"
"                                 stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3);\n"
"}\n"
"\n"
"QComboBox gets the \"on\" state when the popup is open \n"
"\n"
"\n"
"QComboBox:on { /* shift the text when the popup opens \n"
"    padding-top: 3px;\n"
"    padding-left: 4px;\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                stop: 0 #D3D3D3, stop: 0.4 #D8D8D8,\n"
"                                stop: 0.5 #DDDDDD, stop: 1.0 #E1E1E1);\n"
"}\n"
"*/\n"
" QComboBox::drop-down:on {\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                stop: 0 #D3D3D3, stop: 0.4 #D8D8D8,\n"
"                                stop: 0.5 #DDDDDD, stop: 1.0 #E1E1E1);\n"
"}\n"
"\n"
"QComboBox::drop-down {\n"
"    subcontrol-origin: padding;\n"
"    subcontrol-position: top right;\n"
"    width: 30px;  \n"
"    font-weight:normal;\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                 stop: 0 #E1E1E1, stop: 0.4 #DDDDDD,\n"
"                                 stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3);    \n"
"\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"    image: url(Resources/if_angle-down_1608507.png);\n"
"    width:30px;\n"
"height:35px;\n"
"    background-color:qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.946, fx:0.744, fy:0.159, stop:0 rgba(140, 150, 165, 255), stop:1 rgba(82, 98, 112, 255));\n"
"    \n"
"}\n"
"\n"
"QComboBox::down-arrow:on { /* shift the arrow when popup is open */\n"
"    top: 1px;\n"
"    left: 1px;\n"
"}\n"
"\n"
"\n"
"QComboBox QAbstractItemView {\n"
"    border: 2px solid darkgray;\n"
"    selection-background-color: rgb(95, 186, 189,235);\n"
"    background-color:lightgray;\n"
"    font-weight: bold;\n"
"}\n"
"")
        self.microstepping_top_combobox.setObjectName("microstepping_top_combobox")
        self.microstepping_top_combobox.addItem("")
        self.microstepping_top_combobox.addItem("")
        self.microstepping_top_combobox.addItem("")
        self.microstepping_top_combobox.addItem("")
        self.microstepping_top_combobox.addItem("")
        self.microstepping_top_combobox.addItem("")
        self.microstepping_top_combobox.addItem("")
        self.microstepping_top_combobox.addItem("")
        self.gridLayout_3.addWidget(self.microstepping_top_combobox, 1, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.gridLayout_3.addWidget(self.label, 0, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.groupBox)
        self.label_8.setObjectName("label_8")
        self.gridLayout_3.addWidget(self.label_8, 4, 0, 1, 1)
        self.current_velocity_top_spinbox = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.current_velocity_top_spinbox.setDecimals(0)
        self.current_velocity_top_spinbox.setMaximum(15625.0)
        self.current_velocity_top_spinbox.setObjectName("current_velocity_top_spinbox")
        self.gridLayout_3.addWidget(self.current_velocity_top_spinbox, 2, 1, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.groupBox)
        self.label_17.setObjectName("label_17")
        self.gridLayout_3.addWidget(self.label_17, 7, 2, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.groupBox)
        self.label_6.setObjectName("label_6")
        self.gridLayout_3.addWidget(self.label_6, 2, 0, 1, 1)
        self.acc_top_spinbox = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.acc_top_spinbox.setMinimum(14.55)
        self.acc_top_spinbox.setMaximum(59590.0)
        self.acc_top_spinbox.setSingleStep(14.55)
        self.acc_top_spinbox.setProperty("value", 14.55)
        self.acc_top_spinbox.setObjectName("acc_top_spinbox")
        self.gridLayout_3.addWidget(self.acc_top_spinbox, 6, 1, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setObjectName("label_7")
        self.gridLayout_3.addWidget(self.label_7, 5, 0, 1, 1)
        self.gridLayout_9.addWidget(self.groupBox, 1, 0, 1, 5)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_9.addItem(spacerItem1, 0, 2, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_9.addItem(spacerItem2, 0, 3, 1, 1)
        self.groupBox_3 = QtWidgets.QGroupBox(self.crystals_time_tabcontrols)
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_4 = QtWidgets.QLabel(self.groupBox_3)
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, 2, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.groupBox_3)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 3, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.groupBox_3)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 0, 0, 1, 1)
        self.label_31 = QtWidgets.QLabel(self.groupBox_3)
        self.label_31.setObjectName("label_31")
        self.gridLayout_2.addWidget(self.label_31, 0, 3, 1, 1)
        self.label_33 = QtWidgets.QLabel(self.groupBox_3)
        self.label_33.setObjectName("label_33")
        self.gridLayout_2.addWidget(self.label_33, 3, 3, 1, 1)
        self.label_32 = QtWidgets.QLabel(self.groupBox_3)
        self.label_32.setObjectName("label_32")
        self.gridLayout_2.addWidget(self.label_32, 2, 3, 1, 1)
        self.startslope_top_spinbox = QtWidgets.QDoubleSpinBox(self.groupBox_3)
        self.startslope_top_spinbox.setDecimals(3)
        self.startslope_top_spinbox.setMinimum(0.0)
        self.startslope_top_spinbox.setMaximum(4.0)
        self.startslope_top_spinbox.setSingleStep(0.015)
        self.startslope_top_spinbox.setObjectName("startslope_top_spinbox")
        self.gridLayout_2.addWidget(self.startslope_top_spinbox, 0, 1, 1, 2)
        self.acc_slope_top_spinbox = QtWidgets.QDoubleSpinBox(self.groupBox_3)
        self.acc_slope_top_spinbox.setDecimals(3)
        self.acc_slope_top_spinbox.setMaximum(4.0)
        self.acc_slope_top_spinbox.setSingleStep(0.015)
        self.acc_slope_top_spinbox.setProperty("value", 0.01)
        self.acc_slope_top_spinbox.setObjectName("acc_slope_top_spinbox")
        self.gridLayout_2.addWidget(self.acc_slope_top_spinbox, 2, 1, 1, 2)
        self.dec_slope_top_spinbox = QtWidgets.QDoubleSpinBox(self.groupBox_3)
        self.dec_slope_top_spinbox.setDecimals(3)
        self.dec_slope_top_spinbox.setMaximum(4.0)
        self.dec_slope_top_spinbox.setSingleStep(0.015)
        self.dec_slope_top_spinbox.setObjectName("dec_slope_top_spinbox")
        self.gridLayout_2.addWidget(self.dec_slope_top_spinbox, 3, 1, 1, 2)
        self.gridLayout_9.addWidget(self.groupBox_3, 2, 0, 1, 3)
        self.label_min_full_step = QtWidgets.QLabel(self.crystals_time_tabcontrols)
        self.label_min_full_step.setObjectName("label_min_full_step")
        self.gridLayout_9.addWidget(self.label_min_full_step, 0, 4, 1, 1)
        self.comboBox_chooseTopMotor = QtWidgets.QComboBox(self.crystals_time_tabcontrols)
        self.comboBox_chooseTopMotor.setStyleSheet("QComboBox {\n"
"    border: 1px solid gray;\n"
"    border-radius: 3px;\n"
"    padding: 1px 0px 1px 3px;\n"
"    color:white;\n"
"    font:14px;\n"
"background-color:qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.946, fx:0.744, fy:0.159, stop:0 rgba(140, 150, 165, 255), stop:1 rgba(82, 98, 112, 255));\n"
"font-weight:bold;\n"
"}\n"
"\n"
"\n"
"/* \n"
"QComboBox::drop-down {\n"
"     background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                 stop: 0 #E1E1E1, stop: 0.4 #DDDDDD,\n"
"                                 stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3);\n"
"}\n"
"\n"
"QComboBox gets the \"on\" state when the popup is open \n"
"\n"
"\n"
"QComboBox:on { /* shift the text when the popup opens \n"
"    padding-top: 3px;\n"
"    padding-left: 4px;\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                stop: 0 #D3D3D3, stop: 0.4 #D8D8D8,\n"
"                                stop: 0.5 #DDDDDD, stop: 1.0 #E1E1E1);\n"
"}\n"
"*/\n"
" QComboBox::drop-down:on {\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                stop: 0 #D3D3D3, stop: 0.4 #D8D8D8,\n"
"                                stop: 0.5 #DDDDDD, stop: 1.0 #E1E1E1);\n"
"}\n"
"\n"
"QComboBox::drop-down {\n"
"    subcontrol-origin: padding;\n"
"    subcontrol-position: top right;\n"
"    width: 30px;  \n"
"    font-weight:normal;\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                 stop: 0 #E1E1E1, stop: 0.4 #DDDDDD,\n"
"                                 stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3);    \n"
"\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"    image: url(Resources/if_angle-down_1608507.png);\n"
"    width:30px;\n"
"height:35px;\n"
"    background-color:qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.946, fx:0.744, fy:0.159, stop:0 rgba(140, 150, 165, 255), stop:1 rgba(82, 98, 112, 255));\n"
"    \n"
"}\n"
"\n"
"QComboBox::down-arrow:on { /* shift the arrow when popup is open */\n"
"    top: 1px;\n"
"    left: 1px;\n"
"}\n"
"\n"
"\n"
"QComboBox QAbstractItemView {\n"
"    border: 2px solid darkgray;\n"
"    selection-background-color: rgb(95, 186, 189,235);\n"
"    background-color:lightgray;\n"
"    font-weight: bold;\n"
"}\n"
"")
        self.comboBox_chooseTopMotor.setObjectName("comboBox_chooseTopMotor")
        self.comboBox_chooseTopMotor.addItem("")
        self.comboBox_chooseTopMotor.addItem("")
        self.gridLayout_9.addWidget(self.comboBox_chooseTopMotor, 0, 0, 1, 2)
        self.groupBox_2 = QtWidgets.QGroupBox(self.crystals_time_tabcontrols)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_29 = QtWidgets.QLabel(self.groupBox_2)
        self.label_29.setObjectName("label_29")
        self.gridLayout_4.addWidget(self.label_29, 5, 0, 1, 1)
        self.current_top_lineedit = QtWidgets.QLineEdit(self.groupBox_2)
        self.current_top_lineedit.setObjectName("current_top_lineedit")
        self.gridLayout_4.addWidget(self.current_top_lineedit, 1, 1, 1, 1)
        self.label_28 = QtWidgets.QLabel(self.groupBox_2)
        self.label_28.setObjectName("label_28")
        self.gridLayout_4.addWidget(self.label_28, 4, 0, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.groupBox_2)
        self.label_11.setObjectName("label_11")
        self.gridLayout_4.addWidget(self.label_11, 1, 0, 1, 1)
        self.label_20 = QtWidgets.QLabel(self.groupBox_2)
        self.label_20.setObjectName("label_20")
        self.gridLayout_4.addWidget(self.label_20, 3, 0, 1, 1)
        self.phasevoltage_top_lineedit = QtWidgets.QLineEdit(self.groupBox_2)
        self.phasevoltage_top_lineedit.setObjectName("phasevoltage_top_lineedit")
        self.gridLayout_4.addWidget(self.phasevoltage_top_lineedit, 2, 1, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.groupBox_2)
        self.label_10.setObjectName("label_10")
        self.gridLayout_4.addWidget(self.label_10, 0, 0, 1, 1)
        self.voltage_top_lineedit = QtWidgets.QLineEdit(self.groupBox_2)
        self.voltage_top_lineedit.setDragEnabled(False)
        self.voltage_top_lineedit.setClearButtonEnabled(False)
        self.voltage_top_lineedit.setObjectName("voltage_top_lineedit")
        self.gridLayout_4.addWidget(self.voltage_top_lineedit, 0, 1, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.groupBox_2)
        self.label_12.setObjectName("label_12")
        self.gridLayout_4.addWidget(self.label_12, 2, 0, 1, 1)
        self.label_30 = QtWidgets.QLabel(self.groupBox_2)
        self.label_30.setObjectName("label_30")
        self.gridLayout_4.addWidget(self.label_30, 6, 0, 1, 1)
        self.holding_kval_top_lineedit = QtWidgets.QLineEdit(self.groupBox_2)
        self.holding_kval_top_lineedit.setObjectName("holding_kval_top_lineedit")
        self.gridLayout_4.addWidget(self.holding_kval_top_lineedit, 3, 1, 1, 1)
        self.constant_speed_top_lineedit = QtWidgets.QLineEdit(self.groupBox_2)
        self.constant_speed_top_lineedit.setObjectName("constant_speed_top_lineedit")
        self.gridLayout_4.addWidget(self.constant_speed_top_lineedit, 4, 1, 1, 1)
        self.dec_top_lineedit = QtWidgets.QLineEdit(self.groupBox_2)
        self.dec_top_lineedit.setObjectName("dec_top_lineedit")
        self.gridLayout_4.addWidget(self.dec_top_lineedit, 6, 1, 1, 1)
        self.acc_top_lineedit = QtWidgets.QLineEdit(self.groupBox_2)
        self.acc_top_lineedit.setObjectName("acc_top_lineedit")
        self.gridLayout_4.addWidget(self.acc_top_lineedit, 5, 1, 1, 1)
        self.label_66 = QtWidgets.QLabel(self.groupBox_2)
        self.label_66.setObjectName("label_66")
        self.gridLayout_4.addWidget(self.label_66, 0, 2, 1, 1)
        self.label_67 = QtWidgets.QLabel(self.groupBox_2)
        self.label_67.setObjectName("label_67")
        self.gridLayout_4.addWidget(self.label_67, 1, 2, 1, 1)
        self.label_68 = QtWidgets.QLabel(self.groupBox_2)
        self.label_68.setObjectName("label_68")
        self.gridLayout_4.addWidget(self.label_68, 2, 2, 1, 1)
        self.label_69 = QtWidgets.QLabel(self.groupBox_2)
        self.label_69.setObjectName("label_69")
        self.gridLayout_4.addWidget(self.label_69, 3, 2, 1, 1)
        self.label_70 = QtWidgets.QLabel(self.groupBox_2)
        self.label_70.setObjectName("label_70")
        self.gridLayout_4.addWidget(self.label_70, 4, 2, 1, 1)
        self.label_71 = QtWidgets.QLabel(self.groupBox_2)
        self.label_71.setObjectName("label_71")
        self.gridLayout_4.addWidget(self.label_71, 5, 2, 1, 1)
        self.label_72 = QtWidgets.QLabel(self.groupBox_2)
        self.label_72.setObjectName("label_72")
        self.gridLayout_4.addWidget(self.label_72, 6, 2, 1, 1)
        self.gridLayout_9.addWidget(self.groupBox_2, 2, 3, 1, 2)
        self.tabWidget.addTab(self.crystals_time_tabcontrols, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout_12 = QtWidgets.QGridLayout(self.tab)
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.label_min_bot_full_step = QtWidgets.QLabel(self.tab)
        self.label_min_bot_full_step.setObjectName("label_min_bot_full_step")
        self.gridLayout_12.addWidget(self.label_min_bot_full_step, 0, 3, 1, 1)
        self.comboBox_chooseBotMotor = QtWidgets.QComboBox(self.tab)
        self.comboBox_chooseBotMotor.setStyleSheet("QComboBox {\n"
"    border: 1px solid gray;\n"
"    border-radius: 3px;\n"
"    padding: 1px 0px 1px 3px;\n"
"    color:white;\n"
"    font:14px;\n"
"background-color:qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.946, fx:0.744, fy:0.159, stop:0 rgba(140, 150, 165, 255), stop:1 rgba(82, 98, 112, 255));\n"
"font-weight:bold;\n"
"}\n"
"\n"
"\n"
"/* \n"
"QComboBox::drop-down {\n"
"     background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                 stop: 0 #E1E1E1, stop: 0.4 #DDDDDD,\n"
"                                 stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3);\n"
"}\n"
"\n"
"QComboBox gets the \"on\" state when the popup is open \n"
"\n"
"\n"
"QComboBox:on { /* shift the text when the popup opens \n"
"    padding-top: 3px;\n"
"    padding-left: 4px;\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                stop: 0 #D3D3D3, stop: 0.4 #D8D8D8,\n"
"                                stop: 0.5 #DDDDDD, stop: 1.0 #E1E1E1);\n"
"}\n"
"*/\n"
" QComboBox::drop-down:on {\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                stop: 0 #D3D3D3, stop: 0.4 #D8D8D8,\n"
"                                stop: 0.5 #DDDDDD, stop: 1.0 #E1E1E1);\n"
"}\n"
"\n"
"QComboBox::drop-down {\n"
"    subcontrol-origin: padding;\n"
"    subcontrol-position: top right;\n"
"    width: 30px;  \n"
"    font-weight:normal;\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                 stop: 0 #E1E1E1, stop: 0.4 #DDDDDD,\n"
"                                 stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3);    \n"
"\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"    image: url(Resources/if_angle-down_1608507.png);\n"
"    width:30px;\n"
"height:35px;\n"
"    background-color:qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.946, fx:0.744, fy:0.159, stop:0 rgba(140, 150, 165, 255), stop:1 rgba(82, 98, 112, 255));\n"
"    \n"
"}\n"
"\n"
"QComboBox::down-arrow:on { /* shift the arrow when popup is open */\n"
"    top: 1px;\n"
"    left: 1px;\n"
"}\n"
"\n"
"\n"
"QComboBox QAbstractItemView {\n"
"    border: 2px solid darkgray;\n"
"    selection-background-color: rgb(95, 186, 189,235);\n"
"    background-color:lightgray;\n"
"    font-weight: bold;\n"
"}\n"
"")
        self.comboBox_chooseBotMotor.setObjectName("comboBox_chooseBotMotor")
        self.comboBox_chooseBotMotor.addItem("")
        self.gridLayout_12.addWidget(self.comboBox_chooseBotMotor, 0, 0, 1, 1)
        self.groupBox_6 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_6.setObjectName("groupBox_6")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.groupBox_6)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.label_real_bot_step = QtWidgets.QLabel(self.groupBox_6)
        self.label_real_bot_step.setObjectName("label_real_bot_step")
        self.gridLayout_8.addWidget(self.label_real_bot_step, 0, 2, 2, 1)
        self.label_35 = QtWidgets.QLabel(self.groupBox_6)
        self.label_35.setObjectName("label_35")
        self.gridLayout_8.addWidget(self.label_35, 2, 0, 1, 1)
        self.label_36 = QtWidgets.QLabel(self.groupBox_6)
        self.label_36.setObjectName("label_36")
        self.gridLayout_8.addWidget(self.label_36, 0, 0, 1, 1)
        self.label_37 = QtWidgets.QLabel(self.groupBox_6)
        self.label_37.setObjectName("label_37")
        self.gridLayout_8.addWidget(self.label_37, 6, 0, 1, 1)
        self.microstepping_bot_combobox = QtWidgets.QComboBox(self.groupBox_6)
        self.microstepping_bot_combobox.setStyleSheet("QComboBox {\n"
"    border: 1px solid gray;\n"
"    border-radius: 3px;\n"
"    padding: 1px 0px 1px 3px;\n"
"    color:white;\n"
"    font:14px;\n"
"background-color:qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.946, fx:0.744, fy:0.159, stop:0 rgba(140, 150, 165, 255), stop:1 rgba(82, 98, 112, 255));\n"
"font-weight:bold;\n"
"}\n"
"\n"
"\n"
"/* \n"
"QComboBox::drop-down {\n"
"     background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                 stop: 0 #E1E1E1, stop: 0.4 #DDDDDD,\n"
"                                 stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3);\n"
"}\n"
"\n"
"QComboBox gets the \"on\" state when the popup is open \n"
"\n"
"\n"
"QComboBox:on { /* shift the text when the popup opens \n"
"    padding-top: 3px;\n"
"    padding-left: 4px;\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                stop: 0 #D3D3D3, stop: 0.4 #D8D8D8,\n"
"                                stop: 0.5 #DDDDDD, stop: 1.0 #E1E1E1);\n"
"}\n"
"*/\n"
" QComboBox::drop-down:on {\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                stop: 0 #D3D3D3, stop: 0.4 #D8D8D8,\n"
"                                stop: 0.5 #DDDDDD, stop: 1.0 #E1E1E1);\n"
"}\n"
"\n"
"QComboBox::drop-down {\n"
"    subcontrol-origin: padding;\n"
"    subcontrol-position: top right;\n"
"    width: 30px;  \n"
"    font-weight:normal;\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                 stop: 0 #E1E1E1, stop: 0.4 #DDDDDD,\n"
"                                 stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3);    \n"
"\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"    image: url(Resources/if_angle-down_1608507.png);\n"
"    width:30px;\n"
"height:35px;\n"
"    background-color:qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.946, fx:0.744, fy:0.159, stop:0 rgba(140, 150, 165, 255), stop:1 rgba(82, 98, 112, 255));\n"
"    \n"
"}\n"
"\n"
"QComboBox::down-arrow:on { /* shift the arrow when popup is open */\n"
"    top: 1px;\n"
"    left: 1px;\n"
"}\n"
"\n"
"\n"
"QComboBox QAbstractItemView {\n"
"    border: 2px solid darkgray;\n"
"    selection-background-color: rgb(95, 186, 189,235);\n"
"    background-color:lightgray;\n"
"    font-weight: bold;\n"
"}\n"
"")
        self.microstepping_bot_combobox.setObjectName("microstepping_bot_combobox")
        self.microstepping_bot_combobox.addItem("")
        self.microstepping_bot_combobox.addItem("")
        self.microstepping_bot_combobox.addItem("")
        self.microstepping_bot_combobox.addItem("")
        self.microstepping_bot_combobox.addItem("")
        self.microstepping_bot_combobox.addItem("")
        self.microstepping_bot_combobox.addItem("")
        self.microstepping_bot_combobox.addItem("")
        self.gridLayout_8.addWidget(self.microstepping_bot_combobox, 1, 1, 1, 1)
        self.label_38 = QtWidgets.QLabel(self.groupBox_6)
        self.label_38.setObjectName("label_38")
        self.gridLayout_8.addWidget(self.label_38, 1, 0, 1, 1)
        self.label_39 = QtWidgets.QLabel(self.groupBox_6)
        self.label_39.setObjectName("label_39")
        self.gridLayout_8.addWidget(self.label_39, 4, 0, 1, 1)
        self.label_40 = QtWidgets.QLabel(self.groupBox_6)
        self.label_40.setObjectName("label_40")
        self.gridLayout_8.addWidget(self.label_40, 7, 0, 1, 1)
        self.label_41 = QtWidgets.QLabel(self.groupBox_6)
        self.label_41.setObjectName("label_41")
        self.gridLayout_8.addWidget(self.label_41, 5, 0, 1, 1)
        self.dec_bot_spinbox = QtWidgets.QDoubleSpinBox(self.groupBox_6)
        self.dec_bot_spinbox.setMinimum(14.55)
        self.dec_bot_spinbox.setMaximum(59590.0)
        self.dec_bot_spinbox.setSingleStep(14.55)
        self.dec_bot_spinbox.setObjectName("dec_bot_spinbox")
        self.gridLayout_8.addWidget(self.dec_bot_spinbox, 7, 1, 1, 1)
        self.acc_bot_spinbox = QtWidgets.QDoubleSpinBox(self.groupBox_6)
        self.acc_bot_spinbox.setMinimum(14.55)
        self.acc_bot_spinbox.setMaximum(59590.0)
        self.acc_bot_spinbox.setSingleStep(14.55)
        self.acc_bot_spinbox.setObjectName("acc_bot_spinbox")
        self.gridLayout_8.addWidget(self.acc_bot_spinbox, 6, 1, 1, 1)
        self.max_velocity_bot_spinbox = QtWidgets.QDoubleSpinBox(self.groupBox_6)
        self.max_velocity_bot_spinbox.setDecimals(0)
        self.max_velocity_bot_spinbox.setMaximum(15625.0)
        self.max_velocity_bot_spinbox.setObjectName("max_velocity_bot_spinbox")
        self.gridLayout_8.addWidget(self.max_velocity_bot_spinbox, 5, 1, 1, 1)
        self.min_velocity_bot_spinbox = QtWidgets.QDoubleSpinBox(self.groupBox_6)
        self.min_velocity_bot_spinbox.setDecimals(0)
        self.min_velocity_bot_spinbox.setMaximum(15625.0)
        self.min_velocity_bot_spinbox.setObjectName("min_velocity_bot_spinbox")
        self.gridLayout_8.addWidget(self.min_velocity_bot_spinbox, 4, 1, 1, 1)
        self.current_velocity_bot_spinbox = QtWidgets.QDoubleSpinBox(self.groupBox_6)
        self.current_velocity_bot_spinbox.setDecimals(0)
        self.current_velocity_bot_spinbox.setMaximum(15625.0)
        self.current_velocity_bot_spinbox.setObjectName("current_velocity_bot_spinbox")
        self.gridLayout_8.addWidget(self.current_velocity_bot_spinbox, 2, 1, 1, 1)
        self.label_42 = QtWidgets.QLabel(self.groupBox_6)
        self.label_42.setObjectName("label_42")
        self.gridLayout_8.addWidget(self.label_42, 2, 2, 1, 1)
        self.label_43 = QtWidgets.QLabel(self.groupBox_6)
        self.label_43.setObjectName("label_43")
        self.gridLayout_8.addWidget(self.label_43, 7, 2, 1, 1)
        self.label_44 = QtWidgets.QLabel(self.groupBox_6)
        self.label_44.setObjectName("label_44")
        self.gridLayout_8.addWidget(self.label_44, 6, 2, 1, 1)
        self.label_45 = QtWidgets.QLabel(self.groupBox_6)
        self.label_45.setObjectName("label_45")
        self.gridLayout_8.addWidget(self.label_45, 4, 2, 1, 1)
        self.label_46 = QtWidgets.QLabel(self.groupBox_6)
        self.label_46.setObjectName("label_46")
        self.gridLayout_8.addWidget(self.label_46, 5, 2, 1, 1)
        self.Full_step_bot_spinbox = QtWidgets.QDoubleSpinBox(self.groupBox_6)
        self.Full_step_bot_spinbox.setDecimals(1)
        self.Full_step_bot_spinbox.setMaximum(360.0)
        self.Full_step_bot_spinbox.setObjectName("Full_step_bot_spinbox")
        self.gridLayout_8.addWidget(self.Full_step_bot_spinbox, 0, 1, 1, 1)
        self.gridLayout_12.addWidget(self.groupBox_6, 1, 0, 1, 4)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_12.addItem(spacerItem3, 0, 2, 1, 1)
        self.groupBox_8 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_8.setObjectName("groupBox_8")
        self.gridLayout_11 = QtWidgets.QGridLayout(self.groupBox_8)
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.label_54 = QtWidgets.QLabel(self.groupBox_8)
        self.label_54.setObjectName("label_54")
        self.gridLayout_11.addWidget(self.label_54, 5, 0, 1, 1)
        self.label_53 = QtWidgets.QLabel(self.groupBox_8)
        self.label_53.setObjectName("label_53")
        self.gridLayout_11.addWidget(self.label_53, 4, 0, 1, 1)
        self.phasevoltage_bot_lineedit = QtWidgets.QLineEdit(self.groupBox_8)
        self.phasevoltage_bot_lineedit.setObjectName("phasevoltage_bot_lineedit")
        self.gridLayout_11.addWidget(self.phasevoltage_bot_lineedit, 2, 1, 1, 1)
        self.current_bot_lineedit = QtWidgets.QLineEdit(self.groupBox_8)
        self.current_bot_lineedit.setObjectName("current_bot_lineedit")
        self.gridLayout_11.addWidget(self.current_bot_lineedit, 1, 1, 1, 1)
        self.label_55 = QtWidgets.QLabel(self.groupBox_8)
        self.label_55.setObjectName("label_55")
        self.gridLayout_11.addWidget(self.label_55, 3, 0, 1, 1)
        self.label_57 = QtWidgets.QLabel(self.groupBox_8)
        self.label_57.setObjectName("label_57")
        self.gridLayout_11.addWidget(self.label_57, 1, 0, 1, 1)
        self.label_58 = QtWidgets.QLabel(self.groupBox_8)
        self.label_58.setObjectName("label_58")
        self.gridLayout_11.addWidget(self.label_58, 2, 0, 1, 1)
        self.voltage_bot_lineedit = QtWidgets.QLineEdit(self.groupBox_8)
        self.voltage_bot_lineedit.setObjectName("voltage_bot_lineedit")
        self.gridLayout_11.addWidget(self.voltage_bot_lineedit, 0, 1, 1, 1)
        self.label_56 = QtWidgets.QLabel(self.groupBox_8)
        self.label_56.setObjectName("label_56")
        self.gridLayout_11.addWidget(self.label_56, 0, 0, 1, 1)
        self.acc_bot_lineedit = QtWidgets.QLineEdit(self.groupBox_8)
        self.acc_bot_lineedit.setObjectName("acc_bot_lineedit")
        self.gridLayout_11.addWidget(self.acc_bot_lineedit, 5, 1, 1, 1)
        self.label_59 = QtWidgets.QLabel(self.groupBox_8)
        self.label_59.setObjectName("label_59")
        self.gridLayout_11.addWidget(self.label_59, 6, 0, 1, 1)
        self.constant_speed_bot_lineedit = QtWidgets.QLineEdit(self.groupBox_8)
        self.constant_speed_bot_lineedit.setObjectName("constant_speed_bot_lineedit")
        self.gridLayout_11.addWidget(self.constant_speed_bot_lineedit, 4, 1, 1, 1)
        self.holding_kval_bot_lineedit = QtWidgets.QLineEdit(self.groupBox_8)
        self.holding_kval_bot_lineedit.setObjectName("holding_kval_bot_lineedit")
        self.gridLayout_11.addWidget(self.holding_kval_bot_lineedit, 3, 1, 1, 1)
        self.dec_bot_lineedit = QtWidgets.QLineEdit(self.groupBox_8)
        self.dec_bot_lineedit.setObjectName("dec_bot_lineedit")
        self.gridLayout_11.addWidget(self.dec_bot_lineedit, 6, 1, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.groupBox_8)
        self.label_14.setObjectName("label_14")
        self.gridLayout_11.addWidget(self.label_14, 0, 2, 1, 1)
        self.label_60 = QtWidgets.QLabel(self.groupBox_8)
        self.label_60.setObjectName("label_60")
        self.gridLayout_11.addWidget(self.label_60, 1, 2, 1, 1)
        self.label_61 = QtWidgets.QLabel(self.groupBox_8)
        self.label_61.setObjectName("label_61")
        self.gridLayout_11.addWidget(self.label_61, 2, 2, 1, 1)
        self.label_62 = QtWidgets.QLabel(self.groupBox_8)
        self.label_62.setObjectName("label_62")
        self.gridLayout_11.addWidget(self.label_62, 3, 2, 1, 1)
        self.label_63 = QtWidgets.QLabel(self.groupBox_8)
        self.label_63.setObjectName("label_63")
        self.gridLayout_11.addWidget(self.label_63, 4, 2, 1, 1)
        self.label_64 = QtWidgets.QLabel(self.groupBox_8)
        self.label_64.setObjectName("label_64")
        self.gridLayout_11.addWidget(self.label_64, 5, 2, 1, 1)
        self.label_65 = QtWidgets.QLabel(self.groupBox_8)
        self.label_65.setObjectName("label_65")
        self.gridLayout_11.addWidget(self.label_65, 6, 2, 1, 1)
        self.gridLayout_12.addWidget(self.groupBox_8, 2, 2, 1, 2)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_12.addItem(spacerItem4, 0, 1, 1, 1)
        self.groupBox_7 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_7.setObjectName("groupBox_7")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.groupBox_7)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.label_47 = QtWidgets.QLabel(self.groupBox_7)
        self.label_47.setObjectName("label_47")
        self.gridLayout_10.addWidget(self.label_47, 2, 0, 1, 1)
        self.label_48 = QtWidgets.QLabel(self.groupBox_7)
        self.label_48.setObjectName("label_48")
        self.gridLayout_10.addWidget(self.label_48, 1, 0, 1, 1)
        self.label_49 = QtWidgets.QLabel(self.groupBox_7)
        self.label_49.setObjectName("label_49")
        self.gridLayout_10.addWidget(self.label_49, 0, 0, 1, 1)
        self.label_50 = QtWidgets.QLabel(self.groupBox_7)
        self.label_50.setObjectName("label_50")
        self.gridLayout_10.addWidget(self.label_50, 0, 3, 1, 1)
        self.label_51 = QtWidgets.QLabel(self.groupBox_7)
        self.label_51.setObjectName("label_51")
        self.gridLayout_10.addWidget(self.label_51, 1, 3, 1, 1)
        self.label_52 = QtWidgets.QLabel(self.groupBox_7)
        self.label_52.setObjectName("label_52")
        self.gridLayout_10.addWidget(self.label_52, 2, 3, 1, 1)
        self.startslope_bot_spinbox = QtWidgets.QDoubleSpinBox(self.groupBox_7)
        self.startslope_bot_spinbox.setDecimals(3)
        self.startslope_bot_spinbox.setMaximum(4.0)
        self.startslope_bot_spinbox.setSingleStep(0.015)
        self.startslope_bot_spinbox.setObjectName("startslope_bot_spinbox")
        self.gridLayout_10.addWidget(self.startslope_bot_spinbox, 0, 1, 1, 2)
        self.acc_slope_bot_spinbox = QtWidgets.QDoubleSpinBox(self.groupBox_7)
        self.acc_slope_bot_spinbox.setDecimals(3)
        self.acc_slope_bot_spinbox.setMaximum(4.0)
        self.acc_slope_bot_spinbox.setSingleStep(0.015)
        self.acc_slope_bot_spinbox.setProperty("value", 0.01)
        self.acc_slope_bot_spinbox.setObjectName("acc_slope_bot_spinbox")
        self.gridLayout_10.addWidget(self.acc_slope_bot_spinbox, 1, 1, 1, 2)
        self.dec_slope_bot_spinbox = QtWidgets.QDoubleSpinBox(self.groupBox_7)
        self.dec_slope_bot_spinbox.setDecimals(3)
        self.dec_slope_bot_spinbox.setMaximum(4.0)
        self.dec_slope_bot_spinbox.setSingleStep(0.015)
        self.dec_slope_bot_spinbox.setProperty("value", 0.01)
        self.dec_slope_bot_spinbox.setObjectName("dec_slope_bot_spinbox")
        self.gridLayout_10.addWidget(self.dec_slope_bot_spinbox, 2, 1, 1, 2)
        self.gridLayout_12.addWidget(self.groupBox_7, 2, 0, 1, 2)
        self.tabWidget.addTab(self.tab, "")
        self.gridLayout.addWidget(self.tabWidget, 1, 0, 2, 5)
        self.Tes_movement_pushButton = QtWidgets.QPushButton(advanced_parameters_Dialog)
        self.Tes_movement_pushButton.setStyleSheet("QPushButton {/* background-color:  rgb(55, 97, 102); border: none ; color: rgb(255, 210, 73)\n"
"*/}\n"
"QPushButton\n"
"{\n"
"    color: rgb(240, 240, 240);   \n"
" background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.946, fx:0.744, fy:0.159, stop:0 rgba(140, 150, 165, 255), stop:1 rgba(82, 98, 112, 255));\n"
"    border-width: 2px;\n"
"    border-color: #1e1e1e;\n"
"    border-style: solid;\n"
"    border-radius: 8px;\n"
"    padding: 3px;\n"
"    font-size: 16px;\n"
"    padding-left: 5px;\n"
"    padding-right: 5px;\n"
"}\n"
"\n"
"\n"
"\n"
"QPushButton:hover\n"
"{\n"
"    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #565656, stop: 0.1 #525252, stop: 0.5 #4e4e4e, stop: 0.9 #4a4a4a, stop: 1 #464646);\n"
"    border-width: 3px;\n"
"    border-color: rgb(42, 252, 253);\n"
"    border-style: solid;\n"
"    font-weight:bold;\n"
"}\n"
"\n"
"QPushButton:open\n"
"{\n"
"    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #2d2d2d, stop: 0.1 #2b2b2b, stop: 0.5 #292929, stop: 0.9 #282828, stop: 1 #252525);\n"
"    border-width: 3px;\n"
"    border-color: rgb(42, 252, 253);\n"
"    border-style: solid;\n"
"    font-weight:bold;\n"
"}")
        self.Tes_movement_pushButton.setObjectName("Tes_movement_pushButton")
        self.gridLayout.addWidget(self.Tes_movement_pushButton, 3, 3, 1, 1)

        self.retranslateUi(advanced_parameters_Dialog)
        self.tabWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(advanced_parameters_Dialog)

    def retranslateUi(self, advanced_parameters_Dialog):
        _translate = QtCore.QCoreApplication.translate
        advanced_parameters_Dialog.setWindowTitle(_translate("advanced_parameters_Dialog", "Hardware Parameters"))
        self.apply_pushButton.setText(_translate("advanced_parameters_Dialog", "Apply"))
        self.ok_pushButton.setText(_translate("advanced_parameters_Dialog", "Ok"))
        self.default_values_pushButton.setText(_translate("advanced_parameters_Dialog", "Default values"))
        self.fast_acquisition_pushButton.setText(_translate("advanced_parameters_Dialog", "Fast"))
        self.medium_acquisition_pushButton.setText(_translate("advanced_parameters_Dialog", "Medium"))
        self.slow_acquisition_pushButton.setText(_translate("advanced_parameters_Dialog", "Slow"))
        self.pushButton.setText(_translate("advanced_parameters_Dialog", "Calibration"))
        self.groupBox_5.setTitle(_translate("advanced_parameters_Dialog", "References Comparator"))
        self.label_21.setText(_translate("advanced_parameters_Dialog", "Ref 1:"))
        self.label_22.setText(_translate("advanced_parameters_Dialog", "Ref 2:"))
        self.ref2_lineedit.setText(_translate("advanced_parameters_Dialog", "100"))
        self.ref1_lineedit.setText(_translate("advanced_parameters_Dialog", "100"))
        self.label_73.setText(_translate("advanced_parameters_Dialog", "mV"))
        self.label_74.setText(_translate("advanced_parameters_Dialog", "mV"))
        self.number_turns_lineedit.setText(_translate("advanced_parameters_Dialog", "2"))
        self.groupBox_9.setTitle(_translate("advanced_parameters_Dialog", "References"))
        self.ref4_lineedit.setText(_translate("advanced_parameters_Dialog", "50"))
        self.label_23.setText(_translate("advanced_parameters_Dialog", "Ref 3:"))
        self.label_24.setText(_translate("advanced_parameters_Dialog", "Ref 4:"))
        self.label_75.setText(_translate("advanced_parameters_Dialog", "mV"))
        self.ref3_lineedit.setText(_translate("advanced_parameters_Dialog", "50"))
        self.label_76.setText(_translate("advanced_parameters_Dialog", "mV"))
        self.label_27.setText(_translate("advanced_parameters_Dialog", "Number of turns:"))
        self.groupBox_4.setTitle(_translate("advanced_parameters_Dialog", "MMPC\'s Voltage"))
        self.label_25.setText(_translate("advanced_parameters_Dialog", "Side A:"))
        self.mmpc_voltageA_lineedit.setText(_translate("advanced_parameters_Dialog", "74"))
        self.label_26.setText(_translate("advanced_parameters_Dialog", "Side B:"))
        self.mmpc_voltageB_lineedit.setText(_translate("advanced_parameters_Dialog", "74"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.crystals_number_tabcontrols), _translate("advanced_parameters_Dialog", "U-BOARD"))
        self.groupBox.setTitle(_translate("advanced_parameters_Dialog", "Movement"))
        self.label_realstep.setText(_translate("advanced_parameters_Dialog", "Real Step:"))
        self.label_19.setText(_translate("advanced_parameters_Dialog", "steps/s"))
        self.label_2.setText(_translate("advanced_parameters_Dialog", "Microstepping"))
        self.label_13.setText(_translate("advanced_parameters_Dialog", "DEC"))
        self.label_16.setText(_translate("advanced_parameters_Dialog", "<html><head/><body><p><span style=\" color:#526270;\">step/s</span><span style=\" color:#526270; vertical-align:super;\">2</span></p></body></html>"))
        self.label_9.setText(_translate("advanced_parameters_Dialog", "ACC"))
        self.label_18.setText(_translate("advanced_parameters_Dialog", "steps/s"))
        self.label_15.setText(_translate("advanced_parameters_Dialog", "steps/s"))
        self.microstepping_top_combobox.setItemText(0, _translate("advanced_parameters_Dialog", "None"))
        self.microstepping_top_combobox.setItemText(1, _translate("advanced_parameters_Dialog", "2"))
        self.microstepping_top_combobox.setItemText(2, _translate("advanced_parameters_Dialog", "4"))
        self.microstepping_top_combobox.setItemText(3, _translate("advanced_parameters_Dialog", "8"))
        self.microstepping_top_combobox.setItemText(4, _translate("advanced_parameters_Dialog", "16"))
        self.microstepping_top_combobox.setItemText(5, _translate("advanced_parameters_Dialog", "32"))
        self.microstepping_top_combobox.setItemText(6, _translate("advanced_parameters_Dialog", "64"))
        self.microstepping_top_combobox.setItemText(7, _translate("advanced_parameters_Dialog", "128"))
        self.label.setText(_translate("advanced_parameters_Dialog", "Full Step"))
        self.label_8.setText(_translate("advanced_parameters_Dialog", "Min Velocity"))
        self.label_17.setText(_translate("advanced_parameters_Dialog", "<html><head/><body><p><span style=\" color:#526270;\">step/s</span><span style=\" color:#526270; vertical-align:super;\">2</span></p></body></html>"))
        self.label_6.setText(_translate("advanced_parameters_Dialog", "Current Velocity"))
        self.label_7.setText(_translate("advanced_parameters_Dialog", "Max Velocity"))
        self.groupBox_3.setTitle(_translate("advanced_parameters_Dialog", "Slope"))
        self.label_4.setText(_translate("advanced_parameters_Dialog", "ACC Slope"))
        self.label_5.setText(_translate("advanced_parameters_Dialog", "DEC Slope"))
        self.label_3.setText(_translate("advanced_parameters_Dialog", "Start Slope"))
        self.label_31.setText(_translate("advanced_parameters_Dialog", "ms/step"))
        self.label_33.setText(_translate("advanced_parameters_Dialog", "ms/step"))
        self.label_32.setText(_translate("advanced_parameters_Dialog", "ms/step"))
        self.label_min_full_step.setText(_translate("advanced_parameters_Dialog", "Min Full Step: "))
        self.comboBox_chooseTopMotor.setItemText(0, _translate("advanced_parameters_Dialog", "QSH 4218"))
        self.comboBox_chooseTopMotor.setItemText(1, _translate("advanced_parameters_Dialog", "ST 4209"))
        self.groupBox_2.setTitle(_translate("advanced_parameters_Dialog", "Power"))
        self.label_29.setText(_translate("advanced_parameters_Dialog", "ACC starting KVal"))
        self.current_top_lineedit.setText(_translate("advanced_parameters_Dialog", "1.0"))
        self.label_28.setText(_translate("advanced_parameters_Dialog", "Constant Speed Kval"))
        self.label_11.setText(_translate("advanced_parameters_Dialog", "Current"))
        self.label_20.setText(_translate("advanced_parameters_Dialog", "Holding KVal"))
        self.phasevoltage_top_lineedit.setText(_translate("advanced_parameters_Dialog", "5.3"))
        self.label_10.setText(_translate("advanced_parameters_Dialog", "Voltage"))
        self.voltage_top_lineedit.setText(_translate("advanced_parameters_Dialog", "12"))
        self.label_12.setText(_translate("advanced_parameters_Dialog", "Phase Voltage"))
        self.label_30.setText(_translate("advanced_parameters_Dialog", "DEC starting KVal"))
        self.holding_kval_top_lineedit.setText(_translate("advanced_parameters_Dialog", "5.3"))
        self.constant_speed_top_lineedit.setText(_translate("advanced_parameters_Dialog", "5.3"))
        self.dec_top_lineedit.setText(_translate("advanced_parameters_Dialog", "5.3"))
        self.acc_top_lineedit.setText(_translate("advanced_parameters_Dialog", "5.3"))
        self.label_66.setText(_translate("advanced_parameters_Dialog", "V"))
        self.label_67.setText(_translate("advanced_parameters_Dialog", "A"))
        self.label_68.setText(_translate("advanced_parameters_Dialog", "V"))
        self.label_69.setText(_translate("advanced_parameters_Dialog", "V"))
        self.label_70.setText(_translate("advanced_parameters_Dialog", "V"))
        self.label_71.setText(_translate("advanced_parameters_Dialog", "V"))
        self.label_72.setText(_translate("advanced_parameters_Dialog", "V"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.crystals_time_tabcontrols), _translate("advanced_parameters_Dialog", "Top Motor"))
        self.label_min_bot_full_step.setText(_translate("advanced_parameters_Dialog", "Min Full Step: "))
        self.comboBox_chooseBotMotor.setItemText(0, _translate("advanced_parameters_Dialog", "ST5909L1008B"))
        self.groupBox_6.setTitle(_translate("advanced_parameters_Dialog", "Movement"))
        self.label_real_bot_step.setText(_translate("advanced_parameters_Dialog", "Real Step: 0.225º"))
        self.label_35.setText(_translate("advanced_parameters_Dialog", "Current Velocity"))
        self.label_36.setText(_translate("advanced_parameters_Dialog", "Full Step"))
        self.label_37.setText(_translate("advanced_parameters_Dialog", "ACC"))
        self.microstepping_bot_combobox.setItemText(0, _translate("advanced_parameters_Dialog", "None"))
        self.microstepping_bot_combobox.setItemText(1, _translate("advanced_parameters_Dialog", "2"))
        self.microstepping_bot_combobox.setItemText(2, _translate("advanced_parameters_Dialog", "4"))
        self.microstepping_bot_combobox.setItemText(3, _translate("advanced_parameters_Dialog", "8"))
        self.microstepping_bot_combobox.setItemText(4, _translate("advanced_parameters_Dialog", "16"))
        self.microstepping_bot_combobox.setItemText(5, _translate("advanced_parameters_Dialog", "32"))
        self.microstepping_bot_combobox.setItemText(6, _translate("advanced_parameters_Dialog", "64"))
        self.microstepping_bot_combobox.setItemText(7, _translate("advanced_parameters_Dialog", "128"))
        self.label_38.setText(_translate("advanced_parameters_Dialog", "Microstepping"))
        self.label_39.setText(_translate("advanced_parameters_Dialog", "Min Velocity"))
        self.label_40.setText(_translate("advanced_parameters_Dialog", "DEC"))
        self.label_41.setText(_translate("advanced_parameters_Dialog", "Max Velocity"))
        self.label_42.setText(_translate("advanced_parameters_Dialog", "steps/s"))
        self.label_43.setText(_translate("advanced_parameters_Dialog", "<html><head/><body><p><span style=\" color:#526270;\">step/s</span><span style=\" color:#526270; vertical-align:super;\">2</span></p></body></html>"))
        self.label_44.setText(_translate("advanced_parameters_Dialog", "<html><head/><body><p><span style=\" color:#526270;\">step/s</span><span style=\" color:#526270; vertical-align:super;\">2</span></p></body></html>"))
        self.label_45.setText(_translate("advanced_parameters_Dialog", "steps/s"))
        self.label_46.setText(_translate("advanced_parameters_Dialog", "steps/s"))
        self.groupBox_8.setTitle(_translate("advanced_parameters_Dialog", "Power"))
        self.label_54.setText(_translate("advanced_parameters_Dialog", "ACC starting KVal"))
        self.label_53.setText(_translate("advanced_parameters_Dialog", "Constant Speed Kval"))
        self.phasevoltage_bot_lineedit.setText(_translate("advanced_parameters_Dialog", "5.3"))
        self.current_bot_lineedit.setText(_translate("advanced_parameters_Dialog", "1.0"))
        self.label_55.setText(_translate("advanced_parameters_Dialog", "Holding KVal"))
        self.label_57.setText(_translate("advanced_parameters_Dialog", "Current"))
        self.label_58.setText(_translate("advanced_parameters_Dialog", "Phase Voltage"))
        self.voltage_bot_lineedit.setText(_translate("advanced_parameters_Dialog", "12"))
        self.label_56.setText(_translate("advanced_parameters_Dialog", "Voltage"))
        self.acc_bot_lineedit.setText(_translate("advanced_parameters_Dialog", "5.3"))
        self.label_59.setText(_translate("advanced_parameters_Dialog", "DEC starting KVal"))
        self.constant_speed_bot_lineedit.setText(_translate("advanced_parameters_Dialog", "5.3"))
        self.holding_kval_bot_lineedit.setText(_translate("advanced_parameters_Dialog", "5.3"))
        self.dec_bot_lineedit.setText(_translate("advanced_parameters_Dialog", "5.3"))
        self.label_14.setText(_translate("advanced_parameters_Dialog", "V"))
        self.label_60.setText(_translate("advanced_parameters_Dialog", "A"))
        self.label_61.setText(_translate("advanced_parameters_Dialog", "V"))
        self.label_62.setText(_translate("advanced_parameters_Dialog", "V"))
        self.label_63.setText(_translate("advanced_parameters_Dialog", "V"))
        self.label_64.setText(_translate("advanced_parameters_Dialog", "V"))
        self.label_65.setText(_translate("advanced_parameters_Dialog", "V"))
        self.groupBox_7.setTitle(_translate("advanced_parameters_Dialog", "Slope"))
        self.label_47.setText(_translate("advanced_parameters_Dialog", "DEC Slope"))
        self.label_48.setText(_translate("advanced_parameters_Dialog", "ACC Slope"))
        self.label_49.setText(_translate("advanced_parameters_Dialog", "Start Slope"))
        self.label_50.setText(_translate("advanced_parameters_Dialog", "ms/step"))
        self.label_51.setText(_translate("advanced_parameters_Dialog", "ms/step"))
        self.label_52.setText(_translate("advanced_parameters_Dialog", "ms/step"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("advanced_parameters_Dialog", "Bot Motor"))
        self.Tes_movement_pushButton.setText(_translate("advanced_parameters_Dialog", "Test"))

