import os
# from dialog_change_hardware_individual_info_gui import Ui_Dialog as DialogChangeHardware_parameters
from PyQt5.QtQml import qmlRegisterType, QQmlComponent, QQmlEngine, QQmlError
from PyQt5.QtCore import QUrl, Qt, pyqtProperty, pyqtSignal, QObject,QStringListModel, \
    QAbstractListModel, QModelIndex, QAbstractTableModel,QPointF,QRegExp,QLocale, QVariant
from PyQt5.QtQuick import QQuickView
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QColor, QRadialGradient, QIntValidator,QRegExpValidator
import sys
from array import array
from hardware_parameters_file import EasypetVersionHardware
import numpy as np
#from dialogroundedprogressbar import Dialog_progressbar
# from GUI_t import Ui_MainWindow_easypet_client


class DialogChangeHardware(QtWidgets.QDialog,DialogChangeHardware_parameters,Ui_MainWindow_easypet_client):
    def __init__(self, parent=None, serial_number='training_0000', main_window_buttons = None):
        QtWidgets.QDialog.__init__(self, parent)
        main_window_buttons.clicked.connect(self._open_dialog)
        # READING HARDWARE Parameters
        QLocale.setDefault(QLocale(QLocale.Portuguese, QLocale.Brazil))
        self.serial_number = serial_number
        self.setupUi(self)
        self.quickWidget.setWindowFlags(Qt.FramelessWindowHint)
        self.quickWidget.setAttribute(Qt.WA_TranslucentBackground, True)
        self.quickWidget.setClearColor(Qt.transparent)
        self.quickWidget.setAttribute(Qt.WA_AlwaysStackOnTop)
        serial_number_override_class = OverrideLineedit(self.serial_number_lineEdit, model_combobox=self.model_comboBox)
        self.serial_number_lineEdit = serial_number_override_class.lineedit
        self.comboboxes = [self.model_comboBox, self.control_module_version_comboBox, self.scanning_module_comboBox,
                      self.resistive_chain_type_comboBox, self.bed_version_comboBox, self.type_connection_comboBox,
                      self.baudrate_comboBox, self.botmotor_comboBox, self.topmotor_comboBox, self.bed_motor_comboBox,
                      self.fourth_motor_comboBox]

        self.plus_buttons = [self.add_model_toolButton, self.add_controlmodule_toolButton,
                             self.add_scanningmodule_toolButton,
                             self.add_resistive_chaintype_toolButton, self.add_bed_version_toolButton,
                             self.add_type_connection_toolButton,
                             self.add_baudrate_toolButton, self.add_botmotor_toolButton, self.add_topmotor_toolButton,
                             self.add_bedmotor_toolButton, self.add_fourthmotor_toolButton]

        self.minus_buttons = [self.remove_model_toolButton, self.remove_controlmodule_toolButton,
                              self.remove_scanningmodule_toolButton,
                              self.remove_resistive_chaintype_toolButton, self.remove_bed_version_toolButton,
                              self.remove_type_connection_toolButton,
                              self.remove_baudrate_toolButton, self.remove_botmotor_toolButton,
                              self.remove_topmotor_toolButton,
                              self.remove_bedmotor_toolButton, self.remove_fourthmotor_toolButton]

        self.list_editline_from_combobox = []
        self._fill_combobox_data()
        self._init_values_from_binary_file()
        self.number_of_crystals = self.crystals_matrix[0]*self.crystals_matrix[1]
        # Real Size Crystal
        [self.model_crystals_array_A, total_array_size_x, size_reduction] = self._setModel(self.real_size_crystal_dimensions[0:6])

        self.quickWidget.rootContext().setContextProperty("model_crystals_array_A", self.model_crystals_array_A)        # [model_crystals_array_B, total_array_size_x_B, size_reduction_B] = self._setModel(
        self.quickWidget.rootContext().setContextProperty("model_crystals_array_B", self.model_crystals_array_A)

        self.quickWidget.setSource(QUrl('crystal_matrix_generator.qml'))
        self._update_model_sizes(total_array_size_x, size_reduction)
        # load parameters to interface
        self._upload_info_gui()
        self.__init_signals()

    def _init_values_from_binary_file(self):
        reading_hardware_parameters = EasypetVersionHardware(operation='read_file', serial_number=self.serial_number)
        self.number_of_parameters_easpet_harwarefile = reading_hardware_parameters.interval_index
        self.module_control = reading_hardware_parameters.angle_top_correction
        self.u_board_version = reading_hardware_parameters.u_board_version
        self.crystals_matrix = [None] * 2
        self.crystals_matrix[0] = reading_hardware_parameters.array_crystal_x
        self.crystals_matrix[1] = reading_hardware_parameters.array_crystal_y
        self.angle_bot_rotation = reading_hardware_parameters.angle_bot_rotation
        self.angle_top_correction = reading_hardware_parameters.angle_top_correction
        self.multiplexed = reading_hardware_parameters.multiplexed
        self.reading_method = reading_hardware_parameters.reading_method
        self.number_ADC_channels = reading_hardware_parameters.number_adc_channel
        self.bed_version = reading_hardware_parameters.angle_top_correction
        self.bed_diameter = reading_hardware_parameters.bed_diameter
        self.pc_communication = reading_hardware_parameters.pc_communication
        self.baudrate = reading_hardware_parameters.angle_top_correction
        self.motor_top = reading_hardware_parameters.motor_top
        self.motor_bot = reading_hardware_parameters.motor_bot
        self.bed_motor = reading_hardware_parameters.bed_motor
        self.fourth_motor = reading_hardware_parameters.fourth_motor
        self.capable4CT = reading_hardware_parameters.capable4CT
        r = reading_hardware_parameters
        self.real_size_crystal_dimensions = [r.crystal_pitch_x, r.crystal_pitch_y, r.crystal_length,
                                             r.reflector_exterior_thic, r.reflector_interior_A_x,
                                             r.reflector_interior_A_y, r.reflector_interior_B_x,
                                             r.reflector_interior_B_y]
        # mm [pitch_x, pitch_y, length_crystal(z), reflector_exterior, reflector_interior_x, reflector_interior_y]
        self.distance_between_motors = r.distance_between_motors  # mm
        self.detectors_distances = [r.distance_between_crystals, r.centercrystals2topmotor_y,
                                    r.centercrystals2topmotor_x_sideA, r.centercrystals2topmotor_x_sideB]  # [distances between crystals, top motor axis to the center of the crystals
        # side A to top motor, side B to top motor]
        # self.model_comboBox.setFocus(True)

    def __init_signals(self):
        self.override_values_toolButton.clicked.connect(self._override_values)
        self.defaultdata_toolButton.clicked.connect(self._set_default_values)
        self.serial_number_lineEdit.editingFinished.connect(self._serial_number_changed)

        for button in  range(len(self.plus_buttons)):
            self.plus_buttons[button].clicked.connect(self._add_info_combobox)
            self.minus_buttons[button].clicked.connect(self._remove_info_from_dialog_data_file)

        self.number_crystals_x_spinBox.valueChanged.connect(self._matrix_crystals_changed)
        self.number_crystals_y_spinBox.valueChanged.connect(self._matrix_crystals_changed)
        self.crystal_names_widgets = ['pitch_X_crystal', 'pitch_Y_crystal', 'lenght_crystal',
                                      'reflector_crystal_exterior',
                                      'reflector_crystal_interior_A_X', 'reflector_crystal_interior_A_Y',
                                      'reflector_crystal_interior_B_X', 'reflector_crystal_interior_B_Y']

        self.dimensions_text_input = [None]*len(self.crystal_names_widgets)

        for i in range(len(self.crystal_names_widgets)):
            self.dimensions_text_input[i] = self.quickWidget.rootObject().findChild(QObject, self.crystal_names_widgets[i])
            self.dimensions_text_input[i].setProperty('text', '{} mm'.format(self.real_size_crystal_dimensions[i]))
            self.dimensions_text_input[i].update_size_crystal.connect(lambda: self._model_datachanged())

    def _upload_info_gui(self, init=None):
        self.number_crystals_x_spinBox.blockSignals(True)
        self.number_crystals_y_spinBox.blockSignals(True)
        self.serial_number_lineEdit.setText(self.serial_number)
        model_easypet = self.serial_number.split('_')
        model_easypet = model_easypet[0]
        models_types = ['training', 'preclinical']
        model_choosed = models_types.index(model_easypet)
        if model_choosed is not None:
            self.model_comboBox.setCurrentIndex(model_choosed)

        self.number_crystals_x_spinBox.setValue(self.crystals_matrix[0])
        self.number_crystals_y_spinBox.setValue(self.crystals_matrix[1])
        self.rotationbot_angle_doubleSpinBox.setValue(float(self.angle_bot_rotation))
        self.init_angle_top_motor_doubleSpinBox.setValue(float(self.angle_top_correction))
        if self.multiplexed == "yes":
            self.yes_multiplexed_radioButton.setChecked(True)
        else:
            self.no_multiplexed_radioButton.setChecked(True)

        # Resistive Chain Type
        resistive_chains_types = [None]*self.resistive_chain_type_comboBox.count()
        for count in range(self.resistive_chain_type_comboBox.count()):
            resistive_chains_types[count] = self.resistive_chain_type_comboBox.itemText(count)
        try:
            self.resistive_chain_type_comboBox.setCurrentIndex(resistive_chains_types.index(self.reading_method))
        except ValueError:
            self.resistive_chain_type_comboBox.addItem(self.reading_method)
            self.resistive_chain_type_comboBox.setCurrentIndex(self.resistive_chain_type_comboBox.count()-1)
        # ADC's Channel number
        self.numberofADC_spinBox.setValue(self.number_ADC_channels)
        # Bed type
        bed_types = [None] * self.bed_version_comboBox.count()
        for count in range(self.bed_version_comboBox.count()):
            bed_types[count] = self.bed_version_comboBox.itemText(count)

        try:
            self.bed_version_comboBox.setCurrentIndex(bed_types.index(self.bed_version))
        except ValueError:
            self.bed_version_comboBox.setCurrentIndex(0)
        # Bed diameter
        self.bed_diameter_doubleSpinBox.setValue(self.bed_diameter)

        # Communication easypet to pc
        pc_comunication_types = [None] * self.type_connection_comboBox.count()
        for count in range(self.type_connection_comboBox.count()):
            pc_comunication_types[count] = self.type_connection_comboBox.itemText(count)

        try:
            self.type_connection_comboBox.setCurrentIndex(pc_comunication_types.index(self.pc_communication))
        except ValueError:
            self.type_connection_comboBox.setCurrentIndex(0)

        baudrate_types = [None] * self.baudrate_comboBox.count()
        for count in range(self.baudrate_comboBox.count()):
            baudrate_types[count] = self.baudrate_comboBox.itemText(count)
        try:
            self.baudrate_comboBox.setCurrentIndex(baudrate_types.index(self.baudrate))
        except ValueError:
            self.baudrate_comboBox.setCurrentIndex(0)

        motor_types = [None] * self.botmotor_comboBox.count()
        for count in range(self.botmotor_comboBox.count()):
            motor_types[count] = self.botmotor_comboBox.itemText(count)
        try:
            self.botmotor_comboBox.setCurrentIndex(motor_types.index(self.motor_bot))
        except ValueError:
            self.botmotor_comboBox.setCurrentIndex(0)

        motor_types = [None] * self.topmotor_comboBox.count()
        for count in range(self.topmotor_comboBox.count()):
            motor_types[count] = self.topmotor_comboBox.itemText(count)
        try:
            self.topmotor_comboBox.setCurrentIndex(motor_types.index(self.motor_top))
        except ValueError as e:
            print(e)
            self.topmotor_comboBox.setCurrentIndex(0)

        motor_types = [None] * self.bed_motor_comboBox.count()
        for count in range(self.bed_motor_comboBox.count()):
            motor_types[count] = self.bed_motor_comboBox.itemText(count)
        try:
            self.bed_motor_comboBox.setCurrentIndex(motor_types.index(self.bed_motor))
        except ValueError:
            self.bed_motor_comboBox.setCurrentIndex(0)

        motor_types = [None] * self.fourth_motor_comboBox.count()
        for count in range(self.fourth_motor_comboBox.count()):
            motor_types[count] = self.fourth_motor_comboBox.itemText(count)
        try:
            self.fourth_motor_comboBox.setCurrentIndex(motor_types.index(self.fourth_motor))
        except ValueError:
            self.fourth_motor_comboBox.setCurrentIndex(0)

        if self.capable4CT == "yes":
            self.yes_ct_radioButton.setChecked(True)
        else:
            self.no_ct_radioButton.setChecked(True)

        distances_widgets_qml_names = ['spinBox_distance_between_crystals', 'sliderHorizontal_motor_axis_x',
                                       'sliderHorizontal_motor_axis_y', 'spinBox_distance_between_motors' ] # the values introduced are in mm -- 10
        key_word = ['value', 'value', 'value', 'value']
        try:
            key_value = [self.detectors_distances[0], self.detectors_distances[1],
                         self.detectors_distances[2]/self.detectors_distances[0],
                         self.distance_between_motors]
        except ZeroDivisionError:
            key_value = [self.detectors_distances[0], self.detectors_distances[1],
                         0
                ,
                         self.distance_between_motors]
        # will appear 1.0 cm in qlm
        for i in range(len(distances_widgets_qml_names)):
            widget_qml = self.quickWidget.rootObject().findChild(QObject, distances_widgets_qml_names[i])
            widget_qml.setProperty(key_word[i], key_value[i])

        if init is not None:

            self.crystal_names_widgets = ['pitch_X_crystal', 'pitch_Y_crystal', 'lenght_crystal',
                                          'reflector_crystal_exterior',
                                          'reflector_crystal_interior_A_X', 'reflector_crystal_interior_A_Y',
                                          'reflector_crystal_interior_B_X', 'reflector_crystal_interior_B_Y']

            self.dimensions_text_input = [None] * len(self.crystal_names_widgets)

            for i in range(len(self.crystal_names_widgets)):
                self.dimensions_text_input[i] = self.quickWidget.rootObject().findChild(QObject,
                                                                                        self.crystal_names_widgets[i])
                self.dimensions_text_input[i].setProperty('text', '{} mm'.format(self.real_size_crystal_dimensions[i]))
        # Dimensions of the crystals are being updated on init_signals function.
            # self.real_size_crystal_dimensions = [2, 2, 30, 0.28, 0.28, 0.175, 0.28,
            # 0.175]  # mm [pitch_x, pitch_y, length_crystal(z), reflector_exterior,
            #  reflector_interior_x, reflector_interior_y]
        print(key_value)
        self.number_crystals_x_spinBox.blockSignals(False)
        self.number_crystals_y_spinBox.blockSignals(False)

    def _get_data_from_dialog(self):
        list_2_record = [None]*(self.number_of_parameters_easpet_harwarefile-1)
        #list_2_record[0] = self.model_comboBox.currentText()
        list_2_record[0] = self.serial_number_lineEdit.text()
        list_2_record[1] = self.control_module_version_comboBox.currentText()
        list_2_record[2] = self.scanning_module_comboBox.currentText()
        list_2_record[3] = str(self.number_crystals_x_spinBox.value())
        list_2_record[4] = str(self.number_crystals_y_spinBox.value())
        list_2_record[5] = str(self.rotationbot_angle_doubleSpinBox.value())
        list_2_record[6] = str(self.init_angle_top_motor_doubleSpinBox.value())
        if self.yes_multiplexed_radioButton.isChecked():
            self.multiplexed = 'yes'
        else:
            self.multiplexed = 'no'
        list_2_record[7] = self.multiplexed
        list_2_record[8] = self.resistive_chain_type_comboBox.currentText()
        list_2_record[9] = str(self.numberofADC_spinBox.value())
        list_2_record[10] = self.bed_version_comboBox.currentText()
        list_2_record[11] = str(self.bed_diameter_doubleSpinBox.value())
        list_2_record[12] = self.type_connection_comboBox.currentText()
        list_2_record[13] = self.baudrate_comboBox.currentText()
        list_2_record[14] = self.botmotor_comboBox.currentText()
        list_2_record[15] = self.topmotor_comboBox.currentText()
        list_2_record[16] = self.bed_motor_comboBox.currentText()
        list_2_record[17] = self.fourth_motor_comboBox.currentText()
        if self.yes_ct_radioButton.isChecked():
            self.capable4CT == 'yes'
        else:
            self.capable4CT == 'no'
        list_2_record[18] = self.capable4CT
        for i in range(len(self.crystal_names_widgets)):
            p = self.quickWidget.rootObject().findChild(QObject, self.crystal_names_widgets[i])
            text = p.property('text')
            try:
                text = text.split(' ')
                text = text[0]
                self.real_size_crystal_dimensions[i] = float(text.replace(',','.'))
                self.real_size_crystal_dimensions[i] = str(self.real_size_crystal_dimensions[i])

            except ValueError:
                p.setProperty('text', 'error')
                ### open a dialog here

        distances_widgets_qml_names = ['spinBox_distance_between_crystals', 'sliderHorizontal_motor_axis_x',
                                       'sliderHorizontal_motor_axis_y',
                                       'spinBox_distance_between_motors']  # the values introduced are in mm -- 10
        key_word = ['value', 'value', 'value', 'value']
        key_value = [None]*len(distances_widgets_qml_names)
        # will appear 1.0 cm in qlm
        for i in range(len(distances_widgets_qml_names)):
            widget_qml = self.quickWidget.rootObject().findChild(QObject, distances_widgets_qml_names[i])
            key_value[i] = widget_qml.property(key_word[i])


        self.distance_between_motors = str(key_value[3])  # mm
        self.detectors_distances = [key_value[0], key_value[1],key_value[0]*key_value[2] , key_value[0]*(1-key_value[2])]
        self.detectors_distances = [str(i) for i in self.detectors_distances]
        start_index = 19
        end_index = start_index+len(self.real_size_crystal_dimensions)
        list_2_record[start_index:end_index] = self.real_size_crystal_dimensions
        start_index = end_index
        list_2_record[start_index] = self.distance_between_motors
        start_index += 1
        end_index = start_index + len(self.detectors_distances)
        list_2_record[start_index:end_index] = self.detectors_distances
        return list_2_record

    def _setModel(self, real_size_crystal_dimensions):
        maximum_size_component = [400, 220]
        # [2, 2, 30, 0.28, 0.28,0.175,0.28,0.175]
        total_array_size_x = real_size_crystal_dimensions[0] * self.crystals_matrix[1] + \
                             (self.crystals_matrix[1] - 1) * real_size_crystal_dimensions[5] \
                             + 2 * real_size_crystal_dimensions[3]  # mm

        total_array_size_y = real_size_crystal_dimensions[1] * self.crystals_matrix[0] +\
                             (self.crystals_matrix[0] - 1) * real_size_crystal_dimensions[4] \
                             + 2 * real_size_crystal_dimensions[3]  # mm # mm

        size_reduction_matrix = [maximum_size_component[0] / total_array_size_y,
                                 maximum_size_component[1] / total_array_size_x]  # [height, width]
        
        if size_reduction_matrix[0]>size_reduction_matrix[1]:
            size_reduction = [size_reduction_matrix[1]]
        else:
            size_reduction = [size_reduction_matrix[0]]

        # Graphical part size -- size adjust to height
        pitch_x_crystal_size = real_size_crystal_dimensions[0] * size_reduction[0]
        pitch_y_crystal_size = real_size_crystal_dimensions[1] * size_reduction[0]
        thickness_reflector_exterior_crystal = real_size_crystal_dimensions[3] * size_reduction[0]
        thickness_reflector_interior_crystal_y = real_size_crystal_dimensions[5] * size_reduction[0]
        thickness_reflector_interior_crystal_x = real_size_crystal_dimensions[4] * size_reduction[0]
        length_reflector = pitch_x_crystal_size
        depth_reflector = pitch_y_crystal_size
        corner_reflector_exterior = thickness_reflector_exterior_crystal
        corner_reflector_interior_y = thickness_reflector_interior_crystal_y
        corner_reflector_interior_x = thickness_reflector_interior_crystal_x

        model_crystals_array = Model()
        exterior_color_reflector = 'grey'
        interior_color_reflector = 'grey'
        crystal_color = '#bfbfbf'

        for i in range(self.crystals_matrix[0] * 3):
            for j in range(self.crystals_matrix[1]):

                if j == 0 and (i == 0 or i == self.crystals_matrix[0] * 3 - 1):

                    model_crystals_array.addData(
                        Data(corner_reflector_exterior, corner_reflector_exterior, exterior_color_reflector))
                    model_crystals_array.addData(
                        Data(length_reflector, thickness_reflector_exterior_crystal, exterior_color_reflector))
                    model_crystals_array.addData(
                        Data(corner_reflector_interior_y, corner_reflector_exterior, exterior_color_reflector))

                # elif j == 0  and (i != 0 or i !=self.crystals_matrix[0]-1):

                elif j == self.crystals_matrix[1] - 1 and (i == 0 or i == self.crystals_matrix[0] * 3 - 1):

                    model_crystals_array.addData(
                        Data(corner_reflector_interior_y, corner_reflector_exterior, exterior_color_reflector))
                    model_crystals_array.addData(
                        Data(length_reflector, thickness_reflector_exterior_crystal, exterior_color_reflector))
                    model_crystals_array.addData(
                        Data(corner_reflector_exterior, corner_reflector_exterior, exterior_color_reflector))

                elif (j != 0) and (i == 0 or i == self.crystals_matrix[0] * 3 - 1):

                    model_crystals_array.addData(
                        Data(corner_reflector_interior_y, corner_reflector_exterior, interior_color_reflector))
                    model_crystals_array.addData(
                        Data(length_reflector, thickness_reflector_exterior_crystal, interior_color_reflector))

                    model_crystals_array.addData(
                        Data(corner_reflector_interior_y, corner_reflector_exterior, interior_color_reflector))

                elif (i - 1) % 3 == 0:

                    if (j == 0):
                        corner_left = corner_reflector_exterior
                        corner_right = corner_reflector_interior_y
                    elif j == self.crystals_matrix[1] - 1:
                        corner_left = corner_reflector_interior_y
                        corner_right = corner_reflector_exterior
                    else:
                        corner_left = corner_reflector_interior_y
                        corner_right = corner_reflector_interior_y
                    model_crystals_array.addData(
                        Data(corner_left, pitch_y_crystal_size, interior_color_reflector))
                    # model_crystals_array.addData(
                    #     Data(pitch_x_crystal_size, pitch_y_crystal_size, crystal_color,
                    #          str(int(round(i / 3, 0) + self.crystals_matrix[0] * j))))
                    model_crystals_array.addData(
                        Data(pitch_x_crystal_size, pitch_y_crystal_size, crystal_color,
                             str(self.number_of_crystals-int(round(i / 3, 0))*self.crystals_matrix[1]-j )))
                    model_crystals_array.addData(
                        Data(corner_right, pitch_y_crystal_size, interior_color_reflector))
                # #
                elif i % 3 == 0 and (i != 0 or i != self.crystals_matrix[0] - 1):
                    if (j == 0):
                        corner_left = corner_reflector_exterior
                        corner_right = corner_reflector_interior_y
                    elif j == self.crystals_matrix[1] - 1:
                        corner_left = corner_reflector_interior_y
                        corner_right = corner_reflector_exterior
                    else:
                        corner_left = corner_reflector_interior_y
                        corner_right = corner_reflector_interior_y
                    model_crystals_array.addData(
                        Data(corner_left, thickness_reflector_interior_crystal_x, interior_color_reflector))
                    model_crystals_array.addData(
                        Data(pitch_x_crystal_size, thickness_reflector_interior_crystal_x, interior_color_reflector))
                    model_crystals_array.addData(
                        Data(corner_right, thickness_reflector_interior_crystal_x, interior_color_reflector))


        return  model_crystals_array, total_array_size_x, size_reduction

    def _update_model_sizes(self, total_array_size_x, size_reduction):
        slider_x = self.quickWidget.rootObject().findChild(QObject, 'sliderHorizontal_motor_axis_x')
        slider_x.setProperty('width', total_array_size_x * size_reduction[0])
        slider_x.setProperty('from', -(total_array_size_x) / 2)
        slider_x.setProperty('to', (total_array_size_x) / 2)
        slider_x.setProperty('value', self.detectors_distances[1])

        crystal_grid = self.quickWidget.rootObject().findChild(QObject, 'crystalgridA')
        crystal_grid.setProperty('columns', self.crystals_matrix[1] * 3)
        crystal_grid.setProperty('width', total_array_size_x * size_reduction[0])
        crystal_grid_B = self.quickWidget.rootObject().findChild(QObject, 'crystalgridB')
        crystal_grid_B.setProperty('columns', self.crystals_matrix[1] * 3)
        crystal_grid_B.setProperty('width', total_array_size_x * size_reduction[0])

        spin_box_between_crystals = self.quickWidget.rootObject().findChild(QObject,
                                                                            'spinBox_distance_between_crystals')
        spinbox_width = 260 - total_array_size_x * size_reduction[0]
        spinbox_height = 40
        if spinbox_width < 120:
            spinbox_width = 120

        spin_box_between_crystals.setProperty('width', spinbox_width)
        spin_box_between_crystals.setProperty('height', spinbox_height)

    def _matrix_crystals_changed(self):
        print('Crystals CHANGED')
        self.crystals_matrix[0] = self.number_crystals_x_spinBox.value()
        self.crystals_matrix[1] = self.number_crystals_y_spinBox.value()
        self._model_datachanged()

    def _model_datachanged(self):
        print('MODEL CHANGED')


        self.number_of_crystals = self.crystals_matrix[0]*self.crystals_matrix[1]
        for i in range(len(self.crystal_names_widgets)):
            p = self.quickWidget.rootObject().findChild(QObject, self.crystal_names_widgets[i])
            text = p.property('text')
            try:
                text = text.split(' ')
                text = text[0]
                self.real_size_crystal_dimensions[i] = float(text.replace(',','.'))
            except ValueError:
                p.setProperty('text', 'error')

        [self.model_crystals_array_A, total_array_size_x, size_reduction] = self._setModel(
            self.real_size_crystal_dimensions[0:6])

        self.quickWidget.rootContext().setContextProperty("model_crystals_array_A", self.model_crystals_array_A)
        self.quickWidget.rootContext().setContextProperty("model_crystals_array_B", self.model_crystals_array_A)
        self._update_model_sizes(total_array_size_x, size_reduction)

    def _serial_number_changed(self):
        self.serial_number = self.serial_number_lineEdit.text()
        self._set_default_values()

    def _add_info_combobox(self):
        ''' alow to add new information to the comboboxes'''
        sender = self.sender()
        button = self.plus_buttons.index(sender)
        if not self.comboboxes[button].isEditable():
            self.comboboxes[button].setEditable(True)
            lineEdit_combobox = self.comboboxes[button].lineEdit()
            self.list_editline_from_combobox.extend([lineEdit_combobox])
            self.list_editline_from_combobox.extend([self.comboboxes[button]])
            lineEdit_combobox.returnPressed.connect(self._add_info_to_dialog_data_file)
            print('Add')

    def _add_info_to_dialog_data_file(self):
        sender = self.sender()
        index = self.list_editline_from_combobox.index(sender)
        self.list_editline_from_combobox[index+1].setEditable(False)
        all_info_comboboxes = self.info_comboboxes_to_list()
        InformationWidgets(operation='w', list2record=all_info_comboboxes)

        print('add to file')

    def _remove_info_from_dialog_data_file(self):
        sender = self.sender()
        button = self.minus_buttons.index(sender)
        self.comboboxes[button].removeItem(self.comboboxes[button].currentIndex())
        all_info_comboboxes = self.info_comboboxes_to_list()
        InformationWidgets(operation='w', list2record=all_info_comboboxes)
        print('remove to file')

    def info_comboboxes_to_list(self):
        comboboxes = self.comboboxes
        all_info_comboboxes =[None]*len(comboboxes)
        for combobox in range(len(comboboxes)):
            info_combobox = [None] * comboboxes[combobox].count()
            for i in range(comboboxes[combobox].count()):
                info_combobox[i] = comboboxes[combobox].itemText(i)
            all_info_comboboxes[combobox] = info_combobox

        return all_info_comboboxes

    def _fill_combobox_data(self):
        """Populates combobox with the different available versions"""

        info_file = InformationWidgets(operation='r')
        all_info_comboboxes = info_file.data_fromfile
        for i in range(len(self.comboboxes)):
            info2combobox = all_info_comboboxes[i].split(';')
            info2combobox = info2combobox[2:-1]
            self.comboboxes[i].addItems(info2combobox)

    def _set_default_values(self):
        self._init_values_from_binary_file()
        self._upload_info_gui(init=False)
        self._model_datachanged()

    def _override_values(self):
        """ Override file with the hardware specifications
        :return: """
        directory = os.path.dirname(os.path.abspath(__file__))
        file_path = '{}/calibrationdata/x_{}__y_{}'.format(directory, self.crystals_matrix[0], self.crystals_matrix[1])
        try:
            MatrixGeometryCorrection(operation='w',crystals_dimensions=self.real_size_crystal_dimensions,
                                     crystal_matrix=self.crystals_matrix,
                                     distance_between_motors=self.distance_between_motors,
                                     detectors_distances=self.detectors_distances, file_path=file_path)
        except FileNotFoundError:
            warning_msg = 'Was detected a new crystal geometry. '
            QtWidgets.QMessageBox.question(None, 'Create folder calibration', warning_msg,
                                          QtWidgets.QMessageBox.Discard, QtWidgets.QMessageBox.Apply)

            CreateDummyCalibrationData(operation = 'w', crystal_matrix=self.crystals_matrix, directory =file_path)
            MatrixGeometryCorrection(crystals_dimensions=self.real_size_crystal_dimensions,
                                     crystal_matrix=self.crystals_matrix,
                                     distance_between_motors=self.distance_between_motors,
                                     detectors_distances=self.detectors_distances, file_path=file_path)

        record_edited_list = self._get_data_from_dialog()

        EasypetVersionHardware(operation='edit_file', serial_number = self.serial_number,
                               updated_info=record_edited_list)

        print('override values')

    def _open_dialog(self):
        self.show()


class OverrideLineedit(QtWidgets.QLineEdit):
    def __init__(self, lineedit, model_combobox):
        validator = QIntValidator(0,9999)
        self.lineedit = lineedit
        self.model_combobox = model_combobox
        self.lineedit.setValidator(validator)
        self.lineedit.returnPressed.connect(lambda: self._datachanged())
        self.lineedit.selectionChanged.connect(lambda: self._datarejected())

    def _datachanged(self):
        string_model = self.model_combobox.currentText()
        lineedit_text = self.lineedit.text()
        masktext = '{}_{num:{fill}{width}}'.format(string_model,num=lineedit_text, fill='0>', width=4)
        self.lineedit.setText(masktext)

    def _datarejected(self):
        self.lineedit.clear()


class MatrixGeometryCorrection:
    def __init__(self, operation = 'r', crystals_dimensions = None, crystal_matrix = None, distance_between_motors = None, detectors_distances =None, file_path = None):
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            raise FileNotFoundError

        if operation == 'w':
            if (crystals_dimensions or crystal_matrix or distance_between_motors or detectors_distances) is None:
                return
            self.real_size_crystal_dimensions = crystals_dimensions
            self.crystal_matrix = crystal_matrix
            self.distance_between_motors = distance_between_motors
            self.numberofcrystals = crystal_matrix[0] * crystal_matrix[1]
            self.detectors_distances = detectors_distances

            #directory = os.path.dirname(os.path.abspath(__file__))
            #self.file_path = '{}/calibrationdata/x_{}__y_{}'.format(directory,self.crystal_matrix[0], self.crystal_matrix[1])

            # self.real_size_crystal_dimensions = [2, 2, 30, 0.28, 0.14, 0.175, 0.14,0.175]  # mm [pitch_x, pitch_y, length_crystal(z), reflector_exterior, reflector_interior_x, reflector_interior_y]
            # self.distance_between_motors = 60  # mm
            # self.detectors_distances = [60, 0.1, 0, 60]
            self._generate_matrix()
            self._write()
        elif operation == 'r':
            self._read()

    def _write(self):
        print('writing')
        file_name = self.file_path + '\\Geometry matrix.geo'
        np.savetxt(file_name, self.coordinates, delimiter=",")

    def _read(self):
        print('Reading GEOMETRY FiLE')
        file_name = self.file_path + '\\Geometry matrix.geo'
        self.coordinates = np.loadtxt(file_name, delimiter=',')

    def _generate_matrix(self):
        """ all distances are calculated to the center of the FOV ---- its consider that the system only has 2 blocks
        of crystals one of each side"""
        self.coordinates = np.zeros((self.numberofcrystals*2, 3))

        for side in range(2):
            reflector_depth_between_crystals_x = self.real_size_crystal_dimensions[4 + 2 * side]
            reflector_depth_between_crystals_y = self.real_size_crystal_dimensions[5 + 2 * side]
            additional_increment_y = self.real_size_crystal_dimensions[1] / 2 + reflector_depth_between_crystals_y  # Side A --- espessura interior y [4]  Side B ---- [6]
            total_size_array_Z = self.real_size_crystal_dimensions[1]*self.crystal_matrix[0]+(2*reflector_depth_between_crystals_x)*(self.crystal_matrix[0]-1)
            if side == 0:
                crystal_2_center_fov = -(self.detectors_distances[2] + self.distance_between_motors+self.real_size_crystal_dimensions[2]/2)
            elif side == 1:
                crystal_2_center_fov = self.detectors_distances[3] - self.distance_between_motors + self.real_size_crystal_dimensions[2]/2
            for i in range(self.crystal_matrix[0]):
                for j in range(self.crystal_matrix[1]):
                    self.coordinates[i * (self.crystal_matrix[1]) + j + self.numberofcrystals * side, 0] = crystal_2_center_fov
                    self.coordinates[i*(self.crystal_matrix[1])+j+self.numberofcrystals*side,1] = (j-self.crystal_matrix[1]/2)*(self.real_size_crystal_dimensions[0]+2*reflector_depth_between_crystals_y) + additional_increment_y +self.detectors_distances[1]
                    self.coordinates[i * (self.crystal_matrix[1]) + j + self.numberofcrystals * side, 2] =i*(self.real_size_crystal_dimensions[0]+2*reflector_depth_between_crystals_x) + self.real_size_crystal_dimensions[1]/2 - total_size_array_Z/2


class CreateDummyCalibrationData:
    def __init__(self, crystal_matrix = None, directory = None):
        if crystal_matrix is None:
            return
        self.crystal_matrix = crystal_matrix
        self.directory = directory
        import datetime
        time_filename = datetime.datetime.now()
        self.time_filename = time_filename.strftime('%d %b %Y - %Hh %Mm %Ss')
        if not os.path.exists(directory):
            os.makedirs(directory)
        self._energy_file_creation()
        self._peak_position_file_creation()

    def _energy_file_creation(self):
        energy_matrix = np.ones(self.crystal_matrix[0]*self.crystal_matrix[1]*2+2)  # plus 2 ones because the last 2 number of the file correspond to the ADC channel for comparison
        file_name_energy = self.directory  + '/Easypet Scan ' + self.time_filename + '.calbenergy'
        np.savetxt(file_name_energy, energy_matrix, delimiter=",")
        # with open(file_name_energy, 'w') as output_file:
        #     energy_matrix.tofile(output_file)

    def _peak_position_file_creation(self):
        n_division = 1.96 / (self.crystal_matrix[0]*self.crystal_matrix[1])
        ratio_matrix = np.arange(-0.98,0.98,n_division)
        file_name_peak = self.directory + '/Easypet Scan ' + self.time_filename + '.calbpeak'
        np.savetxt(file_name_peak, ratio_matrix, delimiter=",")


class InformationWidgets:
    '''This classes generates a binary  that populates the options of the different
    comboboxes as records new data add by the user'''
    def __init__(self, operation = None, list2record = []):
        #key words
        self.list2record = list2record
        self.data_fromfile =[]
        self.key_words = ['Model', 'Control_Module', 'Scanning_Model', 'Resistive_Chain_type', 'Bed_Version',
                          'Type_connection', 'Baudrate', 'Bot_motor', 'Top_motor', 'Bed_motor', 'Fourth_motor']
        self.directory = os.path.dirname(os.path.abspath(__file__))
        self.file_path = '{}/bin/combobox_hardware_data.dat'.format(self.directory)
        if operation is None:
            return

        if operation == 'w' and len(list2record) > 0:
            self._write()

        if operation == 'r':
            self._read()

    def _write(self):
        new_list = []
        for i in range(len(self.list2record)):
            if i == 0:
                new_list = new_list + [';'+self.key_words[i]] + self.list2record[i]
            else:
                new_list = new_list + [self.key_words[i]]+self.list2record[i]
            new_list[-1] += ';\n'

        new_list = ';'.join(new_list)
        new_list = array('u', new_list)
        with open(self.file_path, 'wb') as output_file:
            new_list.tofile(output_file)

    def _read(self):

        with open(self.file_path, "rb") as binary_file:
            data_fromfile = np.fromfile(binary_file, dtype='|S1', count=-1).astype('|U1')

        data_fromfile = data_fromfile.tolist()
        data_fromfile = ''.join(data_fromfile)
        self.data_fromfile = data_fromfile.split('\n')


class Data(object):
    def __init__(self, width=35, height=35, color=QColor("red"), id_text=''):
        self._width = width
        self._height = height
        self._color = color
        self._id_text = id_text

    def width(self):
        return self._width

    def height(self):
        return self._height

    def color(self):
        return self._color

    def id_text(self):
        return self._id_text


class Model(QAbstractTableModel):

    WidthRole = Qt.UserRole + 1
    HeightRole = Qt.UserRole + 2
    ColorRole = Qt.UserRole + 3
    id_textRole = Qt.UserRole + 4

    _roles = {WidthRole: b"width", HeightRole: b"height", ColorRole: b"color", id_textRole: b'text'}

    def __init__(self, parent=None):
        QAbstractTableModel.__init__(self, parent)

        self._datas = []

    def addData(self, data):
        self.beginInsertRows(QModelIndex(), self.rowCount(), self.rowCount())
        #self.beginInsertColumns(QModelIndex(), self.columnCount(), self.columnCount())
        self._datas.append(data)
        self.endInsertRows()
        #self.endInsertColumns()

    def rowCount(self, parent=QModelIndex()):
        return len(self._datas)

    def columnCount(self, parent = QModelIndex()):
        return 4

    def data(self, index, role=Qt.DisplayRole):

        try:
            data = self._datas[index.row()]

        except IndexError:
            return QVariant()

        if role == self.WidthRole:
            return data.width()

        if role == self.HeightRole:
            return data.height()

        if role == self.ColorRole:

            return data.color()

        if role == self.id_textRole:
            return data.id_text()

        return QVariant()

    def roleNames(self):
        return self._roles

    # def setData(self, index, value, role=QtCore.Qt.EditRole):
    #     if role == QtCore.Qt.EditRole:
    #         self.__data[index.row()] = value
    #         self.dataChanged.emit(index, index)
    #         return True
    #     return False


def teste_coordinates():
    real_size_crystal_dimensions = [2, 2, 30, 0.28, 0.14, 0.175, 0.14,0.175]
    # mm [pitch_x, pitch_y, length_crystal(z), reflector_exterior, reflector_interior_x, reflector_interior_y]
    distance_between_motors = 30  # mm
    detectors_distances = [60, 2*3+2*0.28, 0, 60]
    crystals_matrix=[16,2]
    matrix = MatrixGeometryCorrection(crystals_dimensions=real_size_crystal_dimensions,
                               crystal_matrix=crystals_matrix,
                               distance_between_motors=distance_between_motors,
                               detectors_distances=detectors_distances)

    fig, ((ax1, ax2,ax3)) = plt.subplots(3, 1)
    center = [0, 0]
    ax1.scatter(matrix.coordinates[:,0], matrix.coordinates[:,2], alpha=0.5)
    ax1.set_xlabel('Distance to the FOV mm')
    ax1.set_ylabel('Crystal center  axial direction')
    ax2.scatter(matrix.coordinates[:,1], matrix.coordinates[:,2], alpha=0.5)
    ax2.set_ylabel('Crystal center  axial direction')
    ax3.scatter(matrix.coordinates[:,0], matrix.coordinates[:,1], alpha=0.5)
    ax3.set_xlabel('Distance to the FOV mm')
    ax3.set_ylabel('Crystals center ')

    ax1.scatter(center[0], center[1], alpha=0.8)
    ax2.scatter(center[0], center[1], alpha=0.8)
    ax3.scatter(center[0], center[1], alpha=0.8)



    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)

# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     dialog_t = DialogChangeHardware()
#     dialog_t.exec_()
    # allow to see the position of the crystals to the center
    # import matplotlib.pyplot as plt
    #
    # teste_coordinates()
    # plt.show()
