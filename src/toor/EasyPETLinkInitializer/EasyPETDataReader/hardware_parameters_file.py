import time
import os
import logging
from array import array
import numpy as np


class hardware_parameters:
    # This class create, changes and read a binary file with acquisition parameters
    def __init__(self):
        logging.info('START')
        self.directory = os.path.dirname(os.path.abspath(__file__))
        self.file_path = '{}/dataFiles/hardware_parameters.dat'.format(self.directory)
        self.number_of_parameters = 25
        logging.info('PATH '+ self.file_path)
        logging.info('END')

    def file_creation(self, list_parameters):
        logging.info('START')
        stringdata ='#'
        for i in range(0,len(list_parameters)):
            break_int = (i+1)%self.number_of_parameters

            if break_int == 0 and i>0:
                stringdata = '{}{};'.format(stringdata, list_parameters[i])
            elif i == len(list_parameters)-1:
                stringdata = '{}{}#'.format(stringdata,list_parameters[i])
            else:
                stringdata = '{}{},'.format(stringdata,list_parameters[i])

        len_type_acquisition = np.array([stringdata,stringdata,stringdata,stringdata])

        vectorized_len = np.vectorize(len)
        len_type_acquisition=vectorized_len(len_type_acquisition)
        len_type_acquisition = array('i', len_type_acquisition)
        stringdata = stringdata * 4
        stringdata = array('u', stringdata)
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'wb') as output_file:
                len_type_acquisition.tofile(output_file)
                stringdata.tofile(output_file)
        else:
            with open(self.file_path, 'wb') as output_file:
                len_type_acquisition.tofile(output_file)
                stringdata.tofile(output_file)
        logging.debug('END')

    def change_parameters(self, list_parameters, type_of_acquisition):
        logging.debug('START')
        logging.info('TYPE OF ACQUISITION: '+type_of_acquisition)
        logging.info('LIST OF PARAMETERS: ' + str(list_parameters)[1:-1])
        # new parameters
        if isinstance(list_parameters, str):
            stringdata_new = list_parameters

        if isinstance(list_parameters, list):
            stringdata_new = '#'

            for i in range(0, len(list_parameters)):
                break_int = (i + 1) % self.number_of_parameters

                if break_int == 0 and i > 0:
                    stringdata_new = '{}{};'.format(stringdata_new, list_parameters[i])
                elif i == len(list_parameters) - 1:
                    stringdata_new = '{}{}#'.format(stringdata_new, list_parameters[i])
                else:
                    stringdata_new = '{}{},'.format(stringdata_new, list_parameters[i])

        # reading old file
        [len_type_acquisition, stringdata] = self.read_parameters('all')

        if type_of_acquisition =='fast':
            stringdata = '{}{}{}{}'.format(stringdata_new,
                                         stringdata[len_type_acquisition[0]:np.sum(len_type_acquisition[0:2])],
                                         stringdata[np.sum(len_type_acquisition[0:2]):np.sum(len_type_acquisition[0:3])],
                                         stringdata[np.sum(len_type_acquisition[0:3]):np.sum(len_type_acquisition)])
            len_type_acquisition[0] = len(stringdata_new)


        if type_of_acquisition =='medium':
            stringdata = '{}{}{}{}'.format(stringdata[0:len_type_acquisition[0]],
                                         stringdata_new,
                                         stringdata[np.sum(len_type_acquisition[0:2]):np.sum(len_type_acquisition[0:3])],
                                         stringdata[np.sum(len_type_acquisition[0:3]):np.sum(len_type_acquisition)])

            len_type_acquisition[1] =len(stringdata_new)

        if type_of_acquisition =='slow':
            stringdata = '{}{}{}{}'.format(stringdata[0:len_type_acquisition[0]],
                                         stringdata[len_type_acquisition[0]:np.sum(len_type_acquisition[0:2])],
                                         stringdata_new,
                                         stringdata[np.sum(len_type_acquisition[0:3]):np.sum(len_type_acquisition)])

            len_type_acquisition[2] = len(stringdata_new)

        if type_of_acquisition == 'calibration':
            stringdata = '{}{}{}{}'.format(stringdata[0:len_type_acquisition[0]],
                                         stringdata[len_type_acquisition[0]:np.sum(len_type_acquisition[0:2])],
                                         stringdata[np.sum(len_type_acquisition[0:2]):np.sum(len_type_acquisition[0:3])],
                                         stringdata_new)

            len_type_acquisition[3] = len(stringdata_new)

        len_type_acquisition = array('i', len_type_acquisition)
        logging.info('STRING DATA SAVE: '+stringdata+'\n')
        stringdata = array('u', stringdata)
        with open(self.file_path, 'wb') as output_file:
            len_type_acquisition.tofile(output_file)
            stringdata.tofile(output_file)
        logging.debug('END')

    def read_parameters(self, type_of_acquisition):
        logging.debug('START')
        logging.info('TYPE OF ACQUISITION: ' + type_of_acquisition)
        number_of_types = 4

        with open(self.file_path, 'rb') as output_file:
            len_type_acquisition = np.fromfile(output_file, dtype=np.int, count=number_of_types)

            if type_of_acquisition =='fast':
                output_file.seek(number_of_types*4)
                size_to_read = len_type_acquisition[0]*2

            if type_of_acquisition =='medium':
                output_file.seek(len_type_acquisition[0]*2+number_of_types*4)
                size_to_read = len_type_acquisition[1]*2

            if type_of_acquisition == 'slow':
                output_file.seek((len_type_acquisition[0]+len_type_acquisition[1])*2+number_of_types*4)
                size_to_read =len_type_acquisition[2]*2

            if type_of_acquisition =='calibration':
                output_file.seek(np.sum(len_type_acquisition[0:3]) * 2 + number_of_types*4)
                size_to_read = len_type_acquisition[3] * 2

            if type_of_acquisition =='all':
                output_file.seek(number_of_types*4)
                size_to_read = np.sum(len_type_acquisition)*2

            stringdata= np.fromfile(output_file, dtype='|S1', count=size_to_read).astype('|U1')
            stringdata = stringdata.tolist()
            stringdata = ''.join(stringdata)

        logging.info('STRING DATA READ: ' + stringdata+'\n')
        logging.info('LENGTH OF ACQUISITION TYPE: ['+ str(len_type_acquisition[0]) + ' ' + str(len_type_acquisition[0]) + ' ' + str(len_type_acquisition[0]) + ' ' + str(len_type_acquisition[0]) + ']')
        logging.debug('END')
        return len_type_acquisition, stringdata


class EasypetVersionHardware:
    def __init__(self, operation = None, version = None, serial_connection = None, serial_number = 'training_0000', updated_info = None):
        logging.debug('START')
        logging.info('OPERATION: '+ operation)
        # self.version_hardware = version
        self.serial_connection = serial_connection
        self.serial_number = serial_number
        self.directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.file_path = '{}/dataFiles/easypet_hardware_version.dat'.format(self.directory)
        self.updated_info = updated_info
        self.number_of_versions = None
        self.space_per_version = None
        self.u_board_version = None
        self.module_control = None
        self.array_crystal_x = None
        self.array_crystal_y = None
        self.angle_bot_rotation = None
        self.angle_top_correction = None
        self.multiplexed = None
        self.reading_method = None
        self.number_adc_channel = None
        self.bed_version = None
        self.bed_diameter = None
        self.pc_communication = None
        self.baudrate = None
        self.motor_bot = None
        self.motor_top = None
        self.bed_motor = None
        self.fourth_motor = None
        self.capable4CT = None
        self.crystal_pitch_x = None
        self.crystal_pitch_y = None
        self.crystal_length = None
        self.reflector_exterior_thic = None
        self.reflector_interior_A_x = None
        self.reflector_interior_A_y = None
        self.reflector_interior_B_x = None
        self.reflector_interior_B_y = None
        self.distance_between_crystals = None
        self.distance_between_motors = None
        self.centercrystals2topmotor_x_sideA = None
        self.centercrystals2topmotor_x_sideB = None
        self.centercrystals2topmotor_y = None
        if operation == 'create_file':
            self.binary_file_creation()
        elif operation == 'read_file':
            self.read_file()
        elif operation == 'read_hardware_name':
            self.serial_number = self._get_version_number_from_hardware(self.serial_connection)
        elif operation == 'edit_file':
            self.editing_hardware_file()
        elif operation == 'read_hardware_name_test':
            self.serial_number = self._get_version_number_from_hardware_test()
        logging.debug('END')

    def binary_file_creation(self):
        logging.debug('START')
        number_of_machines = 5000
        space_per_version = 1000
        header_string = '{}\n{}\n'.format(number_of_machines, space_per_version)
        header_string = array('u', header_string)
        total_record_string =''
        for i in range(number_of_machines):
            if i <= int(number_of_machines/2):
                serial_number = 'training_{:04d}'.format(i)
                u_board_version = 'Jun 2018'
                module_control = 'MicroIO set18'
                array_crystal_x = 16
                array_crystal_y = 2
                angle_bot_rotation = 243
                angle_top_correction = -1
                multiplexed = 'yes'
                reading_method = 'Zig-Zag Order'
                number_adc_channel = 4
                bed_version = 'None'
                bed_diameter = 4.8
                pc_communication = 'USB'
                baudrate = '1000000'
                motor_bot = 'ST5909L1000B'
                motor_top = 'ST4209S1000A'
                bed_motor = 'None'
                fourth_motor = 'None'
                capable4CT = 'No'
                crystal_pitch_x = 2
                crystal_pitch_y = 2
                crystal_length = 30
                reflector_exterior_thic = 0.28
                reflector_interior_A_x = 0.14
                reflector_interior_A_y = 0.175
                reflector_interior_B_x = 0.14
                reflector_interior_B_y = 0.175
                distance_between_motors = 30
                distance_between_crystals = 60
                centercrystals2topmotor_x_sideA = 0
                centercrystals2topmotor_x_sideB = 0
                centercrystals2topmotor_y = 0.1

                record_string = '{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n' \
                                '{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n' \
                                '{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n'\
                    .format(serial_number, u_board_version, module_control, array_crystal_x,
                            array_crystal_y, angle_bot_rotation, angle_top_correction, multiplexed,
                            reading_method, number_adc_channel, bed_version, bed_diameter, pc_communication,
                            baudrate, motor_bot, motor_top, bed_motor, fourth_motor, capable4CT, crystal_pitch_x,
                            crystal_pitch_y, crystal_length, reflector_exterior_thic, reflector_interior_A_x,
                            reflector_interior_A_y, reflector_interior_B_x, reflector_interior_B_y,
                            distance_between_crystals, distance_between_motors, centercrystals2topmotor_x_sideA,
                            centercrystals2topmotor_x_sideB, centercrystals2topmotor_y)

            elif i > int(number_of_machines/2):
                serial_number = 'preclinical_{:04d}'.format(int(i-(number_of_machines/2)))
                u_board_version = 'Jun 2018'
                module_control = 'MicroIO set18'
                array_crystal_x = 16
                array_crystal_y = 2
                angle_bot_rotation = 243
                angle_top_correction = -1
                multiplexed = 'yes'
                reading_method = 'Zig-Zag Order'
                number_adc_channel = 4
                bed_version = 'None'
                bed_diameter = 4.8
                pc_communication = 'USB'
                baudrate = '1000000'
                motor_bot = 'ST5909L1000B'
                motor_top = 'ST4209S1000A'
                bed_motor = 'None'
                fourth_motor = 'None'
                capable4CT = 'No'
                crystal_pitch_x = 2
                crystal_pitch_y = 2
                crystal_length = 30
                reflector_exterior_thic = 0.28
                reflector_interior_A_x = 0.14
                reflector_interior_A_y = 0.175
                reflector_interior_B_x = 0.14
                reflector_interior_B_y = 0.175
                distance_between_motors = 30
                distance_between_crystals = 60
                centercrystals2topmotor_x_sideA = 0
                centercrystals2topmotor_x_sideB = 0
                centercrystals2topmotor_y = 0.1

                record_string = '{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n' \
                                '{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n' \
                                '{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n' \
                    .format(serial_number, u_board_version, module_control, array_crystal_x,
                            array_crystal_y, angle_bot_rotation, angle_top_correction, multiplexed,
                            reading_method, number_adc_channel, bed_version, bed_diameter, pc_communication,
                            baudrate, motor_bot, motor_top, bed_motor, fourth_motor, capable4CT, crystal_pitch_x,
                            crystal_pitch_y, crystal_length, reflector_exterior_thic, reflector_interior_A_x,
                            reflector_interior_A_y, reflector_interior_B_x, reflector_interior_B_y,
                            distance_between_crystals, distance_between_motors, centercrystals2topmotor_x_sideA,
                            centercrystals2topmotor_x_sideB, centercrystals2topmotor_y)

            total_record_string += record_string+ ' '*(space_per_version-len(record_string))+'\n'

        total_record_string = array('u', total_record_string)
        with open(self.file_path, 'wb') as output_file:
            header_string.tofile(output_file)
            total_record_string.tofile(output_file)
            #fill_string.tofile(output_file)


        output_file.close()
        logging.debug('END')

    # def _serial_number_binary_file_creation(self):
    #     ''' Quando a eeprom for colocada na placa acabar com esta solução de reescrever o ângulo para cada sistema'''
    #
    # def _serial_number_binary_file_read_file(self):
    #     ''' Quando a eeprom for colocada na placa acabar com esta solução de reescrever o ângulo para cada sistema'''

    def _set_serial_number_on_hardware(self, serial_connection, serial_number):
        logging.debug('START')
        logging.info('SERIAL CONNECTION: '+ serial_connection)
        logging.info('SERIAL NUMBER: ' + serial_number)
        '''future implementation - only with the eeprom installed on the board'''
        serial_connection.write(b'Set serial number!\n')
        serial_connection.write(b'{}\n'.format(serial_number))
        logging.debug('END')

    def _get_hardware_angle_correction(self, serial_connection):
        logging.debug('START')
        logging.info('SERIAL CONNECTION: ' + serial_connection)
        serial_connection.write(b'Angle correction!\n')
        angle_correction = serial_connection.readline().strip().decode()
        logging.info('ANGLE CORRECTION: ' + angle_correction)
        logging.debug('END')
        return angle_correction

    def _set_version_number_on_hardware(self, serial_connection, version_hardware):
        logging.debug('START')
        logging.info('SERIAL CONNECTION: ' + serial_connection)
        logging.info('HW VERSION: ' + version_hardware)
        if version_hardware == None:
            return
        serial_connection.write(b'Set version hardware\n')
        serial_connection.write(b'{}'.format(version_hardware))
        logging.debug('END')

    def _get_version_number_from_hardware(self):
        logging.debug('START')
        self.serial_connection.write(b'Get version hardware\n')
        version_hardware = self.serial_connection.readline().strip().decode()
        logging.info('HW VERSION: ' + version_hardware)
        logging.debug('END')
        return version_hardware

    def _get_version_number_from_hardware_test(self):
        logging.debug('START')
        version_hardware = 'training_0005'
        logging.info('HW VERSION: ' + version_hardware)
        logging.debug('END')
        return version_hardware

    def _get_hardware_firmware_version(self, serial_connection):
        logging.debug('START')
        logging.info('SERIAL CONNECTION: ' + serial_connection)
        serial_connection.write(b'Firmware version!\n')
        firmware_version = serial_connection.readline().strip().decode()
        logging.info('FIRMWARE VERSION: ' + firmware_version)
        logging.debug('END')
        return firmware_version

    def read_file(self):
        logging.debug('START')
        serial_number = self.serial_number
        with open(self.file_path, "rb") as binary_file:
            data_fromfile = np.fromfile(binary_file, dtype='|S1', count=-1).astype('|U1')
            data_fromfile = data_fromfile.tolist()
            data_fromfile = ''.join(data_fromfile)

        data_fromfile = data_fromfile.split('\n')
        interval_index = int(
            (len(data_fromfile) - 2) / int(data_fromfile[0]))  # number of parameters per serial number#
        serial_number = serial_number.split('_')
        begin_index = int(serial_number[1]) * interval_index + 2

        self.interval_index = interval_index
        self.number_of_versions = data_fromfile[0]
        self.space_per_version = data_fromfile[1]

        self.serial_number = data_fromfile[begin_index]
        self.u_board_version = data_fromfile[begin_index + 1]
        self.module_control = data_fromfile[begin_index + 2]
        self.array_crystal_x = 32 #int(data_fromfile[begin_index + 3])
        self.array_crystal_y = int(data_fromfile[begin_index + 4])
        self.angle_bot_rotation = data_fromfile[begin_index + 5]
        self.angle_top_correction = data_fromfile[begin_index + 6]
        self.multiplexed = data_fromfile[begin_index + 7]
        self.reading_method = data_fromfile[begin_index + 8]
        self.number_adc_channel = int(data_fromfile[begin_index + 9])
        self.bed_version = data_fromfile[begin_index + 10]
        self.bed_diameter = float(data_fromfile[begin_index + 11])
        self.pc_communication = data_fromfile[begin_index + 12]
        self.baudrate = data_fromfile[begin_index + 13]
        self.motor_bot = data_fromfile[begin_index + 14]
        self.motor_top = data_fromfile[begin_index + 15]
        self.bed_motor = data_fromfile[begin_index + 16]
        self.fourth_motor = data_fromfile[begin_index + 17]
        self.capable4CT = data_fromfile[begin_index + 18]
        self.crystal_pitch_x = float(data_fromfile[begin_index + 19])
        self.crystal_pitch_y = float(data_fromfile[begin_index + 20])
        self.crystal_length = float(data_fromfile[begin_index + 21])
        self.reflector_exterior_thic = float(data_fromfile[begin_index + 22])
        self.reflector_interior_A_x = float(data_fromfile[begin_index + 23])
        self.reflector_interior_A_y = float(data_fromfile[begin_index + 24])
        self.reflector_interior_B_x = float(data_fromfile[begin_index + 25])
        self.reflector_interior_B_y = float(data_fromfile[begin_index + 26])
        self.distance_between_motors = float(data_fromfile[begin_index + 27])
        self.distance_between_crystals = float(data_fromfile[begin_index + 28])
        self.centercrystals2topmotor_y = float(data_fromfile[begin_index + 29])
        self.centercrystals2topmotor_x_sideA = float(data_fromfile[begin_index + 30])
        self.centercrystals2topmotor_x_sideB = float(data_fromfile[begin_index + 31])
        logging.debug('END')


    def editing_hardware_file(self):
        logging.debug('START')
        '''future implementation, quando a eeprom  estiver implementada, poderá ser alterado os parametros desenhados por default através do software'''

        if len(self.updated_info) == 0 or self.updated_info is None:
            return
        serial_number = self.serial_number
        record_edited_list = self.updated_info
        tic = time.time()

        with open(self.file_path, "rb") as binary_file:
            data_fromfile = np.fromfile(binary_file, dtype='|S1', count=-1).astype('|U1')

        data_fromfile = data_fromfile.tolist()
        data_fromfile = ''.join(data_fromfile)
        data_fromfile = data_fromfile.split('\n')


        interval_index = int(
            (len(data_fromfile) - 2) / int(data_fromfile[0]))  # number of parameters per serial number
        serial_number = serial_number.split('_')
        begin_index = int(serial_number[1]) * interval_index+2 # sum 2 index because of the hearde of binary fle with the number of space per serial number
        data_fromfile[begin_index:interval_index + begin_index - 1] = record_edited_list
        data_tofile = '\n'.join(data_fromfile)
        data_tofile = array('u', data_tofile)

        with open(self.file_path, "wb") as binary_file:
            data_tofile.tofile(binary_file)

        toc = time.time()
        logging.debug('END')
