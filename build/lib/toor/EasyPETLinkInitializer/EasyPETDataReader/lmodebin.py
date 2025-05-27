
import numpy as np
import struct
from errno import ENOENT
from array import array
import time

import os
from os.path import splitext
import gc
import json
import logging
# from PyQt5 import QtWidgets
import platform
try:
    from .calibration_points_upload import calibration_points_init
except ImportError:
    from .calibration_points_upload import CalibrationPoints
from .hardware_parameters_file import EasypetVersionHardware

from .natural_sort import natural_sort
# from macholib.mach_o import data_in_code_entry
from .ratioEnergy_client import ratioEnergy_client
from .crystalsRC_client import crystalsRC_client
from .indexToEnergyCorrection_client import indexToEnergyCorrection_client
# from energyCorrection_client import energyCorrection_client
from .energyCorrection import energyCorrection
# from .energyCorrection_calibration import energyCorrection_calibration
from .peaksFile import peaksFile



class binary_data:
    def data_preparation(self,a1, a2, a3, a4, phi, theta, numberOfCrystals, peakMatrix,energyfactor,timestamp_list,threshold=0.98):
        logging.debug('START')

        # Filter data: Remove the values < 100
        theta = np.array(theta)
        phi = np.array(phi)
        a1 = np.array(a1)
        a2 = np.array(a2)
        a3 = np.array(a3)
        a4 = np.array(a4)
        timestamp_list = np.array(timestamp_list)
        index_a1_100 = np.where(a1 < 100)
        index_a2_100 = np.where(a2 < 100)
        index_a3_100 = np.where(a3 < 100)
        index_a4_100 = np.where(a4 < 100)

        indexes_intersection_A = np.intersect1d(index_a1_100, index_a2_100)
        indexes_intersection_B = np.intersect1d(index_a3_100, index_a4_100)

        # union_indexes_intersection
        union_indexes_intersection = np.union1d(indexes_intersection_A,
                                                indexes_intersection_B)

        data = np.zeros((len(a1),7))
        data[:, 0] = a1
        data[:, 1] = a2
        data[:, 2] = a3
        data[:, 3] = a4
        data[:, 4] = theta
        data[:, 5] = phi
        data[:, 6] = timestamp_list

        data = np.delete(data, union_indexes_intersection, axis=0)

        del a1,a2,a3,a4,theta,phi,timestamp_list
        gc.collect()

        [data, crystalMatrix, EA_corrected, EB_corrected, timestamp_list]=\
            self.data_preparation_energyconversion(data, threshold, numberOfCrystals, peakMatrix, energyfactor)

        logging.debug('END')
        return data, crystalMatrix, EA_corrected, EB_corrected, timestamp_list

    def data_preparation_energyconversion(self, data, threshold, numberOfCrystals, peakMatrix,energyfactor):
        logging.debug('START')

        [ratioMatrix, data] = ratioEnergy_client(data, threshold, numberOfCrystals)

        timestamp_list = np.array([data[:, 6]])
        data = data[:, 0:6]
        crystalMatrix = crystalsRC_client(ratioMatrix, peakMatrix, numberOfCrystals)

        [list_energy_crystals_RCA, list_indexes_crystals_RCA, EA, unique_ECA] = indexToEnergyCorrection_client(
            data[:, 0],
            data[:, 1],
            crystalMatrix[:, 0])

        [list_energy_crystals_RCB, list_indexes_crystals_RCB, EB, unique_ECB] = indexToEnergyCorrection_client(
            data[:, 2],
            data[:, 3],
            crystalMatrix[:, 1])

        [EA_corrected, EB_corrected] = energyCorrection(list_energy_crystals_RCA, list_indexes_crystals_RCA,
                                                        EA, list_energy_crystals_RCB, list_indexes_crystals_RCB, EB,
                                                        numberOfCrystals, unique_ECA, unique_ECB, energyfactor)
        logging.debug('END')
        return data, crystalMatrix, EA_corrected, EB_corrected, timestamp_list

    def only_energycorrectionfactor(self,peak_Matrix_x, a1, a2, a3, a4, phi, theta,timestamp_list, numberOfCrystals, file_name):
        logging.debug('START')
        logging.info('FILE NAME: ' + str(file_name))
        logging.info('NUMBER OF CRYSTALS: '+str(numberOfCrystals[0]) + 'x 2')

        theta = np.array(theta)
        phi = np.array(phi)
        a1 = np.array(a1)
        a2 = np.array(a2)
        a3 = np.array(a3)
        a4 = np.array(a4)
        timestamp_list = np.array(timestamp_list)

        index_a1_100 = np.where(a1 < 100)
        index_a2_100 = np.where(a2 < 100)
        index_a3_100 = np.where(a3 < 100)
        index_a4_100 = np.where(a4 < 100)

        indexes_intersection_A = np.intersect1d(index_a1_100, index_a2_100)
        indexes_intersection_B = np.intersect1d(index_a3_100, index_a4_100)

        # union_indexes_intersection
        union_indexes_intersection = np.union1d(indexes_intersection_A,
                                                indexes_intersection_B)

        data = np.concatenate(([a1], [a2], [a3], [a4], [theta], [phi], [timestamp_list]), axis=0)
        data = data.T
        data = np.delete(data, union_indexes_intersection, axis=0)

        [ratioMatrix, data] = ratioEnergy_client(data, 0.98, numberOfCrystals)

        timestamp_list = np.array([data[:, 6]])
        # print(timestamp_list)
        data = data[:, 0:6]

        crystalMatrix = crystalsRC_client(ratioMatrix, peak_Matrix_x, numberOfCrystals)

        [list_energy_crystals_RCA, list_indexes_crystals_RCA, energyA, unique_ECA] = indexToEnergyCorrection_client(
            data[:, 0], data[:, 1], crystalMatrix[:, 0])
        [list_energy_crystals_RCB, list_indexes_crystals_RCB, energyB, unique_ECB] = indexToEnergyCorrection_client(
            data[:, 2], data[:, 3], crystalMatrix[:, 1])

        try:
            [EA_Corrected, EB_Corrected, correctionFactor_Matrix, peaks_RC] = energyCorrection_calibration(data, 100,
                                                                                                 list_energy_crystals_RCA,
                                                                                                 list_energy_crystals_RCB,
                                                                                                 list_indexes_crystals_RCA,
                                                                                                 list_indexes_crystals_RCB,
                                                                                                 0, unique_ECA)


            head, tail = os.path.split(file_name)

            directory = os.path.dirname(os.path.abspath(__file__))
            path = directory + '\\system_configurations\\x_{}__y_{}\\'.format(numberOfCrystals[0],numberOfCrystals[1])
            tail = tail.split('.')
            tail = tail[0]
            file_name = path + tail + '_CF.calbenergy'

            numberOfCrystals = numberOfCrystals[0] * numberOfCrystals[1]
            CF = np.zeros((numberOfCrystals * 2))

            CF[0:numberOfCrystals] = correctionFactor_Matrix[:, 0]

            CF[numberOfCrystals:(numberOfCrystals * 2)] = correctionFactor_Matrix[:, 1]
            CF = np.append(CF,peaks_RC)
            np.savetxt(file_name, CF, delimiter=", ")
            logging.debug('Peaks Energy file saved: ' + file_name)
            flag_energy_corrected = 1

        except TypeError as e:
            logging.exception('ENERGY NOT CORRECTED')
            logging.exception(e)

            energyCorrection_calibration(data, 100, list_energy_crystals_RCA, list_energy_crystals_RCB,
                                         list_indexes_crystals_RCA, list_indexes_crystals_RCB, 0, unique_ECA)

            EA_Corrected = energyA
            EB_Corrected = energyB
            flag_energy_corrected = 0

        logging.debug('END')
        return data, crystalMatrix, EA_Corrected, EB_Corrected, timestamp_list, flag_energy_corrected

    def save_listmode(self, file_path, file_name_original, data, crystalMatrix, EA_Corrected, EB_Corrected, timestamp,head, time_array,abort_bool, stringdata, acquisitionParameters, part_file_number=None, joining_files=False):
        logging.debug('START')
        logging.info('PART FILE NUMBER: ' + str(part_file_number))
        logging.info('JOINING FILES: ' + str(joining_files))

        if part_file_number is not None:
            temp_file_name = '{}.easypet'.format(part_file_number)
            file_name = os.path.join(file_path, 'temp_files', temp_file_name)

        else:
            string = file_path.split('.')
            file_name = string[0] + ' Original data.easypet'

        # If is the final file
        if joining_files:
            acquisition_name = os.path.basename(os.path.normpath(file_path))
            file_name = os.path.join(file_path, acquisition_name + '.easypet')

        # This part is needed if you wnat to recover files (this part should be reviewed
        if not joining_files:
            file_name_original = file_name_original.split('/')
            file_name_original = file_name_original[-1]
            file_name_original = file_name_original.split('.')
            file_name_original = file_name_original[0]

        if file_name == file_name_original:
            file_name_original = '{} Original data'.format(file_name_original)

        step_bot = round(head[1], 3)/head[2]
        data4save = data

        serial_number = EasypetVersionHardware(operation='read_hardware_name_test').serial_number
        reading_hardware_parameters = EasypetVersionHardware(operation='read_file', serial_number=serial_number)

        if not joining_files:
            if data4save.shape[1] == 7:
                data4save = data[:, 4:6]
            elif data4save.shape[1] == 2:
                data4save = data
            increment = float(reading_hardware_parameters.angle_bot_rotation) / step_bot  # forcing image rotation. should be a parameter 72 for spain #-36 ESSUA
            data4save[:,0]=data4save[:,0]+increment

        crystalMatrix = np.array(crystalMatrix,dtype='int16',copy=False)
        data4save = data4save.astype(dtype='int16', copy=False)
        EA_Corrected=EA_Corrected.astype(dtype='int16', copy=False)
        EB_Corrected = EB_Corrected.astype(dtype='int16', copy=False)
        timestamp=timestamp.astype(dtype='float32', copy=False)
        if not joining_files:
            records = np.rec.fromarrays(
                (data4save[:, 0], data4save[:, 1], crystalMatrix[:, 0], crystalMatrix[:, 1],
                 EA_Corrected[0, :], EB_Corrected[0, :],timestamp[0,:]),
                names=('bot', 'top', 'id1', 'id2', 'EA', 'EB', 'timestamp'))
        else:
            records = np.rec.fromarrays(
                (data4save[:, 0], data4save[:, 1], crystalMatrix[:, 0], crystalMatrix[:, 1],
                 EA_Corrected, EB_Corrected, timestamp),
                names=('bot', 'top', 'id1', 'id2', 'EA', 'EB', 'timestamp'))

        header = array('f', head)
        dates = array('u', time_array)

        Version_binary = 'Version 2'
        Version_binary = array('u', Version_binary)
        acquisitionParameters = json.dumps(acquisitionParameters)
        acquisition_info = array('u', acquisitionParameters)
        acquisition_info_size = [len(acquisitionParameters)]
        acquisition_info_size = array('i', acquisition_info_size)

        stringdata = json.dumps(stringdata)
        stringdata = array('u', stringdata)
        stringdata_size = [len(stringdata)]
        stringdata_size = array('i', stringdata_size)

        systemConfigurations = {
            'serial_number': serial_number,
            'u_board_version': reading_hardware_parameters.u_board_version,
            'module_control': reading_hardware_parameters.module_control,
            'array_crystal_x': reading_hardware_parameters.array_crystal_x,
            'array_crystal_y': reading_hardware_parameters.array_crystal_y,
            'angle_bot_rotation': reading_hardware_parameters.angle_bot_rotation,
            'angle_top_correction': reading_hardware_parameters.angle_top_correction,
            'multiplexed': reading_hardware_parameters.multiplexed,
            'reading_method': reading_hardware_parameters.reading_method,
            'number_adc_channel': reading_hardware_parameters.number_adc_channel,
            'bed_version': reading_hardware_parameters.bed_version,
            'bed_diameter': reading_hardware_parameters.bed_diameter,
            'pc_communication': reading_hardware_parameters.pc_communication,
            'baudrate': reading_hardware_parameters.baudrate,
            'motor_bot': reading_hardware_parameters.motor_bot,
            'motor_top': reading_hardware_parameters.motor_top,
            'bed_motor': reading_hardware_parameters.bed_motor,
            'fourth_motor': reading_hardware_parameters.fourth_motor,
            'capable4CT': reading_hardware_parameters.capable4CT,
            'crystal_pitch_x': reading_hardware_parameters.crystal_pitch_x,
            'crystal_pitch_y': reading_hardware_parameters.crystal_pitch_y,
            'crystal_length': reading_hardware_parameters.crystal_length,
            'reflector_exterior_thic': reading_hardware_parameters.reflector_exterior_thic,
            'reflector_interior_A_x': reading_hardware_parameters.reflector_interior_A_x,
            'reflector_interior_A_y': reading_hardware_parameters.reflector_interior_A_y,
            'reflector_interior_B_x': reading_hardware_parameters.reflector_interior_B_x,
            'reflector_interior_B_y': reading_hardware_parameters.reflector_interior_B_y,
            'distance_between_motors': reading_hardware_parameters.distance_between_motors,
            'distance_between_crystals': reading_hardware_parameters.distance_between_crystals,
            'centercrystals2topmotor_x_sideA': reading_hardware_parameters.centercrystals2topmotor_x_sideA,
            'centercrystals2topmotor_x_sideB': reading_hardware_parameters.centercrystals2topmotor_x_sideB,
            'centercrystals2topmotor_y': reading_hardware_parameters.centercrystals2topmotor_y
        }

        systemConfigurations = json.dumps(systemConfigurations)
        systemConfigurations_info = array('u', systemConfigurations)
        systemConfigurations_info_size = [len(systemConfigurations_info )]
        systemConfigurations_info_size = array('i', systemConfigurations_info_size)

        [peakMatrix, calibration_file, energyfactor] = calibration_points_init(
            [int(reading_hardware_parameters.array_crystal_x), int(reading_hardware_parameters.array_crystal_y)])

        energyfactor = energyfactor[0].tolist()
        energyfactor_str = str(energyfactor).strip('[]')
        energyfactor_str = energyfactor_str.replace(" ", "")
        energyfactor_size = array('i', [len(energyfactor_str)])
        energyfactor_info = array('u', energyfactor_str)

        peakMatrix = peakMatrix.flatten('F')
        peakMatrix_str = str(peakMatrix).strip('[]')

        peakMatrix_size = array('i', [len(peakMatrix_str)])
        peakMatrix_info = array('u', peakMatrix_str)

        size_header = [2 * 9 + 4 + 24 + 102 + 4 + acquisition_info_size[0] * 2 + 4 + stringdata_size[0] * 2 +
                       4 + systemConfigurations_info_size[0]*2 + 4 + energyfactor_size[0]*2 +
                       4 + peakMatrix_size[0]*2]
        size_header = array('i', size_header)

        with open(file_name, 'wb') as output_file:
            Version_binary.tofile(output_file)
            size_header.tofile(output_file)
            header.tofile(output_file)
            dates.tofile(output_file)
            acquisition_info_size.tofile(output_file)
            stringdata_size.tofile(output_file)
            systemConfigurations_info_size.tofile(output_file)
            energyfactor_size.tofile(output_file)
            peakMatrix_size.tofile(output_file)
            acquisition_info.tofile(output_file)
            stringdata.tofile(output_file)
            systemConfigurations_info.tofile(output_file)
            energyfactor_info.tofile(output_file)
            peakMatrix_info.tofile(output_file)
            records.tofile(output_file)

        output_file.close()
        del data4save, EA_Corrected, EB_Corrected, timestamp,
        gc.collect()
        logging.debug('END')

    def _join_parts_into_easypetfile(self, acquisitionParameters, file_study_path = None):
        logging.debug('START')
        logging.info('file_study_path ' + str(file_study_path))

        tic = time.time()
        file_path = os.path.join(file_study_path, 'temp_files')
        parts = [f.path for f in os.scandir(file_path) if f.is_file() and f.name.endswith("easypet")]
        # size = [f.for f in os.scandir(file_path) if f.is_file() and f.name.endswith("easypetoriginal")]
        parts = natural_sort(parts)
        list_test = [None] * len(parts)
        listmode_partial = [None] * len(parts)

        i = 0
        total_size_chunks = 0
        listmode = None
        for part in parts:
            [listmode_partial[i], Version_binary, header, dates, otherinfo, acquisitionInfo, stringdata, systemConfigurations_info, energyfactor_info, peakMatrix_info] = binary_data().open(part,open_files_to_join=True)
            list_test[i] = len(listmode_partial[i])
            total_size_chunks += len(listmode_partial[i])
            i += 1

        listmode = np.zeros((int(total_size_chunks), listmode_partial[0].shape[1]))
        array_index = np.array(list_test)
        for l in range(len(list_test)):
            if l == 0:
                begin_index = 0
            else:
                begin_index = np.sum(array_index[:l])
            end_index = np.sum(array_index[:l + 1])
            listmode[begin_index:end_index, :] = listmode_partial[l]

        if end_index not in acquisitionParameters["Turn end index"]:
            acquisitionParameters["Turn end index"].append(int(end_index))
        binary_data().save_listmode(file_study_path, acquisitionParameters["File name"], listmode[:,4:6], listmode[:,2:4], listmode[:,1],
                                    listmode[:, 0], listmode[:,6],
                                    header, dates, acquisitionParameters["Abort bool"], stringdata, acquisitionParameters,
                                    part_file_number=None, joining_files=True)

        logging.debug('END')

    def _join_parts_into_easypetoriginalfile(self, acquisitionParameters, file_study_path = None):
        logging.debug('START')
        logging.info('file_study_path '+ str(file_study_path))

        tic = time.time()
        file_path = os.path.join(file_study_path, 'temp_files')
        parts = [f.path for f in os.scandir(file_path) if f.is_file() and f.name.endswith("easypetoriginal")]
        # size = [f.for f in os.scandir(file_path) if f.is_file() and f.name.endswith("easypetoriginal")]
        parts= natural_sort(parts)
        list_test = [None]*len(parts)
        listmode_partial = [None]*len(parts)

        i=0
        total_size_chunks=0
        listmode =None
        for part in parts:
            listmode_partial[i] = binary_data().open_original_data(part)
            list_test[i]=len(listmode_partial[i])
            total_size_chunks+=len(listmode_partial[i])
            i+=1

        listmode = np.zeros((int(total_size_chunks),listmode_partial[0].shape[1]))
        array_index= np.array(list_test)
        for l in range(len(list_test)):
            if l ==0:
                begin_index=0
            else:
                begin_index = np.sum(array_index[:l])
            end_index= np.sum(array_index[:l+1])
            listmode[begin_index:end_index,:] = listmode_partial[l]

        if end_index not in acquisitionParameters["Turn end index original data"]:
            acquisitionParameters["Turn end index original data"].append(int(end_index))
        binary_data().save_original_data(file_study_path,listmode[:,0],listmode[:,1],listmode[:,2],listmode[:,3],listmode[:,4], listmode[:,5], listmode[:,6],joining_files=True)
        logging.debug('END')

    def save_original_data(self, file_path, a1, a2, a3, a4, phi, theta,timestamp_list, part_file_number=None, joining_files=False):
        logging.debug('START')
        logging.info('part_file_number: '+str(part_file_number))
        logging.info('joining_files: '+str(joining_files))

        theta = np.array(theta)
        phi = np.array(phi)
        a1 = np.array(a1)
        a2 = np.array(a2)
        a3 = np.array(a3)
        a4 = np.array(a4)
        timestamp_list = np.array(timestamp_list) #? is necessary change to ms?

        a1 = a1.astype(dtype='int16', copy=False)
        a2 = a2.astype(dtype='int16', copy=False)
        a3 = a3.astype(dtype='int16', copy=False)
        a4 = a4.astype(dtype='int16', copy=False)
        theta = theta.astype(dtype='int16', copy=False)
        phi = phi.astype(dtype='int16', copy=False)
        timestamp = timestamp_list.astype(dtype='float32', copy=False)
        if part_file_number is not None:
            temp_file_name='Original data {}.easypetoriginal'.format(part_file_number)
            file_name = os.path.join(file_path, 'temp_files', temp_file_name)
            #file_name = '{}\\temp_files\\Original data {}.easypetoriginal'.format(file_path,part_file_number)
        else:
            # CPROBABLY COULD BE DEPRECATED...
            string = file_path.split('.')
            file_name = string[0] + ' Original data.easypetoriginal'

        if joining_files:
            acquisition_name = os.path.basename(os.path.normpath(file_path))
            #file_name = file_path + acquisition_name + ' Original data.easypetoriginal'
            file_name = os.path.join(file_path, acquisition_name + ' Original data.easypetoriginal')

        try:
            records = np.rec.fromarrays(
                (a1, a2, a3, a4, theta, phi, timestamp),
                names=('a1', 'a2', 'a3', 'a4', 'theta', 'phi', 'timestamp'))

            with open(file_name, 'wb') as output_file:
                records.tofile(output_file)
                # data4save.tofile(output_file,format="%i%h%h%h%h%h%h")
            output_file.close()
        except ValueError:
            logging.exception('Falhou')

        del theta,phi,a1,a2,a3,a4,timestamp_list
        gc.collect()
        logging.debug('END')

    def open_original_data(self, file_name):
        logging.debug('START')
        logging.info('FILE NAME: '+ file_name)

        if not os.path.isfile(file_name):
            raise IOError(ENOENT, 'Not a file', file_name)

        with open(file_name, "rb") as binary_file:
            reading_data = np.fromfile(binary_file,
                                       dtype=[('a1', np.int16), ('a2', np.int16), ('a3', np.int16), ('a4', np.int16),
                                              ('step_bot', np.int16), ('step_top', np.int16), ('time', np.float32)])


        listMode = np.zeros((len(reading_data["a1"]), 7), dtype=np.float32)
        listMode[:, 0] = reading_data["a1"]
        listMode[:, 1] = reading_data["a2"]
        listMode[:, 2] = reading_data["a3"]
        listMode[:, 3] = reading_data["a4"]
        listMode[:, 4] = reading_data["step_bot"]
        listMode[:, 5] = reading_data["step_top"]
        listMode[:, 6] = reading_data["time"]

        logging.debug('END')
        return listMode

    def open(self, file_name, open_files_to_join=False):
        logging.debug('START')

        acquisitionInfo = stringdata = otherinfo = systemConfigurations_info = energyfactor_info = peakMatrix_info = []
        print(file_name)
        try:
            with open(file_name, "rb") as binary_file:
                try:
                    Version_binary = np.fromfile(binary_file, dtype='|S1', count=18).astype('|U1')
                    Version_binary = Version_binary.tolist()
                    Version_binary = ''.join(Version_binary)

                    logging.info('HEADER VERSION: '+Version_binary)
                    print(Version_binary)
                    seek_values = np.array([18, 22, 46, 148, 152, 156, 160,164, 168])
                    if Version_binary.startswith('Version'):
                        os_creation_file = "Windows"

                    else:
                        os_creation_file = "Linux"
                        binary_file.seek(0)
                        Version_binary = np.fromfile(binary_file, dtype='|S1', count=36).astype('|U1')
                        Version_binary = Version_binary.tolist()
                        Version_binary = ''.join(Version_binary)
                        seek_values += 18
                        seek_values[3:]+=102


                    binary_file.seek(seek_values[0])
                    size_header = np.fromfile(binary_file, dtype=np.int32, count=1)

                    binary_file.seek(seek_values[1])
                    header = np.fromfile(binary_file, dtype=np.float32, count=6)

                    binary_file.seek(seek_values[2])
                    if os_creation_file == "Windows":
                        dates = np.fromfile(binary_file, dtype='|S1', count=102).astype('|U1')
                    elif os_creation_file == "Linux":
                        dates = np.fromfile(binary_file, dtype='|S1', count=204).astype('|U1')

                    dates = dates.tolist()
                    dates = ''.join(dates)

                    binary_file.seek(seek_values[3])
                    # Version_binary = 'Version 1' # for files with version 2 tag but dont have the correct fields
                    if Version_binary == 'Version 1':


                        otherinfo = np.fromfile(binary_file, dtype='|S1', count=2000).astype('|U1')
                        otherinfo = otherinfo.tolist()
                        otherinfo = ''.join(otherinfo)
                        otherinfo = otherinfo.split(';')

                        otherinfo ={
                            otherinfo[0],
                            otherinfo[1]
                        }
                        serial_number = EasypetVersionHardware(operation='read_hardware_name_test').serial_number
                        reading_hardware_parameters = EasypetVersionHardware(operation='read_file',
                                                                             serial_number=serial_number)
                        systemConfigurations_info = {
                            'serial_number': serial_number,
                            'u_board_version': reading_hardware_parameters.u_board_version,
                            'module_control': reading_hardware_parameters.module_control,
                            'array_crystal_x': 16,
                            # 'array_crystal_x': reading_hardware_parameters.array_crystal_x,
                            'array_crystal_y': reading_hardware_parameters.array_crystal_y,
                            'angle_bot_rotation': reading_hardware_parameters.angle_bot_rotation,
                            'angle_top_correction': reading_hardware_parameters.angle_top_correction,
                            'multiplexed': reading_hardware_parameters.multiplexed,
                            'reading_method': reading_hardware_parameters.reading_method,
                            'number_adc_channel': reading_hardware_parameters.number_adc_channel,
                            'bed_version': reading_hardware_parameters.bed_version,
                            'bed_diameter': reading_hardware_parameters.bed_diameter,
                            'pc_communication': reading_hardware_parameters.pc_communication,
                            'baudrate': reading_hardware_parameters.baudrate,
                            'motor_bot': reading_hardware_parameters.motor_bot,
                            'motor_top': reading_hardware_parameters.motor_top,
                            'bed_motor': reading_hardware_parameters.bed_motor,
                            'fourth_motor': reading_hardware_parameters.fourth_motor,
                            'capable4CT': reading_hardware_parameters.capable4CT,
                            'crystal_pitch_x': reading_hardware_parameters.crystal_pitch_x,
                            'crystal_pitch_y': reading_hardware_parameters.crystal_pitch_y,
                            'crystal_length': reading_hardware_parameters.crystal_length,
                            'reflector_exterior_thic': reading_hardware_parameters.reflector_exterior_thic,
                            'reflector_interior_A_x': reading_hardware_parameters.reflector_interior_A_x,
                            'reflector_interior_A_y': reading_hardware_parameters.reflector_interior_A_y,
                            'reflector_interior_B_x': reading_hardware_parameters.reflector_interior_B_x,
                            'reflector_interior_B_y': reading_hardware_parameters.reflector_interior_B_y,
                            'distance_between_motors': 30,
                            'distance_between_crystals': 60,
                            'centercrystals2topmotor_x_sideA': reading_hardware_parameters.centercrystals2topmotor_x_sideA,
                            'centercrystals2topmotor_x_sideB': reading_hardware_parameters.centercrystals2topmotor_x_sideB,
                            'centercrystals2topmotor_y': reading_hardware_parameters.centercrystals2topmotor_y
                        }
                        acquisitionInfo = {
                            "Tracer": "FDG-18",
                            "Half life": 6586.26,
                            "Positron Fraction": "",
                            "Description of batch": "",
                            "Volume tracer": 10,
                            "Total Dose": 18500000,
                            "Start date time": "start_date + " " + start_time",
                            "Residual Activity": 18500000,
                            "End date time": "end_date + " " + end_time",
                            "Turn end index": [],
                            "Turn end index original data": [],
                            "Acquisition start time": 10,
                            "Number of turns": 0,
                            "ID": "Dummy",
                            "Type of subject": "Quick Scan",
                        }
                        number_of_arrays = 7
                        reading_data = np.fromfile(binary_file,
                                                   dtype=[('stepbot', np.int16), ('steptop', np.int16),
                                                          ('idA', np.int16),
                                                          ('idB', np.int16),
                                                          ('EA', np.int16), ('EB', np.int16), ('time', np.float32)])

                    else:
                        size_acquisition_info = np.fromfile(binary_file, dtype=np.int32, count=1)

                        binary_file.seek(seek_values[4])
                        size_stringdata = np.fromfile(binary_file, dtype=np.int32, count=1)

                        binary_file.seek(seek_values[5])
                        size_systemConfigurations = np.fromfile(binary_file, dtype=np.int32, count=1)

                        binary_file.seek(seek_values[6])
                        size_energyfactor = np.fromfile(binary_file, dtype=np.int32, count=1)

                        binary_file.seek(seek_values[7])
                        size_peakMatrix = np.fromfile(binary_file, dtype=np.int32, count=1)

                        binary_file.seek(seek_values[8])

                        if os_creation_file == 'Linux':
                            size_acquisition_info[0] = size_acquisition_info[0] * 2
                            size_stringdata[0] = size_stringdata[0] * 2
                            size_systemConfigurations[0] = size_systemConfigurations[0] * 2
                            size_energyfactor[0] = size_energyfactor[0] * 2
                            size_peakMatrix[0] = size_peakMatrix[0] * 2
                        # if os_creation_file == 'Windows':
                        acquisitionInfo = np.fromfile(binary_file, dtype='|S1',
                                                      count=size_acquisition_info[0] * 2).astype('|U1')
                        # else:
                        #     acquisitionInfo = np.fromfile(binary_file, dtype='|S1',
                        #                                   count=size_acquisition_info[0] * 2*2).astype('|U1')
                        acquisitionInfo = acquisitionInfo.tolist()
                        acquisitionInfo = ''.join(acquisitionInfo)
                        acquisitionInfo = json.loads(acquisitionInfo)

                        binary_file.seek(seek_values[8] + size_acquisition_info[0]*2)
                        stringdata = np.fromfile(binary_file, dtype='|S1', count=size_stringdata[0] * 2).astype('|U1')
                        stringdata = stringdata.tolist()
                        stringdata = ''.join(stringdata)
                        stringdata = stringdata.split(';')

                        binary_file.seek(seek_values[8] + size_acquisition_info[0] * 2 + size_stringdata[0] * 2)
                        systemConfigurations_info = np.fromfile(binary_file, dtype='|S1', count=size_systemConfigurations[0]* 2).astype('|U1')
                        systemConfigurations_info = systemConfigurations_info.tolist()
                        systemConfigurations_info = ''.join(systemConfigurations_info)
                        systemConfigurations_info = json.loads(systemConfigurations_info)

                        binary_file.seek(seek_values[8] + size_acquisition_info[0] * 2 + size_stringdata[0] * 2
                                         + size_systemConfigurations[0]*2)
                        energyfactor_info = np.fromfile(binary_file, dtype='|S1', count=size_energyfactor[0] * 2).astype('|U1')
                        energyfactor_info = energyfactor_info.tolist()
                        energyfactor_info = ''.join(energyfactor_info)

                        binary_file.seek(seek_values[8] + size_acquisition_info[0] * 2 + size_stringdata[0] * 2 + size_systemConfigurations[0] * 2 + size_energyfactor[0]*2)
                        peakMatrix_info = np.fromfile(binary_file, dtype='|S1', count=size_peakMatrix[0] * 2).astype(
                            '|U1')
                        peakMatrix_info = peakMatrix_info.tolist()
                        peakMatrix_info = ''.join(peakMatrix_info)


                    # binary_file.seek(size_header[0])

                    if Version_binary == 'Version 2':
                        binary_file.seek(
                            size_acquisition_info[0] * 2 + size_energyfactor[0] * 2 + size_peakMatrix[0] * 2 +
                            size_stringdata[0] * 2 + size_systemConfigurations[0] * 2 + seek_values[8])
                        number_of_arrays = 7
                        reading_data = np.fromfile(binary_file,
                                                   dtype=[('stepbot', np.int16), ('steptop', np.int16), ('idA', np.int16),
                                                          ('idB', np.int16),
                                                          ('EA', np.int16), ('EB', np.int16), ('time', np.float32)])
                    elif Version_binary == 'Version 3':
                        binary_file.seek(
                            size_acquisition_info[0] * 2 + size_energyfactor[0] * 2 + size_peakMatrix[0] * 2 +
                            size_stringdata[0] * 2 + size_systemConfigurations[0] * 2 + seek_values[8])
                        number_of_arrays = 8
                        reading_data = np.fromfile(binary_file,
                                                   dtype=[('stepbot', np.int16), ('steptop', np.int16),
                                                          ('idA', np.int16),
                                                          ('idB', np.int16),
                                                          ('EA', np.int16), ('EB', np.int16), ('time', np.float64),
                                                          ('time_b', np.float64)])

                except UnicodeDecodeError as e:
                    print(e)
                    binary_file.seek(0)
                    header = np.fromfile(binary_file, dtype=np.float32, count=6)
                    size_header = 126
                    binary_file.seek(size_header)
                    reading_data = np.fromfile(binary_file,
                                               dtype=[('stepbot', np.int16), ('steptop', np.int16), ('idA', np.int16), ('idB', np.int16),
                                                      ('EA', np.int16), ('EB', np.int16), ('time', np.float32)])
                    number_of_arrays = 7
                    logging.exception('FALHEI')

            step_bot = round(header[1], 4) / header[2]
            step_top = round(header[3], 4) / header[4]
            topRange = header[5]

            listMode = np.zeros((len(reading_data["stepbot"]), number_of_arrays))
            listMode[:, 0] = reading_data["EA"]  # energias A
            listMode[:, 1] = reading_data["EB"]  # energias B
            listMode[:, 2] = reading_data["idA"]  # id A
            listMode[:, 3] = reading_data["idB"]  # id b
            if open_files_to_join:
                # This dont change the data in the way to merge the acquisitions files in just one
                listMode[:, 4] = (reading_data["stepbot"]) # *2  # bot angle remove factor of 2 because it is already saved that way
                listMode[:, 5] = (reading_data["steptop"] )  # top_angle

            else:
                listMode[:, 4] = (reading_data["stepbot"]) * step_bot #*2  # bot angle remove factor of 2 because it is already saved that way
                listMode[:, 5] = (reading_data["steptop"] * step_top - topRange / 2)  # top_angle
            listMode[:, 6] = reading_data["time"]  # time

            if Version_binary ==  "Version 3":
                listMode[:, 7] = reading_data["time_b"]  # time

            logging.debug('END')
            return listMode, Version_binary, header, dates, otherinfo, acquisitionInfo, stringdata, systemConfigurations_info, energyfactor_info, peakMatrix_info

        except FileNotFoundError:
            raise FileNotFoundError


    def open_new_calibration_files(self, calibration_file_name, numberOfCrystals):
        '''This function reads the files where is stored the calibration points. Allow to the user to choose the pair
        of files. Accept one file and found the pair in the same directory, or two files with different extensions
         given by the user. More than 2 files raises an exception. Two files with the same extension raises an exception
         19/03/2018 - Used in advanced_info_dialog().change_calibration_file()
         19/03/2018 - Used to change the calibration points of the program. Change the .easypet files generated after
         this change'''

        logging.debug('START')
        logging.info('CALIBRATION FILE NAME: '+str(calibration_file_name))
        logging.info('NUMBER OF CRYSTALS: '+str(numberOfCrystals))
        file_names = calibration_file_name
        logging.info('NUMBER OF FILES CHOOSED BY USER: '+str(len(file_names[0])))
        if len(file_names[0]) == 1:
            file, extension1 = splitext(file_names[0][0])
            logging.info('FILE EXTENSION: '+str(extension1))
            if extension1 == '.calbpeak':
                peak_ratio_filename = file_names[0][0]
                # Search for the file with energy factors data that was created in the same calibration
                peak_energy_filename = file+'_CF.calbenergy'
                print(peak_energy_filename)

            elif extension1 == '.calbenergy':
                peak_energy_filename = file_names[0][0]
                # Search for the file with peaks ratio data that was created in the same calibration
                peak_ratio_filename = file.split('_')
                peak_ratio_filename=peak_ratio_filename[0]+'.calbpeak'

        elif len(file_names[0]) == 2:
            try:
                file, extension1 = splitext(file_names[0][0])
                file, extension2 = splitext(file_names[0][1])
                logging.info('FILE EXTENSION :' + extension1)
                logging.info('FILE EXTENSION :' + extension2)
                # Allows that the order of choice of the files is not important
                if extension1 == '.calbpeak' and extension2 == '.calbenergy':
                    peak_ratio_filename = file_names[0][0]
                    peak_energy_filename = file_names[0][1]

                if extension2 == '.calbpeak' and extension1 == '.calbenergy':
                    peak_ratio_filename = file_names[0][1]
                    peak_energy_filename = file_names[0][0]

                if extension1 == extension2:
                    # raises the exception when the files have the same extension
                    logging.exception('BOTH FILES HAVE THE SAME EXTENSION!')
                    raise IndexError('Both files have the extension {}.Please choose two files with different types one with *.calbenergy extension and '
                                     ' the other with*.calbpeak.'.format(extension1))

            except IndexError as e:
                QtWidgets.QMessageBox.warning(None, 'File Errors', 'Error: ' + str(e),
                                              QtWidgets.QMessageBox.Ok)
                logging.exception('Index error: {}'.format(e))
                return

            except OSError as err:
                QtWidgets.QMessageBox.warning(None, 'File Errors', "OS error: {0}".format(err),
                                              QtWidgets.QMessageBox.Ok)
                logging.exception('OS error: {}'.format(err))
                return
        else:
            QtWidgets.QMessageBox.warning(None, 'File error', 'Wrong number of files. \nPlease choose only 2. '
                                                                  'One with the energy information \n and another '
                                                                  'with the ratio information',QtWidgets.QMessageBox.Ok)
            return
        logging.info('Files names:\nPeaks {}\n Energy factors{}'.format(peak_ratio_filename,peak_ratio_filename))

        peakList = []
        try:
            # Reads and stores in array the ratio of each crystal. If the U-board has 32 Crystals in total, the number
            # of crystals is expected to be half of that value. Ex: Peak_matrix shape (16,2)
            file = open(peak_ratio_filename, "r")
            for line in file.readlines():
                peakList.append([])
                for i in line.split():
                    peakList[-1].append(float(i))

            peakArray = np.array(peakList).T
            peakMatrix = np.zeros((numberOfCrystals, 2))
            peakMatrix[:, 0] = peakArray[0][0:numberOfCrystals]
            peakMatrix[:, 1] = peakArray[0][numberOfCrystals: numberOfCrystals * 2]

            # In the case of the list of energy factors, is stored in an array of (1, 32+2). Because 32 correponds to
            # the factor for each crystal and 2 is the adc channel of the peaks (511KeV) of each resistive chain.

            CF = []
            file = open(peak_energy_filename, "r")
            for line in file.readlines():
                CF.append([])
                for i in line.split():
                    CF[-1].append(float(i))

            CFArray = np.array(CF).T
        except FileNotFoundError:
            logging.exception('File path not found.')
            QtWidgets.QMessageBox.critical(None, 'File Error', 'File not found.\nPeak Ratio file path {}\nEnergy Factors file path:{}'.format(peak_ratio_filename, peak_energy_filename),
                                          QtWidgets.QMessageBox.Ok)
            raise FileNotFoundError('Files not found.\nPeak Ratio filename{}\nEnergy Factors file name:{}'.format(peak_ratio_filename, peak_energy_filename))

        file_name = 'teste'
        logging.debug('END')
        return peakMatrix, CFArray, file_name

    def convertion_files_without_header(self, file_name, size_header):
        logging.debug('START')
        logging.info('FILE NAME: '+str(file_name))
        logging.info('SIZE OF HEADER: '+str(size_header))

        with open(file_name, "rb") as binary_file:

            header = np.fromfile(binary_file, dtype=np.float32, count=6)
            binary_file.seek(24)
            dates = np.fromfile(binary_file, dtype='|S1',count=102).astype('|U1')
            dates=dates.tolist()
            dates=''.join(dates)
            binary_file.seek(size_header)
            reading_data = np.fromfile(binary_file,
                                       dtype=[('a', np.int16), ('b', np.int16), ('c', np.int16), ('d', np.int16),
                                              ('e', np.int16), ('f', np.int16), ('g', np.float32)])

        listMode = np.zeros((len(reading_data["a"]), 7))
        listMode[:, 0] = reading_data["a"]
        listMode[:, 1] = reading_data["b"]
        listMode[:, 2] = reading_data["c"]
        listMode[:, 3] = reading_data["d"]
        listMode[:, 4] = reading_data["e"]
        listMode[:, 5] = reading_data["f"]
        listMode[:, 6] = reading_data["g"]

        data4save = np.reshape(listMode, len(listMode) * 7)
        data4save = np.int_(data4save)

        directory = os.path.dirname(os.path.abspath(__file__))
        relative_path = directory + "/Acquisitions/"
        file_name_teste = relative_path + 'Easypet Scan 05 Nov 2017 - 13h 02m 31s converted.easypet'

        file_name=file_name_teste
        output_file = open(file_name, 'wb')

        Version_binary = 'Version 1'
        Version_binary = array('u', Version_binary)
        Version_binary.tofile(output_file)

        size_header = [2 * 9 + 4 + 2000 + 24 + 102]
        size_header = array('i', size_header)
        size_header.tofile(output_file)

        header = header.tolist()
        header = array('f', header)
        header.tofile(output_file)

        dates=array('u', dates)
        dates.tofile(output_file)

        other_info = 'N_Crystals_' + str(16) + ';Aborted:' + 'False' + ' ' * 973
        other_info = array('u', other_info)
        other_info.tofile(output_file)

        teste = struct.pack('hhhhhhf' * len(data4save[6::7]), *data4save)
        output_file.write(teste)
        output_file.close()

        logging.debug('END')



#-------------------------------------------------------
# VALIDATION
#------------------------------------------------------

class validationModule():
    '''
    read_header_v2(): check if the information is well recorded in the header
    '''

    def read_header_v2(fileName):
        [listMode, Version_binary, header, dates, otherinfo, acquisitionInfo, stringdata, systemConfigurations_info,
         energyfactor_info, peakMatrix_info] = binary_data().open(fileName)
        print("\n-- HEADER --")
        print(header)
        print("\n-- DATES --")
        print(dates)
        print("\n-- OTHERINFO --")
        print(otherinfo)
        print("\n-- ACQUISITION PARAMETERS --")
        #acquisitionInfo = acquisitionInfo.replace(',', '\n')
        print(acquisitionInfo)
        print("\n-- SYSTEM CONFIGURATION -- {" + "".join(
            "\n""{!r}: {!r},".format(k, v) for k, v in systemConfigurations_info.items()) + "}\n")
        #print(systemConfigurations_info)
        string = stringdata[0].split(',')
        bot_info = string[0:24]
        bed_info = string[24:48]
        top_info = string[48:72]
        uboard_info = string[72:len(string) + 1]
        stringdata = {
            'TOP': {
                'voltage': top_info[0],
                '360/min full step': top_info[1],
                'current': top_info[2],
                'phase voltage': top_info[3],
                'current velocity': top_info[4],
                'acc': top_info[5],
                'dec': top_info[6],
                'max velocity': top_info[7],
                'min velocity': top_info[8],
                'fullstepspeedTop': top_info[9],
                'holding kval': top_info[10],
                'constant speed kval': top_info[11],
                'acc starting kval': top_info[12],
                'dec starting kval': top_info[13],
                'intspeedTop': top_info[14],
                'start slope': top_info[15],
                'acc slope': top_info[16],
                'dec slope': top_info[17],
                'thermalcompensationTop': top_info[18],
                'OCD_thresholdTop': top_info[19],
                'Stall_thresholdTop': top_info[20],
                'log2(microstepping)': top_info[21],
                'alarmTop': top_info[22],
                'configTop': top_info[23],
            },
            'BOT': {
                'voltage': bot_info[0],
                '360/min full step': bot_info[1],
                'current': bot_info[2],
                'phase voltage': bot_info[3],
                'current velocity': bot_info[4],
                'acc': bot_info[5],
                'dec': bot_info[6],
                'max velocity': bot_info[7],
                'min velocity': bot_info[8],
                'fullstepspeedTop': bot_info[9],
                'holding kval': bot_info[10],
                'constant speed kval': bot_info[11],
                'acc starting kval': bot_info[12],
                'dec starting kval': bot_info[13],
                'intspeedTop': bot_info[14],
                'start slope': bot_info[15],
                'acc slope': bot_info[16],
                'dec slope': bot_info[17],
                'thermalcompensationTop': bot_info[18],
                'OCD_thresholdTop': bot_info[19],
                'Stall_thresholdTop': bot_info[20],
                'log2(microstepping)': bot_info[21],
                'alarmTop': bot_info[22],
                'configTop': bot_info[23]
            },
            'BED': {
                'voltage': bed_info[0],
                '360/min full step': bed_info[1],
                'current': bed_info[2],
                'phase voltage': bed_info[3],
                'current velocity': bed_info[4],
                'acc': bed_info[5],
                'dec': bed_info[6],
                'max velocity': bed_info[7],
                'min velocity': bed_info[8],
                'fullstepspeedTop': bed_info[9],
                'holding kval': bed_info[10],
                'constant speed kval': bed_info[11],
                'acc starting kval': bed_info[12],
                'dec starting kval': bed_info[13],
                'intspeedTop': bed_info[14],
                'start slope': bed_info[15],
                'acc slope': bed_info[16],
                'dec slope': bed_info[17],
                'thermalcompensationTop': bed_info[18],
                'OCD_thresholdTop': bed_info[19],
                'Stall_thresholdTop': bed_info[20],
                'log2(microstepping)': bed_info[21],
                'alarmTop': bed_info[22],
                'configTop': bed_info[23]
            },
            'U-BOARD': {
                'RangeBot': uboard_info[0],
                'RangeTop': uboard_info[1],
                'number of turns': uboard_info[2],
                'side A': uboard_info[3],
                'side B': uboard_info[4],
                'pot1apint': uboard_info[5],
                'Ref 1': uboard_info[6],
                'Ref 2': uboard_info[7],
                'Ref 3': uboard_info[8],
                'Ref 4': uboard_info[9],
                'tacq': uboard_info[10],
                'ResolutionTop': uboard_info[11],
                'ResolutionBot': uboard_info[12]
            }
        }
        print("stringdata BOT = {" + "".join(
            "\n""{!r}: {!r},".format(k, v) for k, v in stringdata['BOT'].items()) + "}\n")
        print("stringdata TOP = {" + "".join(
            "\n""{!r}: {!r},".format(k, v) for k, v in stringdata['TOP'].items()) + "}\n")
        print("stringdata BED = {" + "".join(
            "\n""{!r}: {!r},".format(k, v) for k, v in stringdata['BED'].items()) + "}\n")
        print("stringdata BOARD = {" + "".join(
            "\n""{!r}: {!r},".format(k, v) for k, v in stringdata['U-BOARD'].items()) + "}\n")
        print("\n-- ENERGY FACTORS --")
        print(energyfactor_info)
        print("\n-- PEAK MATRIX --")
        print(peakMatrix_info)
        print("\n-- VERSION BINARY --")
        print(Version_binary)


#fileName = r'C:\Users\info\Documents\GitHub\easyPETtraining\EasyPET training versions\Acquisitions\Easypet Scan 18 Dec 2020 - 18h 11m 20s\Easypet Scan 18 Dec 2020 - 18h 11m 20s.easypet'
#validationModule.read_header_v2(fileName)
