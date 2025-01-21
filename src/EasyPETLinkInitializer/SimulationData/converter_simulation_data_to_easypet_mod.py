import tkinter as tk
from tkinter import filedialog
import time
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import datetime
from array import array
from src.EasyPETLinkInitializer.EasyPETDataReader import binary_data
from src.EasyPETLinkInitializer.EasyPETDataReader import time_discrimination
# from src.EasyPETDataReader import binary_data


class ConverterSimulationDataToEasyPET:
    def __init__(self, file_name=None, easypet_file_init=None, geometry=[16,2], create_folders=True):
        self.file_name = file_name
        self.easypet_file_init = easypet_file_init
        dummy_file = ValidationModule(easypet_file_init)
        self.dummy_list_mode = dummy_file.listMode
        self.dummy_version_binary = dummy_file.Version_binary
        self.dummy_header = dummy_file.header
        self.dummy_dates = dummy_file.dates
        self.dummy_other_info = dummy_file.otherinfo
        self.dummy_acquisitionInfo = dummy_file.acquuisitionInfo
        self.dummy_string_data = dummy_file.stringdata
        self.dummy_systemConfiguration_info = dummy_file.systemConfiguration_info
        self.dummy_energyfactor_info = dummy_file.energyfactor_info
        self.dummy_peakMatrix_info = dummy_file.peakMatrix_info
        self.simulation_file = ReadSimulationData(MyFileName=file_name)
        temp_file_name = os.path.basename(self.file_name)
        temp_file_name = os.path.splitext(os.path.basename(temp_file_name))
        main_folder = os.path.join(os.path.dirname(self.file_name), temp_file_name[0])

        try:
            os.mkdir(main_folder)
            os.mkdir(os.path.join(main_folder, "static_image"))
            # os.mkdir(main_folder)
        except OSError:
            print("Creation of the directory %s failed" % main_folder)
        else:
            print("Successfully created the directory %s " % main_folder)
        self.output_file_name = os.path.join(os.path.dirname(self.file_name), main_folder, temp_file_name[0]+".easypet")
        # shutil.move(self.output_file_name , main_folder)

        self.write_simulation_easypet_format()
        dummy_file = ValidationModule(self.output_file_name)
        print("Finishing")

    def write_simulation_easypet_format(self):
        listmode = self.simulation_file.simulation_listmode
        header = self.simulation_file.header_simulation
        self.dummy_systemConfiguration_info['array_crystal_x'] = self.simulation_file.crystal_array[0]
        self.dummy_systemConfiguration_info['array_crystal_y'] = self.simulation_file.crystal_array[1]
        self.dummy_systemConfiguration_info['angle_bot_rotation'] = 0
        self.dummy_systemConfiguration_info['angle_top_correction'] = 0
        # self.dummy_systemConfiguration_info['crystal_pitch_x'] = 1.5
        # self.dummy_systemConfiguration_info['crystal_pitch_y'] = 1.5
        # self.dummy_systemConfiguration_info['crystal_length'] = 20.0
        # self.dummy_systemConfiguration_info['reflector_exterior_thic'] = 0.2
        # self.dummy_systemConfiguration_info['reflector_interior_A_x'] = 0.05
        # self.dummy_systemConfiguration_info['reflector_interior_A_y'] = 0.05
        # self.dummy_systemConfiguration_info['reflector_interior_B_x'] = 0.05
        # self.dummy_systemConfiguration_info['reflector_interior_B_y'] = 0.05
        # self.dummy_systemConfiguration_info['distance_between_motors'] = 30.0
        self.dummy_systemConfiguration_info['crystal_pitch_x'] = 2
        self.dummy_systemConfiguration_info['crystal_pitch_y'] = 2
        self.dummy_systemConfiguration_info['crystal_length'] = 30.0
        self.dummy_systemConfiguration_info['reflector_exterior_thic'] = 0.2
        self.dummy_systemConfiguration_info['reflector_interior_A_x'] = 0.14
        self.dummy_systemConfiguration_info['reflector_interior_A_y'] = 0.175
        self.dummy_systemConfiguration_info['reflector_interior_B_x'] = 0.14
        self.dummy_systemConfiguration_info['reflector_interior_B_y'] = 0.175
        self.dummy_systemConfiguration_info['distance_between_motors'] = 30.0

        self.dummy_acquisitionInfo["Type of subject"] = "Phantom"
        # self.dummy_acquisitionInfo["Type of subject"] = "Source"
        self.dummy_acquisitionInfo["Acquisition method"] = "Simulation"
        self.dummy_acquisitionInfo["Number of turns"] = header[0]
        self.dummy_acquisitionInfo["Positron Fraction"] = 1
        try:
            self.dummy_acquisitionInfo["ID"] = os.path.basename(self.file_name).split("_")[4]
        except IndexError:
            self.dummy_acquisitionInfo["ID"] = os.path.basename(self.file_name)
        # self.dummy_acquisitionInfo["Volume tracer"] = np.pi*25**2*80*0.001
        # self.dummy_acquisitionInfo["Volume tracer"] = 22.35
        self.dummy_acquisitionInfo["Volume tracer"] = np.pi*22.81**2*75*0.001
        # self.dummy_acquisitionInfo["Volume tracer"] = 4/3*np.pi*0.25**3
        # self.dummy_acquisitionInfo["Volume tracer"] = 4/3*np.pi*0.25**3
        self.dummy_acquisitionInfo["Total Dose"] = 300*37000
        # self.dummy_acquisitionInfo["Half life"] = 8.205e+7 #Na22
        self.dummy_acquisitionInfo["Half life"] = 6582
        # self.dummy_acquisitionInfo["Half life"] = 6582
        self.dummy_acquisitionInfo["Abort bool"] = False
        acquisition_time = (header[5] / header[3] + 1) * (360 / header[1]) * 0.005 * 2
        print("acquisition_time {}".format(acquisition_time))
        acquisition_time = str(datetime.timedelta(seconds=acquisition_time))
        self.dummy_acquisitionInfo["Start date time"] = "26.1.22 {}".format(acquisition_time)
        self.dummy_acquisitionInfo["Residual Activity"] = 0
        self.dummy_acquisitionInfo["Injection date time"] = "26.1.22 {}".format(acquisition_time)
        self.dummy_acquisitionInfo["End date time"] = "26.1.22 {}".format(acquisition_time)

        acquisition_time_for = time.strftime("%Hh %Mm %Ss",time.strptime(acquisition_time, "%H:%M:%S"))
        self.dummy_acquisitionInfo["Acquisition start time"] = "26 Jan 2022 - {}".format(acquisition_time_for)
        time_indexes = (time_discrimination(listmode, simulationFile=True))[0]
        # diff_time_indexes = np.diff(time_indexes)
        # time_indexes = time_indexes[diff_time_indexes > 1]
        time_indexes = [int(el) for el in time_indexes]
        if len(time_indexes) == 0:
            time_indexes = [len(listmode)]
        self.dummy_acquisitionInfo["Turn end index"] = time_indexes# remover
        # self.dummy_acquisitionInfo["Turn end index"] = [0, 8133158, 9129935, 10023839, 10543612, 11023706, 11415573]

        # self.dummy_acquisitionInfo["Turn end index"] = [0, 10572579, 18629056, 24460951, 26762214, 27758991, 28652895, 29172668, 29652762, 30044629]
        # self.dummy_acquisitionInfo["Turn end index"] = [0, 23679418, 43625879, 60153401, 73557587, 84130166, 92186643, 98018538, 100319801, 101316578, 102210482, 102730255, 103210349, 103602216]
        crystalMatrix = np.array(listmode[:, 2:4], dtype='int16', copy=False)
        data4save = np.round(listmode[:, 0:2],4).astype(dtype='int16', copy=False)
        EA_Corrected = listmode[:, 4].astype(dtype='int16', copy=False)
        EB_Corrected = listmode[:, 5].astype(dtype='int16', copy=False)
        timestamp_a = listmode[:, 6].astype(dtype='float64', copy=False)
        timestamp_b = listmode[:, 7].astype(dtype='float64', copy=False)

        records = np.rec.fromarrays(
            (data4save[:, 0], data4save[:, 1], crystalMatrix[:, 0], crystalMatrix[:, 1],
             EA_Corrected, EB_Corrected, timestamp_a, timestamp_b),
            names=('bot', 'top', 'id1', 'id2', 'EA', 'EB', 'timestamp_A', 'timestamp_B'))

        # records = np.rec.fromarrays(
        #     (data4save[:, 0], data4save[:, 1], crystalMatrix[:, 0], crystalMatrix[:, 1],
        #      EA_Corrected, EB_Corrected, timestamp_a),
        #     names=('bot', 'top', 'id1', 'id2', 'EA', 'EB', 'timestamp_A'))

        header = array('f', header)
        dates = array('u', self.dummy_dates)

        Version_binary = 'Version 3'
        Version_binary = array('u', Version_binary)
        acquisitionParameters = json.dumps(self.dummy_acquisitionInfo)
        acquisition_info = array('u', acquisitionParameters)
        acquisition_info_size = [len(acquisitionParameters)]
        acquisition_info_size = array('i', acquisition_info_size)

        stringdata = json.dumps(self.dummy_string_data)
        stringdata = array('u', stringdata)
        stringdata_size = [len(stringdata)]
        stringdata_size = array('i', stringdata_size)

        # systemConfigurations = {
        #     'serial_number': serial_number,
        #     'u_board_version': reading_hardware_parameters.u_board_version,
        #     'module_control': reading_hardware_parameters.module_control,
        #     'array_crystal_x': reading_hardware_parameters.array_crystal_x,
        #     'array_crystal_y': reading_hardware_parameters.array_crystal_y,
        #     'angle_bot_rotation': reading_hardware_parameters.angle_bot_rotation,
        #     'angle_top_correction': reading_hardware_parameters.angle_top_correction,
        #     'multiplexed': reading_hardware_parameters.multiplexed,
        #     'reading_method': reading_hardware_parameters.reading_method,
        #     'number_adc_channel': reading_hardware_parameters.number_adc_channel,
        #     'bed_version': reading_hardware_parameters.bed_version,
        #     'bed_diameter': reading_hardware_parameters.bed_diameter,
        #     'pc_communication': reading_hardware_parameters.pc_communication,
        #     'baudrate': reading_hardware_parameters.baudrate,
        #     'motor_bot': reading_hardware_parameters.motor_bot,
        #     'motor_top': reading_hardware_parameters.motor_top,
        #     'bed_motor': reading_hardware_parameters.bed_motor,
        #     'fourth_motor': reading_hardware_parameters.fourth_motor,
        #     'capable4CT': reading_hardware_parameters.capable4CT,
        #     'crystal_pitch_x': reading_hardware_parameters.crystal_pitch_x,
        #     'crystal_pitch_y': reading_hardware_parameters.crystal_pitch_y,
        #     'crystal_length': reading_hardware_parameters.crystal_length,
        #     'reflector_exterior_thic': reading_hardware_parameters.reflector_exterior_thic,
        #     'reflector_interior_A_x': reading_hardware_parameters.reflector_interior_A_x,
        #     'reflector_interior_A_y': reading_hardware_parameters.reflector_interior_A_y,
        #     'reflector_interior_B_x': reading_hardware_parameters.reflector_interior_B_x,
        #     'reflector_interior_B_y': reading_hardware_parameters.reflector_interior_B_y,
        #     'distance_between_motors': reading_hardware_parameters.distance_between_motors,
        #     'distance_between_crystals': reading_hardware_parameters.distance_between_crystals,
        #     'centercrystals2topmotor_x_sideA': reading_hardware_parameters.centercrystals2topmotor_x_sideA,
        #     'centercrystals2topmotor_x_sideB': reading_hardware_parameters.centercrystals2topmotor_x_sideB,
        #     'centercrystals2topmotor_y': reading_hardware_parameters.centercrystals2topmotor_y
        # }

        systemConfigurations = json.dumps(self.dummy_systemConfiguration_info)
        systemConfigurations_info = array('u', systemConfigurations)
        systemConfigurations_info_size = [len(systemConfigurations_info)]
        systemConfigurations_info_size = array('i', systemConfigurations_info_size)

        # [peakMatrix, calibration_file, energyfactor] = calibration_points_init(
        #     [int(reading_hardware_parameters.array_crystal_x), int(reading_hardware_parameters.array_crystal_y)])

        energyfactor = self.dummy_energyfactor_info
        # energyfactor = energyfactor[0].tolist()
        # energyfactor_str = str(energyfactor).strip('[]')
        # energyfactor_str = energyfactor_str.replace(" ", "")
        energyfactor_size = array('i', [len(energyfactor)])
        energyfactor_info = array('u', energyfactor)

        peakMatrix_str = self.dummy_peakMatrix_info
        # peakMatrix = peakMatrix.flatten('F')
        # peakMatrix_str = str(peakMatrix).strip('[]')

        peakMatrix_size = array('i', [len(peakMatrix_str)])
        peakMatrix_info = array('u', peakMatrix_str)

        size_header = [2 * 9 + 4 + 24 + 102 + 4 + acquisition_info_size[0] * 2 + 4 + stringdata_size[0] * 2 +
                       4 + systemConfigurations_info_size[0] * 2 + 4 + energyfactor_size[0] * 2 +
                       4 + peakMatrix_size[0] * 2]
        size_header = array('i', size_header)

        with open(self.output_file_name, 'wb') as output_file:
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


class ReadSimulationData:
    def __init__(self, MyFileName=None, numpy_file=True):
        self.MyFileName = MyFileName
        self.ListmodeData = None
        self.StepsRang = [None]*4
        self.initial_data = {}
        self.crystal_array = [32, 2]
        if numpy_file:
            self.read_numpy_data()
        else:
            self.binary_file()
        self.simulation_listmode = self.ListmodeData
        self.header_simulation = [self.StepsRang[0], self.StepsRang[1], 1, self.StepsRang[2],1,self.StepsRang[3]]

    # def validation_original_data(self):
    def binary_file(self):
        MyFile = open(self.MyFileName, "rb")  # read data

        FileDataString = MyFile.read()  # read all data into a string

        MyArray = array("d", FileDataString)

        NoOfLORs = int(len(MyArray) / 7)
        ShapedData = np.reshape(MyArray, (7, NoOfLORs))

        ListmodeData = np.flipud(np.rot90(ShapedData))  # Bot Top id1 id2 E1 E2 T1 ; Dimention = NoOfLORs * 7
        ListmodeData[:, 2:4] = ListmodeData[:, 2:4] + 1
        MyFileName = os.path.basename(self.MyFileName)
        self.StepsRang = MyFileName.split('_')
        self.StepsRang = self.StepsRang[0:3]
        for i in range(len(self.StepsRang)):
            self.StepsRang[i] = float(self.StepsRang[i])
        print("ListmodeData: Bot Top id1 id2 E1 E2 T1")

        print("===================================")
        print("BotStep =", self.StepsRang[0])
        print("TopStep =", self.StepsRang[1])
        print("TopRange =", self.StepsRang[2])
        print("============== E N D ==============")

    def removeErrorsEasyPET(self):
        self.ListmodeData = self.ListmodeData[self.ListmodeData[:, 4] > 0]
        self.ListmodeData = self.ListmodeData[self.ListmodeData[:, 5] > 0]
        self.ListmodeData = self.ListmodeData[self.ListmodeData[:, 0] > 0]
        self.ListmodeData = self.ListmodeData[self.ListmodeData[:, 0] <= 360]

    def read_numpy_data(self):
        self.ListmodeData = np.load(self.MyFileName)
        self.ListmodeData[:, 4] *= 1000
        self.ListmodeData[:, 5] *= 1000

        top = np.copy(self.ListmodeData[:, 0])
        bot = np.copy(self.ListmodeData[:, 1])
        self.ListmodeData[:, 0] = bot
        self.ListmodeData[:, 1] = top
        self.ListmodeData[:, 2:4] = self.ListmodeData[:, 2:4] + 1
        if self.ListmodeData.shape[1] == 6:
            self.ListmodeData = np.vstack((self.ListmodeData.T, np.arange(len(self.ListmodeData)).T)).T

        _dir = os.path.dirname(self.MyFileName)
        name, tail = os.path.split(self.MyFileName)[1].split(".")
        name = "_".join(name.split("_")[0:-1])
        parameters_file_path = os.path.join(_dir, "{}_parameters.txt".format(name))

        with open(parameters_file_path, "r") as file:
            for line in file:
                arg = line.split("=")

                if len(arg) == 2:
                    self.initial_data[arg[0]] = float(arg[1])

        self.StepsRang[0] = self.initial_data['numberOfTurns']
        self.StepsRang[1] = self.initial_data['botStep']
        self.StepsRang[2] = self.initial_data['topStep']
        self.StepsRang[3] = self.initial_data['topAng']
        # self.removeErrorsEasyPET()

        self.ListmodeData[:, 1] = (self.ListmodeData[:, 1]+self.initial_data['topAng']/2)/self.initial_data['topStep']
        self.ListmodeData[:, 0] = self.ListmodeData[:, 0]/self.initial_data['botStep']
        # plt.hist(self.ListmodeData[:, 1], 1000, range=(0,1000))
        # plt.show()

        # diff_top = np.ceil(np.abs(np.diff(self.ListmodeData[:, 1])))
        # diff_bot = np.abs(np.diff(self.ListmodeData[:, 0]))
        # time = 0
        # for t in range(len(diff_top)):
        #     self.ListmodeData[t, 6] = time
        #     if diff_top[t] != 0 or diff_bot[t] != 0:
        #         time += 0.005
        #
        # self.ListmodeData[:, 6] = self.ListmodeData[:, 7] # alterar quando vier corrigido

        print(self.initial_data)


class ValidationModule:
    '''
    read_header_v2(): check if the information is well recorded in the header
    '''
    def __init__(self, fileName=None):
        self.fileName = fileName

        self.listMode = None
        self.Version_binary = None
        self.header = None
        self.dates = None
        self.otherinfo = None
        self.acquuisitionInfo = None
        self.stringdata = None
        self.systemConfiguration_info = None
        self.energyfactor_info = None
        self.peakMatrix_info = None
        self.read_header_v2()

    def read_header_v2(self):

        [listMode, Version_binary, header, dates, otherinfo, acquisitionInfo, stringdata,
         systemConfigurations_info,
         energyfactor_info, peakMatrix_info] = binary_data().open(self.fileName)
        self.listMode = listMode
        self.Version_binary = Version_binary
        self.header = header
        self.dates = dates
        self.otherinfo = otherinfo
        self.acquuisitionInfo = acquisitionInfo
        self.stringdata = stringdata
        self.systemConfiguration_info = systemConfigurations_info
        self.energyfactor_info = energyfactor_info
        self.peakMatrix_info = peakMatrix_info

        print("\n-- HEADER --")
        print(header)
        print("\n-- DATES --")
        print(dates)
        print("\n-- OTHERINFO --")
        print(otherinfo)
        print("\n-- ACQUISITION PARAMETERS --")
        # acquisitionInfo = acquisitionInfo.replace(',', '\n')
        print(acquisitionInfo)
        print("\n-- SYSTEM CONFIGURATION -- {" + "".join(
            "\n""{!r}: {!r},".format(k, v) for k, v in systemConfigurations_info.items()) + "}\n")
        # print(systemConfigurations_info)
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


if __name__ == "__main__":
    easy_pet_init = "C:\\Users\\pedro.encarnacao\\Documents\\GitHub\\easyPETtraining\\EasyPET training versions\\Acquisitions\\Easypet Scan 29 Dec 2020 - 19h 03m 00s\\Easypet Scan 29 Dec 2020 - 19h 03m 00s.easypet"
    easy_pet_init = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "dataFiles", "easypet_dummy_scan")
    print(easy_pet_init)
    for file in os.listdir(easy_pet_init):
        print(file)
        if file.endswith(".easypet"):
            init_file = os.path.join(easy_pet_init, file)
            print(os.path.join(easy_pet_init, file))

    root = tk.Tk()
    root.withdraw()
    MyFileName = filedialog.askopenfilename()
    # from src.GateLink.RootToTor import GenerateCoincidencesAndAnhnilationsPositions
    #
    # coinc = GenerateCoincidencesAndAnhnilationsPositions(
    #     filename='/home/crispim/Documentos/Simulations/easyPET_part0(1).root')
    # coinc.readRoot()
    # arrays_keys = ['sourcePosX', 'sourcePosY', 'sourcePosZ', 'time', 'baseID', 'energy', 'globalPosX', 'globalPosY',
    #                'globalPosZ', 'level1ID', 'level2ID', 'level3ID']
    # coinc.setArraysToConvert(coinc.singlesScanner1, arrays_keys)
    # coinc.setArraysToConvert(coinc.singlesScanner2, arrays_keys)
    # coinc.find_coincidences()



    ConverterSimulationDataToEasyPET(file_name=MyFileName, easypet_file_init=init_file)