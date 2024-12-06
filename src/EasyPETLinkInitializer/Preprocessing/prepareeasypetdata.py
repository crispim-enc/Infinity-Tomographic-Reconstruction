import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from src.EasyPETLinkInitializer.EasyPETDataReader.time_discrimination import time_discrimination
from src.EasyPETLinkInitializer.EasyPETDataReader.lmodebin import binary_data
from src.EasyPETLinkInitializer.EasyPETDataReader import calibration_points_init
from src.EasyPETLinkInitializer.SimulationData import SimulationStatusData
from src.EasyPETLinkInitializer.Utilities.exceptions import OnlyOneIncompleteTurn


class PrepareEasyPETdata:
    def __init__(self, study_file=None, reconstruction_data_type=".easypet", energy_window=None,
                 remove_first_crystals_row=False, remove_last_crystal_row=False,
                 remove_peripheral_crystal_on_resistive_chain=False, only_left_side_crystals=False,
                 only_right_side_crystals=False, right_left_crystals=False, left_right_crystals=False,
                 parameters_2D_cut=None, adc_cuts=None,
                 threshold_ratio=0.60, top_correction_angle=0.0, bot_rotation_angle=0, cut_acc_ramp=0.0,
                 swap_sideAtoB=True,
                 crystal_geometry=None, save_validation_data=True, save_spectrum_file=False,
                 remove_incomplete_turn=True, mutex=None, live_acquisition=False, signals_interface=None,
                 simulation_file=False, join_turns=None, coincidence_window=None):
        # top_correction_angle = 0.705
        # if reconstruction_data_type == ".easypetoriginal":
        if join_turns is None:
            join_turns = [False, 16]
        self.join_turns = join_turns
        self.remove_first_crystals_row = remove_first_crystals_row
        self.remove_last_crystal_row = remove_last_crystal_row
        self.remove_peripheral_crystal_on_resistive_chain = remove_peripheral_crystal_on_resistive_chain
        self.only_left_side_crystals = only_left_side_crystals
        self.only_right_side_crystals = only_right_side_crystals
        self.right_left_crystals = right_left_crystals
        self.left_right_crystals = left_right_crystals
        self.swap_sideAtoB = swap_sideAtoB
        self.save_validation_data = save_validation_data
        self.remove_incomplete_turn = remove_incomplete_turn
        self.cut_acc_ramp = cut_acc_ramp
        self.live_acquisition = live_acquisition
        self.fov_real = None
        self._coincidence_window = coincidence_window
        if adc_cuts is None:
            adc_cuts = [50, 50, 50, 50]
        if crystal_geometry is None:
            crystal_geometry = [32, 2]
        if parameters_2D_cut is None:
            parameters_2D_cut = [1200, 600, 1200, 600]
        if energy_window is None:
            energy_window = [400, 550]
        if signals_interface is not None:
            signals_interface.trigger_update_label_reconstruction_status.emit("Preparing EasyPET data")
        self.adc_cuts = adc_cuts

        self.crystal_geometry = crystal_geometry
        self.parameters_2D_cut = parameters_2D_cut
        self.energy_window = energy_window
        self.top_correction_angle = top_correction_angle
        self.bot_rotation_angle = bot_rotation_angle
        self.reconstruction_data_type = reconstruction_data_type
        self.path_data_validation = None
        self.save_spectrum_file = save_spectrum_file
        self.crystal_geometry = crystal_geometry
        self.simulation_file = simulation_file
        # self.path_preparing_file = os.path.join(study_path, "")
        self.list_of_figures = ["Spectrum Original", "Ratio", "Spectrum", "Ratio2D", "Selectable", "Time"]
        self.study_file = study_file
        self.study_file_original = os.path.join(os.path.dirname(self.study_file),
                                                "{} Original data.easypetoriginal".format(
                                                    os.path.basename(os.path.dirname(self.study_file))))

        self.error_prepare_easypet = None
        self.error_msg = ""
        self.error_title = ""

        # [self.listMode, Version_binary, self.header, dates, otherinfo] = binary_data().open(self.study_file)
        if mutex is not None:
            mutex.lock()
        try:
            [self.listMode, self.Version_binary, self.header, self.dates, self.otherinfo, self.acquisitionInfo,
             self.stringdata, self.systemConfigurations_info, self.energyfactor_info,
             self.peakMatrix_info] = binary_data().open(self.study_file)
            # self.listMode[:,2] = np.abs(64-self.listMode[:,2])
            # self.listMode[:,3] = np.abs(64-self.listMode[:,3])
            # self.listMode = self.listMode[np.array(np.round(((self.listMode[:,4]) / (self.header[1]/self.header[2])), 0)) %128==64]

        except FileNotFoundError:
            raise FileNotFoundError

        for key, value in self.acquisitionInfo.items():
            # name, age = value
            print("{}: {} ".format(key, value))
        print("-------------------")
        for key, value in self.systemConfigurations_info.items():
            # name, age = value
            print("{}: {} ".format(key, value))


        # print(self.systemConfigurations_info)
        # self.listMode = OpenSimulationData().listMode

        if simulation_file is False:
            self.listMode_original = binary_data().open_original_data(self.study_file_original)

        # self.Version_binary = "Version 1"  # for files with version 2 tag but dont have the correct fields
        if self.Version_binary == "Version 1":
            time_indexes = time_discrimination(self.listMode)
            self.acquisitionInfo["Turn end index"] = time_indexes
            self.acquisitionInfo["Number of turns"] = len(time_indexes) - 1
            # time_indexes = [0, len(self.listMode)]  # remover
        if self.Version_binary == "Version 2" or self.Version_binary == "Version 3" :
            self.crystal_geometry = [int(self.systemConfigurations_info["array_crystal_x"]),
                                     int(self.systemConfigurations_info["array_crystal_y"])]

            time_indexes = self.acquisitionInfo["Turn end index"]
            # if simulation_file:
            #     time_indexes = [0, len(self.listMode)]
            #     time_indexes = list(time_discrimination(self.listMode))# remover

            if isinstance(time_indexes, str):
                indexes = self.acquisitionInfo['Turn end index'].split(' ')
                time_indexes = [None] * (len(indexes))
                time_indexes[0] = 0
                for i in range(1, len(time_indexes)):
                    time_indexes[i] = int(indexes[i])
                time_indexes = np.sort(time_indexes)
            time_indexes.insert(0, 0)

        self.time_indexes = time_indexes
        print("top {}".format(top_correction_angle))
        try:
            print("{}: Tempo de frame: {} s".format(self.acquisitionInfo['Acquisition start time'],
                                                    self.listMode[self.acquisitionInfo['Turn end index'][1] - 1, 6] -
                                                    self.listMode[self.acquisitionInfo['Turn end index'][0], 6]))
        except IndexError:
            pass
        # if self.Version_binary == "Version 1":

        if mutex is not None:
            mutex.unlock()
        # self.listMode = self.listMode[time_indexes[0]:time_indexes[2], :]
        [self.peakMatrix, calibration_file, energyfactor] = calibration_points_init(self.crystal_geometry)
        if reconstruction_data_type == ".easypetoriginal":
            print("Preparing data for easypet")
            self._use_raw_data(adc_cuts, parameters_2D_cut, threshold_ratio)
            self.time_indexes = self.acquisitionInfo["Turn end index"]
            self.listMode[:, 4] += float(self.systemConfigurations_info["angle_bot_rotation"])
            # self.listMode[np.ceil(np.array(np.round((self.listMode[:, 4]), 4)) / (self.header[3] / 32)) % 128 == 0] -= 0.1
            # self.listMode = self.listMode[np.ceil(np.array(np.round((self.listMode[:, 4]), 4)) / (self.header[1] / self.header[2])) % 128!=64]
            # self.listMode[:,5] += 0.07
            swap_sideAtoB = True
        # self.listMode = self.listMode[time_indexes[8]::, :]
        # self.listMode[:,5] *= -1
        if join_turns[0]:
            _sum_turn_indexes = len(self.time_indexes) // join_turns[1]
            time_indexes_temp = np.array(self.time_indexes)
            self.time_indexes = time_indexes_temp[::int(_sum_turn_indexes)].tolist()

    def dataRemove(self):
        if self.remove_incomplete_turn:
            # otherinfo= list(self.otherinfo)
            # otherinfo = [i for i in otherinfo if i.startswith("Aborted")]
            # aborted_acquisition = otherinfo[0].split(':')
            # aborted_acquisition = aborted_acquisition[1]
            try:
                aborted_acquisition = self.acquisitionInfo["Abort bool"]
            except KeyError:
                otherinfo = list(self.otherinfo)
                otherinfo = [i for i in otherinfo if i.startswith("Aborted")]
                aborted_acquisition = otherinfo[0].split(':')
                aborted_acquisition = aborted_acquisition[1]
                print("Aborted acquisition: {}".format(aborted_acquisition))

            aborted_acquisition = aborted_acquisition.lower() in ("yes", "true", "t", "1")
            # aborted_acquisition = True

            if aborted_acquisition:
                if len(self.time_indexes) <= 2:
                    print("Acquisition with only one incomplete turn")
                    self.error_msg = "Scanning with only one imcomplete turn"
                    self.error_title = " Error invalid data"
                    self.error_prepare_easypet = True
                    raise OnlyOneIncompleteTurn("Acquire at least one complete turn")

                self.listMode = self.listMode[self.time_indexes[0]:self.time_indexes[-1], :]
                self.time_indexes = self.time_indexes[0:-1]
                # if self.Version_binary == "Version 1":
                #     # time_indexes = time_discrimination(self.listMode)
                #     # print("Number _turns: {}".format(len(time_indexes)))
                #
                # elif self.Version_binary == "Version 2":
                #     print("Version 2")

        top_cut_range = ((1 - self.cut_acc_ramp) * self.header[5] / 2)
        time_indexes = self.time_indexes
        list_mode_list = [None] * (len(time_indexes) - 1)
        # list_mode_list = [None] * len(range(0,len(time_indexes)-1,2))
        # list_mode_list = [None]*(int(len(time_indexes)/2))
        number_of_events_after_cuts = 0
        # for i in range(0, len(time_indexes) - 1, 2):
        #for simulation cut for and keep decay correction correct
        if self.cut_acc_ramp != 0 and self.simulation_file:
            number_of_positions_top_lost = ((self.header[5] / 2)-top_cut_range)/ self.header[3]
            time_gain = 0.005 * number_of_positions_top_lost
            diff = np.diff(self.listMode[:, 4])
            diff[diff != 0] = 1
            cumsum = np.cumsum(diff) * time_gain
            self.listMode[1:, 6] = self.listMode[1:, 6] - cumsum
            self.listMode[1:, 7] = self.listMode[1:, 7] - cumsum

        for i in range(0, len(time_indexes) - 1):

            list_mode_partial = self.listMode[time_indexes[i]:time_indexes[i + 1], :]
            list_mode_partial = list_mode_partial[np.abs(list_mode_partial[:, 5]) < top_cut_range]

            # if (i+1) % 2 == 0:
            #     list_mode_partial[:, 5] = -list_mode_partial[:, 5]
            list_mode_partial[:, 5] = list_mode_partial[:, 5] + self.top_correction_angle
            list_mode_partial[:, 4] = list_mode_partial[:, 4] + self.bot_rotation_angle

            index_ea = np.where(
                (list_mode_partial[:, 0] < self.energy_window[0]) | (list_mode_partial[:, 1] < self.energy_window[0]))
            index_eb = np.where(
                (list_mode_partial[:, 0] > self.energy_window[1]) | (list_mode_partial[:, 1] > self.energy_window[1]))
            union_indexes_intersection = np.union1d(index_ea, index_eb)
            list_mode_partial = np.delete(list_mode_partial, union_indexes_intersection, axis=0)

            index_ida = np.where((list_mode_partial[:, 2] < 0) | (list_mode_partial[:, 3] < 0))
            index_idb = np.where((list_mode_partial[:, 2] > self.crystal_geometry[0] * self.crystal_geometry[1]) | (
                    list_mode_partial[:, 3] > self.crystal_geometry[0] * self.crystal_geometry[1]))
            union_indexes_intersection = np.union1d(index_ida, index_idb)
            list_mode_partial = np.delete(list_mode_partial, union_indexes_intersection, axis=0)

            # index_bot = np.where(self.listMode[:, 4] < -360)
            # list_mode_partial = list_mode_partial[list_mode_partial[:, 4] > -360,:]
            # list_mode_partial = list_mode_partial[list_mode_partial[:, 4] < 720,:]

            # self.listMode = self.listMode[self.listMode[:, 2] == self.listMode[:, 3],:]
            # self.listMode = self.listMode[(self.listMode[:,2]+1) % 2 == 0]
            # self.listMode = self.listMode[(self.listMode[:,3]) % 2 == 0]

            # # remove IDs diff larger than 30
            # index_ida = np.where(np.abs(list_mode_partial[:, 2] - list_mode_partial[:, 3]) > 32)
            # list_mode_partial = np.delete(list_mode_partial, index_ida, axis=0)

            # index_ida=np.where((self.listMode[:,2]<=crystal_geometry[0]) & (self.listMode[:,3]<=crystal_geometry[0]))
            # index_idb=np.where((self.listMode[:,2]>crystal_geometry[0]) & (self.listMode[:,3]>crystal_geometry[0]))
            # union_indexes_intersection = np.union1d(index_ida, index_idb)
            # self.listMode=self.listMode[union_indexes_intersection]
            if self.remove_first_crystals_row:
                list_mode_partial = list_mode_partial[list_mode_partial[:, 2] != np.min(list_mode_partial[:, 2])]
                list_mode_partial = list_mode_partial[list_mode_partial[:, 2] != np.min(list_mode_partial[:, 2])]
                list_mode_partial = list_mode_partial[list_mode_partial[:, 3] != np.min(list_mode_partial[:, 3])]
                list_mode_partial = list_mode_partial[list_mode_partial[:, 3] != np.min(list_mode_partial[:, 3])]

            if self.remove_last_crystal_row:
                list_mode_partial = list_mode_partial[list_mode_partial[:, 2] != np.max(list_mode_partial[:, 2])]
                list_mode_partial = list_mode_partial[list_mode_partial[:, 2] != np.max(list_mode_partial[:, 2])]
                list_mode_partial = list_mode_partial[list_mode_partial[:, 3] != np.max(list_mode_partial[:, 3])]
                list_mode_partial = list_mode_partial[list_mode_partial[:, 3] != np.max(list_mode_partial[:, 3])]

            if self.remove_peripheral_crystal_on_resistive_chain:
                try:
                    list_mode_partial = list_mode_partial[list_mode_partial[:, 2] != np.min(list_mode_partial[:, 2])]
                    list_mode_partial = list_mode_partial[list_mode_partial[:, 3] != np.min(list_mode_partial[:, 3])]
                    list_mode_partial = list_mode_partial[list_mode_partial[:, 2] != np.max(list_mode_partial[:, 2])]
                    list_mode_partial = list_mode_partial[list_mode_partial[:, 3] != np.max(list_mode_partial[:, 3])]
                except ValueError:
                    pass
            # list_mode_partial_original = binary_data().open_original_data(filename)
            if self.only_left_side_crystals:
                list_mode_partial = list_mode_partial[list_mode_partial[:, 2] % 2 == 0, :]
                list_mode_partial = list_mode_partial[list_mode_partial[:, 3] % 2 == 0, :]

            if self.right_left_crystals:
                list_mode_partial = list_mode_partial[(list_mode_partial[:, 2] + 1) % 2 == 0, :]
                list_mode_partial = list_mode_partial[list_mode_partial[:, 3] % 2 == 0, :]

            if self.only_right_side_crystals:
                list_mode_partial = list_mode_partial[(list_mode_partial[:, 2] + 1) % 2 == 0, :]
                list_mode_partial = list_mode_partial[(list_mode_partial[:, 3] + 1) % 2 == 0, :]

            if self.left_right_crystals:
                list_mode_partial = list_mode_partial[list_mode_partial[:, 2] % 2 == 0, :]
                list_mode_partial = list_mode_partial[(list_mode_partial[:, 3] + 1) % 2 == 0, :]

            if self._coincidence_window is not None:
                list_mode_partial = self.applyCoincidenceWindow(list_mode_partial, self._coincidence_window)  # ps
            list_mode_list[i] = list_mode_partial
            number_of_events_after_cuts += len(list_mode_partial)
        self.listMode = np.zeros((number_of_events_after_cuts, self.listMode.shape[1]))
        begin_event = 0
        c = 1
        for partial in list_mode_list:
            end_event = begin_event + len(partial)
            self.listMode[begin_event:end_event, :] = partial
            time_index_to_save = end_event - 1
            if time_index_to_save < 0:
                time_index_to_save = 0
            time_indexes[c] = time_index_to_save
            begin_event = end_event
            c += 1
        self.acquisitionInfo["Turn end index"] = time_indexes

        if self.swap_sideAtoB:
            print(self.swap_sideAtoB)
            a = np.copy(self.listMode[:, 2])
            b = np.copy(self.listMode[:, 3])
            ea = np.copy(self.listMode[:, 0])
            eb = np.copy(self.listMode[:, 1])
            self.listMode[:, 2] = b
            self.listMode[:, 3] = a
            self.listMode[:, 0] = eb
            self.listMode[:, 1] = ea

        # self.listMode = self.listMode[self.listMode[:, 3] % 2 != 0, :]
        # self.listMode=self.listMode[self.listMode[:,2]==self.listMode[:,3]]
        # self.listMode=self.listMode[self.listMode[:,3]==[1,2]]
        # self.listMode=self.listMode[self.listMode[:,4]==self.listMode[0,4]]
        # self.listMode = self.listMode[0:5,:]
        # self.listMode[:,2] = 32
        # self.listMode[:,3] = 31
        # self.listMode[:,4] = 90
        # self.listMode[:,5] = 0
        # # self.listMode=self.listMode[self.listMode[:,5]==np.min((self.listMode[:,5]))]

        # self._creat_test_geometry()


        # self.listMode = self.listMode[time_indexes[6]:]
        # time_indexes = time_indexes[15:-1]
        if self.save_validation_data:
            if self.simulation_file:
                status = SimulationStatusData(listMode=self.listMode, study_file=self.study_file,
                                              crystal_geometry=self.crystal_geometry, acquisitionInfo=self.acquisitionInfo)

                # if self._coincidence_window is not None:
                #     self.listMode = status.applyCoincidenceWindow(self.listMode, self._coincidence_window) #ps
                status.listMode = self.listMode
                status.validation_status()
            else:
                self._create_figures_data_selected()


        self.fov_real = np.sin(
            np.radians((np.max(self.listMode[:, 5]) + np.abs(np.min(self.listMode[:, 5]))) / 2 + 4 * np.degrees(
                np.arctan(1.175 / self.systemConfigurations_info["distance_between_crystals"])))) * \
                        self.systemConfigurations_info[
                            "distance_between_crystals"]  # por dependente do ficheiro da geometria
        print("Fov Real: {} mm".format(self.fov_real))

        # self.fov_real = 60 #for testing
        # print("Number of Events: {}".format(len(self.listMode)))
        # self.listMode[0,6] = self.listMode[-1,6]
        # self.listMode = self.listMode[0:1,:]
        # self.listMode[:,2] = 1
        # self.listMode[:,3] = 1
        # self._organize_matrix_for_coaslence()
        # self.listMode = self.listMode[:80, :]
        # self.listMode[:,0] = 511
        # self.listMode[:,1] = 511
        # self.listMode[:,2] = 63
        # self.listMode[:,3] = 1
        # self.listMode[:,4] = np.repeat(np.linspace(0,360,8),10)
        # self.listMode[:,5] = np.tile(np.linspace(-45,45,10),8)

        # self.listMode = self.listMode[:1, :]
        # self.listMode[:,0] = 511
        # self.listMode[:,1] = 511
        # self.listMode[:,2] = 31
        # self.listMode[:,3] = 1
        # self.listMode[:,4] = 0
        # self.listMode[:,5] = 0

    def applyCoincidenceWindow(self,listmode, value):
        value *= 10 ** -12 #convert to s
        diff_time = np.abs(listmode[:, 6] - listmode[:, 7])
        _all_events = diff_time[diff_time < value]
        listmode = listmode[diff_time < value]
        return listmode

    def _use_raw_data(self, adc_cuts, parameters, threshold_ratio):
        parameter_cut_a = parameters[0]
        parameter_cut_b = parameters[1]
        parameter_cut_c = parameters[2]
        parameter_cut_d = parameters[3]
        listMode = self.listMode_original
        if self.Version_binary == "Version 1":
            time_indexes = time_discrimination(self.listMode_original, datatype=".easypetoriginal")
            # time_indexes = [0, len(self.listMode)]  # remover
        if self.Version_binary == "Version 2":

            time_indexes = self.acquisitionInfo["Turn end index original data"]
            # time_indexes = time_discrimination(self.listMode_original, datatype=".easypetoriginal")
            # time_indexes = list(time_indexes)
            # self.acquisitionInfo["Abort bool"] ="False"
            # time_indexes = [len(listMode)]
            time_indexes.insert(0, 0)
            if isinstance(time_indexes, str):
                indexes = self.acquisitionInfo['Turn end index original data'].split(' ')
                time_indexes = [None] * (len(indexes))
                time_indexes[0] = 0
                for i in range(1, len(time_indexes)):
                    time_indexes[i] = int(indexes[i])
                time_indexes = np.sort(time_indexes)

        list_mode_list = [None] * (len(time_indexes) - 1)
        list_mode_original_list = [None] * (len(time_indexes) - 1)
        number_of_events_after_cuts = 0
        number_of_events_after_cuts_original = 0
        # listMode[:,4][(listMode[:,5] / 64 + 1) % 2 == 0] += 64
        # listMode= listMode[(listMode[:,5] / 64 + 1) % 2 == 0]

        for i in range(len(time_indexes) - 1):
            list_mode_partial = listMode[time_indexes[i]:time_indexes[i + 1], :]
            # list_mode_partial = list_mode_partial[(list_mode_partial[:,5] / 8) % 4 == 0]

            index_a1_100 = np.where(list_mode_partial[:, 0] < adc_cuts[0])
            index_a2_100 = np.where(list_mode_partial[:, 1] < adc_cuts[1])
            index_a3_100 = np.where(list_mode_partial[:, 2] < adc_cuts[2])
            index_a4_100 = np.where(list_mode_partial[:, 3] < adc_cuts[3])

            # indexes_intersection_A = np.intersect1d(index_a1_100, index_a2_100)
            # indexes_intersection_B = np.intersect1d(index_a3_100, index_a4_100)

            indexes_intersection_A = np.union1d(index_a1_100, index_a2_100)
            indexes_intersection_B = np.union1d(index_a3_100, index_a4_100)

            union_indexes_intersection = np.union1d(indexes_intersection_A,
                                                    indexes_intersection_B)

            # print('{}  Initial Number Counts:  {} '.format(file_folder, len(listMode)))
            list_mode_partial = np.delete(list_mode_partial, union_indexes_intersection, axis=0)

            print('After cutting threshold:  {} '.format(len(list_mode_partial)))

            # np.savetxt(relative_path+'teste.txt', list_mode_partial[:,0:4])
            # np.savetxt('teste.txt', list_mode_partial, delimiter='\t')

            spectrumA = list_mode_partial[:, 0] + list_mode_partial[:, 1]
            spectrumB = list_mode_partial[:, 2] + list_mode_partial[:, 3]
            ratioA = (list_mode_partial[spectrumA != 0, 0] - list_mode_partial[spectrumA != 0, 1]) / spectrumA[
                spectrumA != 0]
            ratioB = (list_mode_partial[spectrumB != 0, 3] - list_mode_partial[spectrumB != 0, 2]) / spectrumB[
                spectrumB != 0]

            index_a1_100 = np.where((list_mode_partial[:, 0] < np.abs(ratioA) * parameter_cut_a + parameter_cut_b))
            index_a2_100 = np.where((list_mode_partial[:, 1] < np.abs(ratioA) * parameter_cut_a + parameter_cut_b))
            index_a3_100 = np.where(
                (list_mode_partial[:, 2] < np.abs(ratioB) * parameter_cut_c + parameter_cut_d))
            index_a4_100 = np.where(
                (list_mode_partial[:, 3] < np.abs(ratioB) * parameter_cut_c + parameter_cut_d))
            indexes_intersection_A = np.intersect1d(index_a1_100, index_a2_100)
            indexes_intersection_B = np.intersect1d(index_a3_100, index_a4_100)

            union_indexes_intersection = np.union1d(indexes_intersection_A,
                                                    indexes_intersection_B)
            # list_mode_partial = np.delete(list_mode_partial, union_indexes_intersection, axis=0)

            # threshold_ratio = 0.6
            # [self.peakMatrix, calibration_file, energyfactor] = calibration_points_init(self.crystal_geometry)
            #
            self.peakMatrix = self.peakMatrix_info.split(" ")
            peakMatrix = np.array(self.peakMatrix)
            peakMatrix = peakMatrix[peakMatrix != ""].astype(float)
            self.peakMatrix = np.reshape(peakMatrix, (2, int(len(peakMatrix) // 2))).T
            energyfactor = self.energyfactor_info
            energyfactor = energyfactor.split(",")
            energyfactor = np.array(energyfactor)
            energyfactor = energyfactor[energyfactor != ""].astype(float)
            energyfactor = np.reshape(energyfactor, (1, len(energyfactor)))
            threshold_ratio = self.peakMatrix.max() + (self.peakMatrix[-1, 0] - self.peakMatrix[-2, 0]) / 2
            # threshold_ratio = 0.7
            union_indexes_intervals_total = np.array([], dtype=np.int64)
            for peak in range(len(self.peakMatrix) - 1):
                min_A = self.peakMatrix[peak, 0] + np.abs(
                    self.peakMatrix[peak, 0] - self.peakMatrix[peak + 1, 0]) * 0.4
                max_A = self.peakMatrix[peak, 0] + np.abs(
                    self.peakMatrix[peak, 0] - self.peakMatrix[peak + 1, 0]) * 0.6
                min_B = self.peakMatrix[peak, 1] + np.abs(
                    self.peakMatrix[peak, 1] - self.peakMatrix[peak + 1, 1]) * 0.4
                max_B = self.peakMatrix[peak, 1] + np.abs(
                    self.peakMatrix[peak, 1] - self.peakMatrix[peak + 1, 1]) * 0.6
                index_remove_intervals_A = np.where((ratioA >= min_A) & (ratioA < max_A))
                index_remove_intervals_B = np.where((ratioB >= min_B) & (ratioB < max_B))
                union_indexes_intervals = np.union1d(index_remove_intervals_A,
                                                     index_remove_intervals_B)
                union_indexes_intervals_total = np.append(union_indexes_intervals, union_indexes_intervals_total)

            union_indexes_intervals_total = np.append(union_indexes_intersection, union_indexes_intervals_total)
            list_mode_partial = np.delete(list_mode_partial, union_indexes_intervals_total, axis=0)

            # listMode=listMode[listMode[:, 0] > np.abs(ratioA)*1000 + 500]
            # spectrumA = listMode[:, 0] + listMode[:, 1]
            # spectrumB = listMode[:, 2] + listMode[:, 3]
            # ratioA = (listMode[spectrumA != 0, 0] - listMode[spectrumA != 0, 1]) / spectrumA[spectrumA != 0]
            # ratioB = (listMode[spectrumB != 0, 3] - listMode[spectrumB != 0, 2]) / spectrumB[spectrumB != 0]
            # # listMode = listMode[np.abs(listMode[:, 0] - listMode[:, 1]) > 200]
            # self.listMode_original = listMode
            list_mode_original_list[i] = list_mode_partial
            number_of_events_after_cuts_original += len(list_mode_partial)
            a1 = list_mode_partial[:, 0]
            a2 = list_mode_partial[:, 1]
            a3 = list_mode_partial[:, 2]
            a4 = list_mode_partial[:, 3]
            theta = list_mode_partial[:, 4]
            phi = list_mode_partial[:, 5]
            timestamp_list = list_mode_partial[:, 6]

            #
            [data, crystalMatrix, EA_corrected, EB_corrected, timestamp_list] = binary_data().data_preparation(a1, a2,
                                                                                                               a3, a4,
                                                                                                               phi,
                                                                                                               theta,
                                                                                                               self.crystal_geometry,
                                                                                                               self.peakMatrix,
                                                                                                               energyfactor,
                                                                                                               timestamp_list,
                                                                                                               threshold=threshold_ratio)
            header = self.header

            step_bot = round(header[1], 3) / header[2]
            step_top = round(header[3], 3) / header[4]
            topRange = header[5]

            list_mode_partial = np.zeros((len(crystalMatrix), 7))
            list_mode_partial[:, 0] = (EA_corrected[0, :]).astype(int)
            list_mode_partial[:, 1] = (EB_corrected[0, :]).astype(int)
            list_mode_partial[:, 2] = crystalMatrix[:, 0]
            list_mode_partial[:, 3] = crystalMatrix[:, 1]
            list_mode_partial[:, 4] = data[:, 5] * step_bot
            list_mode_partial[:, 5] = data[:, 4] * step_top - topRange / 2
            # list_mode_partial[:,4] = data[:, 4] * step_bot
            # list_mode_partial[:,5] = data[:, 5] * step_top - topRange / 2
            list_mode_partial[:, 6] = timestamp_list

            list_mode_list[i] = list_mode_partial
            number_of_events_after_cuts += len(list_mode_partial)

        self.listMode = np.zeros((number_of_events_after_cuts, 7))
        self.listMode_original = np.zeros((number_of_events_after_cuts_original, 7))
        begin_event = 0
        c = 1
        for partial in list_mode_list:
            end_event = begin_event + len(partial)
            self.listMode[begin_event:end_event, :] = partial
            time_indexes[c] = end_event - 1
            begin_event = end_event
            c += 1
        self.acquisitionInfo["Turn end index"] = time_indexes
        # self.listMode = self.listMode[time_indexes[9]:]

        c = 1
        begin_event = 0
        for partial in list_mode_original_list:
            end_event = begin_event + len(partial)
            self.listMode_original[begin_event:end_event, :] = partial
            # time_indexes[c] = end_event - 1
            begin_event = end_event
            # c += 1

    def _organize_matrix_for_coaslence(self):

        pair_vector = self.listMode[:, 2] % 2
        pair_vector_2 = self.listMode[:, 3] % 2
        # ind = np.lexsort((self.listMode[:,5],self.listMode[:,4], self.listMode[:,3], self.listMode[:,2]))
        # ind = np.lexsort((pair_vector_2,pair_vector,self.listMode[:,3], self.listMode[:,2]))
        # ind = np.lexsort((self.listMode[:, 3], self.listMode[:, 2]))
        # self.listMode = self.listMode[ind]
        diff_vector = np.abs(self.listMode[:, 3] - self.listMode[:, 2])
        sum_vector = np.abs(self.listMode[:, 3] + self.listMode[:, 2])
        # ind = np.lexsort((self.listMode[:, 3], diff_vector,sum_vector,self.listMode[:, 2]))
        # ind = np.lexsort((self.listMode[:, 2], diff_vector))
        # ind = np.lexsort((self.listMode[:,2 ], diff_vector))
        ind = np.lexsort((diff_vector,sum_vector,self.listMode[:, 2], self.listMode[:, 5], self.listMode[:, 4]))
        # ind = np.lexsort((diff_vector, self.listMode[:, 5], sum_vector, self.listMode[:, 4]))
        self.listMode = self.listMode[ind]

    def _create_figures_data_selected(self):
        # Creating folder for image storage
        crystal_geometry = self.crystal_geometry
        self.path_data_validation = os.path.join(os.path.dirname(self.study_file), "Data_Validation")
        if not os.path.isdir(self.path_data_validation):
            os.makedirs(self.path_data_validation)

        f_energy_corrected, ((ax_EA, ax_EB)) = plt.subplots(1, 2)
        f_energy_corrected.suptitle('Energy Cut (keV)', fontsize=16)
        # f_individuals
        f_ids, (ax_idA, ax_idB) = plt.subplots(1, 2)
        f_ids.suptitle('Crystal ID', fontsize=16)
        f_motor_c, (ax_top_c, ax_bot_c) = plt.subplots(1, 2)
        f_motor_c.suptitle('Motors detection angles (.easypet)', fontsize=16)
        f_time, ax_time = plt.subplots(1, 1)
        list_figures_crystal_energy_side_A = [None] * crystal_geometry[0] * crystal_geometry[1]
        list_axes_crystal_energy_side_A = [None] * crystal_geometry[0] * crystal_geometry[1]
        list_figures_crystal_energy_side_B = [None] * crystal_geometry[0] * crystal_geometry[1]
        list_axes_crystal_energy_side_B = [None] * crystal_geometry[0] * crystal_geometry[1]

        if self.save_spectrum_file:
            with PdfPages(os.path.join(self.path_data_validation, "Crystals_spectrum.pdf")) as pdf:
                for i in range(crystal_geometry[0] * crystal_geometry[1]):
                    list_figures_crystal_energy_side_A[i], list_axes_crystal_energy_side_A[i] = plt.subplots(1, 1)
                    list_figures_crystal_energy_side_B[i], list_axes_crystal_energy_side_B[i] = plt.subplots(1, 1)
                    events_A = self.listMode[self.listMode[:, 2] == i + 1, 0]
                    events_B = self.listMode[self.listMode[:, 3] == i + 1, 1]
                    u_id_EA, indices_id_EA = np.unique(events_A, return_index=True)
                    u_id_EB, indices_id_EB = np.unique(events_B, return_index=True)

                    list_axes_crystal_energy_side_A[i].hist(events_A, u_id_EA, [0, 1200])
                    list_axes_crystal_energy_side_A[i].set_xlabel("KeV")
                    list_axes_crystal_energy_side_A[i].set_ylabel("Counts")
                    list_axes_crystal_energy_side_A[i].set_title("Crystal side A {}".format(i))
                    pdf.savefig(list_figures_crystal_energy_side_A[i])
                    plt.close(list_figures_crystal_energy_side_A[i])

                    list_axes_crystal_energy_side_B[i].hist(events_B, u_id_EB, [0, 1200])
                    list_axes_crystal_energy_side_B[i].set_xlabel("KeV")
                    list_axes_crystal_energy_side_B[i].set_ylabel("Counts")
                    list_axes_crystal_energy_side_B[i].set_title("Crystal side B {}".format(i))
                    pdf.savefig(list_figures_crystal_energy_side_B[i])
                    plt.close(list_figures_crystal_energy_side_B[i])

        u_EA, indices_EA = np.unique(self.listMode[:, 0], return_index=True)
        u_EB, indices_EB = np.unique(self.listMode[:, 1], return_index=True)
        u_idA, indices_idA = np.unique(self.listMode[:, 2], return_index=True)
        u_idB, indices_idB = np.unique(self.listMode[:, 3], return_index=True)
        u_bot_c, indices_bot = np.unique(self.listMode[:, 4], return_index=True)
        u_c, indices = np.unique(self.listMode[:, 5], return_index=True)
        u_time, indices_time = np.unique(self.listMode[:, 6], return_index=True)

        ax_EA.hist(self.listMode[:, 0], u_EA, [0, 1200])
        ax_EA.set_xlabel("KeV")
        ax_EA.set_ylabel("Counts")
        ax_EB.hist(self.listMode[:, 1], u_EB, [0, 1200])
        ax_idA.hist(self.listMode[:, 2], len(u_idA) + 1, [np.min(self.listMode[:, 2]), np.max(self.listMode[:, 2]) + 1])
        ax_idB.hist(self.listMode[:, 3], len(u_idB) + 1, [np.min(self.listMode[:, 3]), np.max(self.listMode[:, 3]) + 1])
        ax_bot_c.hist(self.listMode[:, 4], u_bot_c)
        ax_top_c.hist(self.listMode[:, 5], u_c)
        ax_time.hist(self.listMode[:, 6], u_time)

        # Pictures .easypetoriginal
        f_histogram2d, (ax_histogram2d_A, ax_histogram2d_B) = plt.subplots(1, 2)
        f_spectrum, ((ax_spectrumA, ax_spectrumB), (ax_ratioA, ax_ratioB)) = plt.subplots(2, 2)
        f_motor, ((ax_top, ax_bot)) = plt.subplots(1, 2)

        spectrumA = self.listMode_original[:, 0] + self.listMode_original[:, 1]
        spectrumB = self.listMode_original[:, 2] + self.listMode_original[:, 3]
        ratioA = (self.listMode_original[spectrumA != 0, 0] - self.listMode_original[spectrumA != 0, 1]) / spectrumA[
            spectrumA != 0]
        ratioB = (self.listMode_original[spectrumB != 0, 3] - self.listMode_original[spectrumB != 0, 2]) / spectrumB[
            spectrumB != 0]

        nBins = 750
        threshold = 0.99
        [n, bins, patches] = ax_ratioA.hist(ratioA, nBins, [-threshold, threshold])
        peak_amplitude = np.digitize(self.peakMatrix[:, 0], bins, right=True)
        ax_ratioA.plot(self.peakMatrix[:, 0], n[peak_amplitude - 1], '.', markersize=1)
        [n, bins, patches] = ax_ratioB.hist(ratioB, nBins, [-threshold, threshold])
        peak_amplitude = np.digitize(self.peakMatrix[:, 1], bins, right=True)
        ax_ratioB.plot(self.peakMatrix[:, 1], n[peak_amplitude - 1], '.', markersize=1)
        [n, bins, patches] = ax_spectrumA.hist(spectrumA, nBins, [0, 8096])
        [n, bins, patches] = ax_spectrumB.hist(spectrumB, nBins, [0, 8096])

        # HistogrM 2D
        # Z_A, X, Y = np.histogram2d(ratioA, spectrumA, 750, range=[[-0.95, 0.95], [200, 6000]])
        spectrumA = spectrumA[spectrumA != 0]
        spectrumB = spectrumB[spectrumB != 0]
        # Z_A, X, Y = np.histogram2d(ratioA, spectrumA, 750, range=[[np.nanmin(ratioA), np.nanmax(ratioB)], [np.nanmin(spectrumA), np.nanmax(spectrumA)]])
        Z_A, X, Y = np.histogram2d(ratioA, spectrumA, [750, 375], range=[[-threshold, threshold], [200, 6000]])
        Z_B, X, Y = np.histogram2d(ratioB, spectrumB, [750, 375], range=[[-threshold, threshold], [200, 6000]])

        ax_histogram2d_A.matshow(Z_A, cmap="jet")
        ax_histogram2d_B.matshow(Z_B, cmap="jet")
        # kernel_filter_size = 11
        # Z_A = median_filter(Z_A, kernel_filter_size)
        # fp = findpeaks(method='topology')
        # # Example 2d image
        # # X = fp.import_example('2dpeaks')
        # # Fit topology method on the 1d-vector
        # results = fp.fit(Z_A)
        # # The output contains multiple variables
        # print(results["Xraw"])
        # fp.plot()
        # dict_keys(['Xraw', 'Xproc', 'Xdetect', 'Xranked', 'persistence', 'peak', 'valley', 'groups0'])
        ### MOTORS
        u, indices = np.unique(self.listMode_original[:, 5], return_index=True)
        u_bot, indices_bot = np.unique(self.listMode_original[:, 4], return_index=True)

        ax_top.hist(self.listMode_original[:, 4], u_bot)
        ax_top.set_title("TOP")
        ax_bot.hist(self.listMode_original[:, 5], u)
        ax_bot.set_title("BOT")
        plt.show()
        plt.close(f_energy_corrected)
        plt.close(f_ids)
        plt.close(f_time)
        plt.close(f_motor_c)
        plt.close(f_spectrum)
        plt.close(f_motor)
        plt.close((f_histogram2d))
        with PdfPages(os.path.join(self.path_data_validation, "DataValidation.pdf")) as pdf:
            # Pictures .easypet
            # Energy Histograms

            # f_motor_c.savefig(os.path.join(self.path_data_validation, "Motors easypet.png"))
            pdf.savefig(f_energy_corrected)
            pdf.savefig(f_ids)
            pdf.savefig(f_time)
            pdf.savefig(f_motor_c)
            pdf.savefig(f_spectrum)
            pdf.savefig(f_motor)
            pdf.savefig(f_histogram2d)

            # for i in range(crystal_geometry[0] * crystal_geometry[1]):
            #     pdf.savefig(list_figures_crystal_energy_side_A)
            #     plt.close(list_figures_crystal_energy_side_A)
            #     pdf.savefig(list_figures_crystal_energy_side_B)
            #     plt.close(list_figures_crystal_energy_side_B)

    def _creat_test_geometry(self):
        self.listMode = np.zeros((32, 7))
        self.listMode[:, 0] = 511
        self.listMode[:, 1] = 511
        self.listMode[:, 2] = np.arange(1, 33)
        self.listMode[:, 3] = -np.arange(1, 33) + 33
        self.listMode[:, 4] = 0
        self.listMode[:, 5] = 0.01 * np.random.rand(32)
        self.listMode[:, 6] = 1
        # self.listMode = self.listMode[0:10,:]
        self.listMode = self.listMode[0::2, :]
        self.listMode = self.listMode[7:9, :]
