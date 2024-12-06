import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from src.EasyPETLinkInitializer.EasyPETDataReader import binary_data
import tkinter as tk
from tkinter import filedialog
import scipy
from scipy.optimize import curve_fit
from scipy import ndimage, misc
from scipy.stats import binom, t
from skimage import filters
from src.Geometry import SetParametricCoordinates, MatrixGeometryCorrection
from src.EasyPETLinkInitializer.Preprocessing import Sinogram
from src.Corrections.DecayCorrection import DecayCorrection
from src.EasyPETLinkInitializer.SimulationData import SimulationStatusData


class NECAnalysis:
    def __init__(self, file_path=None, generate_files=True, phantom_type="mouse"):

        self.f_gaussian_mean = plt.figure()
        self.file_path = file_path
        self.listMode_per_file = []
        self.folders = None
        self.id = 0
        self.counts_per_acquisition = None
        self.bq_per_acquisition = None
        path_folder = os.path.dirname(os.path.dirname(self.file_path))
        self.path_data_validation = os.path.join(path_folder, "NEC Analysis")
        self.sinogram = None
        self.phantom_type = phantom_type
        self.physical_center = 79
        self.motor_center = 62
        self.half_axial_fov = 72.68 / 2
        self.half_motor_axial_fov = 114 / 2
        self.bq_background = None
        self.crystals_geometry = None
        self.acquisition_time = None
        self.center_method = "Maximum"
        self.cr_t = None
        self.cr_interpolated_t = None
        self.Rtot_i_j = None
        self.Rtrue_i_j = None
        self.sf_i_j = None
        self.Rrandom_i_j = None
        self.SeventRate_i_j = None
        self.RNECi_j = None
        self.Average_activity = None
        self.r_int = 0
        if generate_files:
            self._createDirectory()
            # self._getIntrinsicDetectorRadiation()
            self._readFiles(self.file_path)
            self._generateData(_background_calculation=False)

        else:
            self._loadYData()
        print(self.folders)

    def _loadYData(self):
        self.cr_t = np.load(os.path.join(self.path_data_validation, "cr_t.npy"))
        self.cr_interpolated_t = np.load(os.path.join(self.path_data_validation, "cr_interpolated_t.npy"))
        self.Rtot_i_j = np.loadtxt(os.path.join(self.path_data_validation, "Rtot_i_j"))
        self.Rtrue_i_j = np.loadtxt(os.path.join(self.path_data_validation, "Rtrue_i_j"))
        self.Rrandom_i_j = np.loadtxt(os.path.join(self.path_data_validation, "Rrandom_i_j"))
        self.sf_i_j = np.loadtxt(os.path.join(self.path_data_validation, "sf_i_j"))
        self.SeventRate_i_j = np.loadtxt(os.path.join(self.path_data_validation, "SeventRate_i_j"))
        self.RNECi_j = np.loadtxt(os.path.join(self.path_data_validation, "Rtrue_i_j"))
        self.Average_activity = np.loadtxt(os.path.join(self.path_data_validation, "Average_activity"))
        # self.r_int = np.load(os.path.join(self.path_data_validation, "r_int.npy"))
        self.C_tot_ij_t = np.loadtxt(os.path.join(self.path_data_validation, "C_tot_ij_t"))
        self.c_rs_ij_t = np.loadtxt(os.path.join(self.path_data_validation, "c_rs_ij_t"))


    def _createDirectory(self):
        path_folder = os.path.dirname(os.path.dirname(self.file_path))
        self.path_data_validation = os.path.join(path_folder, "NEC Analysis")
        if not os.path.isdir(self.path_data_validation):
            os.makedirs(self.path_data_validation)

    def _readFiles(self, file_name):
        try:
            [self.listMode, self.Version_binary, self.header, self.dates, self.otherinfo, self.acquisitionInfo,
             self.stringdata, self.systemConfigurations_info, self.energyfactor_info,
             self.peakMatrix_info] = binary_data().open(file_name)
            self.crystals_geometry = [self.systemConfigurations_info['array_crystal_x'],
                                      self.systemConfigurations_info['array_crystal_y']]

        except FileNotFoundError:
            raise FileNotFoundError


        # self.applyCoincidenceWindow
        print(len(self.listMode))

    def _applyDualEnergyWindow(self, lower=350, higher=750):
        """

        """
        time_indexes = np.array(self.acquisitionInfo["Turn end index"])
        list_mode_list = [None] * (len(time_indexes) - 1)
        # list_mode_list = [None] * len(range(0,len(time_indexes)-1,2))
        # list_mode_list = [None]*(int(len(time_indexes)/2))
        number_of_events_after_cuts = 0
        # for i in range(0, len(time_indexes) - 1, 2):

        for i in range(0, len(time_indexes) - 1):

            list_mode_partial = self.listMode[time_indexes[i]:time_indexes[i + 1], :]

            index_ea = np.where(
                (list_mode_partial[:, 0] < lower) | (list_mode_partial[:, 1] < lower))
            index_eb = np.where(
                (list_mode_partial[:, 0] > higher) | (list_mode_partial[:, 1] > higher))
            union_indexes_intersection = np.union1d(index_ea, index_eb)
            list_mode_partial = np.delete(list_mode_partial, union_indexes_intersection, axis=0)


            list_mode_list[i] = list_mode_partial
            number_of_events_after_cuts += len(list_mode_partial)
        self.listMode = np.zeros((number_of_events_after_cuts, 7))
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


    def _breakListModeIntoParts(self, numberOfEvents=500000):
        time_indexes = np.array(self.acquisitionInfo["Turn end index"])
        valid_turns = [np.int32(0)]

        diff_time_indexes = np.diff(time_indexes)
        diff_time_indexes = np.append(0, diff_time_indexes)
        t_s = diff_time_indexes[0]
        for t in range(len(time_indexes)) :
            value = t_s // numberOfEvents
            if value >= 1:
                valid_turns.append(time_indexes[t-2])
                t_s = diff_time_indexes[t]
            else:
                t_s += diff_time_indexes[t]

        # division = time_indexes // numberOfEvents
        # unique = np.unique(division, return_counts=True)
        # valid_turns = time_indexes[np.cumsum(unique[1][:-1])]
        # print(valid_turns)
        return np.array(valid_turns)

    def _generateData(self, _background_calculation=False):
        a = np.copy(self.listMode[:, 2])
        b = np.copy(self.listMode[:, 3])
        ea = np.copy(self.listMode[:, 0])
        eb = np.copy(self.listMode[:, 1])
        self.listMode[:, 2] = b
        self.listMode[:, 3] = a
        self.listMode[:, 0] = eb
        self.listMode[:, 1] = ea
        # self._applyDualEnergyWindow(lower=100, higher=900)
        listmode = self.listMode

        system_configurations_dir = os.path.dirname(os.path.dirname(os.path.dirname(
                                                            os.path.dirname(os.path.abspath(__file__)))))
        MatrixCorrection = MatrixGeometryCorrection(operation='r',
                                                    file_path=os.path.join(system_configurations_dir,
                                                        'system_configurations', 'x_{}__y_{}'.format(
                                                            self.crystals_geometry[0], self.crystals_geometry[1])))
        geometry_file = MatrixCorrection.coordinates
        crystals_geometry = self.crystals_geometry
        height = self.systemConfigurations_info["crystal_pitch_x"]
        crystal_width = self.systemConfigurations_info["crystal_pitch_y"]
        reflector_y = 2 * self.systemConfigurations_info["reflector_interior_A_y"]

        geometry_file[:, 1] = np.tile(np.round(np.arange(0, crystals_geometry[1] * crystal_width + 2 * reflector_y,
                                                         crystal_width + reflector_y) - (
                                                       crystal_width + reflector_y) *
                                               (crystals_geometry[1] - 1) / 2, 3), crystals_geometry[0] * 2)
        if not self.acquisitionInfo["Type of subject"] == "Simulation":
            geometry_file[:crystals_geometry[0] * crystals_geometry[1], 1] *= -1

        if _background_calculation:
            for_dim = 1
        else:
            # if not self.acquisitionInfo["Type of subject"] == "Simulation":
            #     valid_turns = self._breakListModeIntoParts()
            # else:
            valid_turns = self.acquisitionInfo["Turn end index"]#hack for simulation
            valid_turns = [0, 5831895, 8133158, 9129935, 10023839, 10543612, 11023706, 11250857]#hack for simulation
            # saved_time = (listmode_temp[-1-1, 6] - listmode_temp[0, 6])
            saved_time = [481.0047301997547, 481.0047301997547, 481.0037258126249, 962.0039543666644, 962.000346938381, 1923.9923457532423, 5771.985315153957]
            valid_turns =  [0, 23679418, 43625879, 60153401, 73557587, 84130166, 92186643, 98018538, 100319801, 101316578, 102210482, 102730255, 103210349, 103602216]#hack for simulation
            saved_time = [481.00497392637703,481.00497392637703, 481.00494717908197, 481.004956082609, 481.004850723411, 481.00487213812,
             481.004913950128, 481.004730199745, 481.00372581264696, 962.003954366732, 962.0003469383939,
             1923.9923457532382, 5771.985315153976]

            for_dim = len(valid_turns) - 1

        # bins_x = len(np.unique(self.listMode[:,4]))
        bins_x = 25
        bins_y = 50
        axial_cuts = self.crystals_geometry[0] * 2
        sf_i_j = np.zeros(
            (for_dim, axial_cuts - 1))
        c_rs_ij_t = np.zeros(
            (for_dim, axial_cuts - 1))
        C_tot_ij_t = np.zeros(
            (for_dim, axial_cuts - 1))  # Scatter fraction per slice per aquisition
        Rtot_i_j = np.zeros((for_dim, axial_cuts - 1))  # Total event Rate
        Rtrue_i_j = np.zeros((for_dim, axial_cuts - 1))  # Total event Rate
        # Decay Correction
        decay_correction_class = DecayCorrection(listMode=listmode, acquisition_info=self.acquisitionInfo,
                                                 correct_decay=True)
        # self.decay_correction_class.list_mode_decay_correction()
        initial_activity = decay_correction_class.activity_on_subject_at_scanning_time()

        Average_activity = np.zeros(for_dim)
        # s_forinterpolation = np.arange(self.sinogram[2].min(), self.sinogram[2].max(), 0.5)
        s_forinterpolation = np.arange(-30, 30, 0.5)
        cr_t = np.zeros((for_dim, axial_cuts - 1, bins_y + 1))
        cr_interpolated_t = np.zeros((for_dim, axial_cuts - 1, len(s_forinterpolation)))
        for j in range(0, for_dim):
            path = os.path.join(self.path_data_validation,str(j))

            if not os.path.exists(path):
                os.makedirs(path)

            self.current_path = path
            # for j in range(2):
            if _background_calculation:
                listmode_temp = listmode
            else:
                listmode_temp = listmode[valid_turns[j]:valid_turns[j + 1]]
                print("Número de eventos: {}".format(len(listmode_temp)))
                coinc_window = SimulationStatusData(listmode_temp, study_file=self.path_data_validation)
                # listmode_temp = coinc_window.applyCoincidenceWindow(listmode_temp, 300)
                print("Número de eventos 300 ps: {}".format(len(listmode_temp)))
                A0_frame = decay_correction_class.apply_decay(initial_activity, listmode[valid_turns[j], 6],
                                                              decay_correction_class.decay_half_life)
                print("Initial Activity: {}".format(initial_activity))
                print("A0_frame: {}".format(A0_frame/37000))
                print("time: {}".format(listmode[valid_turns[j], 6]))
                # Average_activity[j] = decay_correction_class.average_activity(A0_frame,
                #                                                               listmode[valid_turns[j + 1], 6] -
                #                                                               listmode[valid_turns[j], 6],
                #                                                               decay_correction_class.decay_half_life)

                Average_activity[j] = decay_correction_class.average_activity(A0_frame,
                                                                              saved_time[j],
                                                                              decay_correction_class.decay_half_life)
            print("Time Turn: {}".format(saved_time[j]))
            parametric = SetParametricCoordinates(listMode=listmode_temp, geometry_file=geometry_file,
                                                  crystal_width=self.systemConfigurations_info[
                                                      "crystal_pitch_x"],
                                                  crystal_height=self.systemConfigurations_info[
                                                      "crystal_pitch_y"],
                                                  distance_between_motors=self.systemConfigurations_info[
                                                      "distance_between_motors"],
                                                  distance_crystals=self.systemConfigurations_info[
                                                      "distance_between_crystals"],
                                                  crystal_depth=self.systemConfigurations_info['crystal_length'],
                                                  simulation_files=True, transform_into_positive=False)

            sinoClass = Sinogram(listMode=listmode_temp, parametric=parametric)
            sinoClass.calculate_s_phi()
            sinoClass.updateLimits()
            sinoClass.calculateMichelogram(bins_x=bins_x, bins_y=bins_y, f2f_or_reb="reb")

            s = sinoClass.s
            phi = sinoClass.phi

            self.sinogram = sinoClass.michelogram
            # number_s_steps = len(np.unique(s))
            number_s_steps = len(self.sinogram[2])
            s_step_size = (s.max() + np.abs(s.min())) / number_s_steps
            # number_of_s_bins = int(np.ceil(20.36 / s_step_size))
            number_of_s_bins = int(np.round(s.max() / s_step_size,0))

            angles_axial = np.zeros(self.sinogram[0].shape[2])

            print("________")

            for i in range(self.sinogram[0].shape[2]):
                self._center_sinogram(number_s_steps, number_of_s_bins, slice_=i, time_point=j,
                                                        _background_calculation=_background_calculation)
                c_r = np.sum(self.centered_sinogram, axis=0)
                cr_t[j, i] = c_r
                s_or = self.sinogram[2]

                c_r_interpolated = np.interp(s_forinterpolation, self.sinogram[2], c_r)

                cr_interpolated_t[j, i] = c_r_interpolated
                pixel_r_finder = np.abs(s_forinterpolation - 7)
                pixel_l_finder = np.abs(s_forinterpolation + 7)
                pixel_r = c_r_interpolated[pixel_r_finder == pixel_r_finder.min()][0]  # sum along S
                pixel_l = c_r_interpolated[pixel_l_finder == pixel_l_finder.min()][0]
                # print(s_forinterpolation[pixel_r_finder == pixel_r_finder.min()][0])# sum along S
                # print(s_forinterpolation[pixel_l_finder == pixel_l_finder.min()][0])# sum along S
                if j==0:
                    plt.plot(s_or, c_r)
                    plt.plot(s_forinterpolation, c_r_interpolated)
                    plt.plot(s_forinterpolation[pixel_r_finder == pixel_r_finder.min()][0], pixel_r, '.',color='red')
                    plt.plot(s_forinterpolation[pixel_l_finder == pixel_l_finder.min()][0], pixel_l, '.', color='red')
                    # plt.show()
                _pixel_average = (pixel_l + pixel_r) / 2
                indexes = (s_or <= 7) & (s_or > -7)
                wide_strip_pixels = c_r[indexes]
                outside_strip_pixels = c_r[~indexes]
                c_rs_ij = _pixel_average * len(wide_strip_pixels) + np.sum(outside_strip_pixels)
                C_tot_ij = np.sum(c_r)

                Rtot_i_j[j, i] = C_tot_ij / saved_time[j]

                Rtrue_i_j[j, i] = (C_tot_ij - c_rs_ij) / saved_time[j]
                c_rs_ij_t[j, i] = c_rs_ij
                C_tot_ij_t[j, i] = C_tot_ij

                # sf_i_j[j, i] = c_rs_ij / C_tot_ij
        if _background_calculation:
            np.save(os.path.join(self.path_data_validation, "r_int"), np.array(Rtot_i_j))
            self.r_int = Rtot_i_j
        else:
            np.save(os.path.join(self.path_data_validation, "cr_t"), np.array(cr_t))
            np.save(os.path.join(self.path_data_validation, "cr_interpolated_t"), np.array(cr_interpolated_t))
            np.savetxt(os.path.join(self.path_data_validation, "Rtot_i_j"), np.array(Rtot_i_j))
            np.savetxt(os.path.join(self.path_data_validation, "Rtrue_i_j"), np.array(Rtrue_i_j))
            np.savetxt(os.path.join(self.path_data_validation, "C_tot_ij_t"), np.array(C_tot_ij_t))
            np.savetxt(os.path.join(self.path_data_validation, "c_rs_ij_t"), np.array(c_rs_ij_t))
            # np.savetxt(os.path.join(self.path_data_validation, "sf_i_j"), np.array(sf_i_j))
            act_cond = Average_activity / 37000 > 3
            sf_i_j = np.sum(c_rs_ij_t[act_cond], axis=0) / np.sum(C_tot_ij_t[act_cond], axis=0)

            Rrandom_i_j = Rtot_i_j - (Rtrue_i_j / (1 - np.tile(np.sum(sf_i_j, axis=0), (Rtrue_i_j.shape[0], 1))))
            SeventRate_i_j = Rtot_i_j - Rtrue_i_j - Rrandom_i_j - self.r_int
            RNECi_j = Rtrue_i_j ** 2 / Rtot_i_j
            np.savetxt(os.path.join(self.path_data_validation, "sf_i_j"), np.array(sf_i_j))
            np.savetxt(os.path.join(self.path_data_validation, "Rrandom_i_j"), np.array(Rrandom_i_j))
            np.savetxt(os.path.join(self.path_data_validation, "SeventRate_i_j"), np.array(SeventRate_i_j))
            np.savetxt(os.path.join(self.path_data_validation, "RNECi_j"), np.array(RNECi_j))
            np.savetxt(os.path.join(self.path_data_validation, "Average_activity"), np.array(Average_activity))
            self.cr_t = cr_t
            self.cr_interpolated_t = cr_interpolated_t
            self.Rtot_i_j = Rtot_i_j
            self.Rtrue_i_j = Rtrue_i_j
            self.sf_i_j = sf_i_j
            self.Rrandom_i_j = Rrandom_i_j
            self.SeventRate_i_j = SeventRate_i_j
            self.RNECi_j = RNECi_j
            self.Average_activity = Average_activity
            self.c_rs_ij_t = c_rs_ij_t
            self.C_tot_ij_t = C_tot_ij_t

    def _getIntrinsicDetectorRadiation(self):
        name_subfolder = "Background_RInt"
        name_background_file = "Easypet Scan 31 Oct 2022 - 16h 29m 11s"
        path_file = os.path.join(os.path.dirname(os.path.dirname(self.file_path)),
                                 name_subfolder, name_background_file,
                                 "{}.easypet".format(name_background_file))
        self._readFiles(path_file)
        self._generateData(_background_calculation=True)

    def _center_sinogram(self, number_of_s_bins, bin_range, slice_=0, time_point=0, _background_calculation=False):
        self.centered_sinogram = np.zeros((len(self.sinogram[1]), number_of_s_bins))
        # bin_range = int(number_of_s_bins/2)
        result = self.sinogram[0][:, :, slice_]
        gaussian_r = np.zeros((result.shape[0], 4))
        failed_maxim = 0
        previous_loc_max = 0
        for i in range(result.shape[0]):
            # max_value = sinogram[i, :].max()
            # loc_max = np.where(sinogram[i, :] == max_value)[0][0]
            if self.center_method == "Maximum":
                popt = np.ones((4))
                s_y = self.sinogram[2][:-1]
                mean_posix = s_y[result[i, :] == result[i, :].max()][0]
                popt[1] = mean_posix
            elif self.center_method == "Gaussian":
                try:
                    max_value = result[i, :].max()
                    s_y = self.sinogram[2][:-1]
                    mean_posix = s_y[result[i, :] == result[i, :].max()]

                    p0 = np.array([max_value, mean_posix[0], 1 / max_value * np.sqrt(2 * np.pi), int(0)])
                    popt, pcov = curve_fit(NECAnalysis.gaussian_fit, self.sinogram[2][:-1], result[i, :],
                                           method='lm', p0=p0)
                    # print(max_value)
                    # print((1 / popt[2] * np.sqrt(2 * np.pi)))
                    # print("_______")

                except RuntimeError as e:
                    print(e)
                    popt = np.ones((4))
                    s_y = self.sinogram[2][:-1]
                    mean_posix = s_y[result[i, :] == result[i, :].max()][0]
                    popt[1] = mean_posix
                gaussian_r[i] = np.array(popt)

            diff_vector = np.abs(popt[1] - self.sinogram[2][:-1])
            loc_max = int(np.where(diff_vector == diff_vector.min())[0])

            # gaussian_r = np.array(gaussian_r)

            try:
                diff = 0
                value_min = int(loc_max - bin_range / 2)
                # value_max = loc_max+bin_range + np.abs(bin_range*2-number_of_s_bins)
                value_max = int(loc_max + bin_range / 2)
                centered_min = int(number_of_s_bins / 2 - bin_range / 2)
                centered_max = int(number_of_s_bins / 2 + bin_range / 2)

                if value_min < 0:
                    diff = np.abs(value_min)
                    value_min = 0
                    centered_min += diff

                if value_max > number_of_s_bins - 2:
                    diff = np.abs(value_max - number_of_s_bins - 2)
                    value_max = number_of_s_bins - 2
                    centered_max -= diff

                diff = (value_max - value_min) - (centered_max - centered_min)
                if diff != 0:
                    centered_max += diff

                self.centered_sinogram[i, centered_min:centered_max] = result[i, value_min:value_max]
                # print("MAXIMUM: {}".format(loc_max))
                # print("Min range: {}".format(loc_max - bin_range / 2))
                # print("Max range: {}".format(loc_max + bin_range / 2))
                # print("________")

            except ValueError as e:
                print(e)
                print("Miss maximum")
                print(loc_max)
                # print(loc_max-bin_range/2)
                # print(result[i, centered_min:centered_max])

                failed_maxim += 1
                continue
            previous_loc_max = loc_max
        if _background_calculation:
            name_sinogram = "Sinogram_back"
            name_center_sinogram = "Centered_sinogram_back"
        else:
            name_sinogram = "Sinogram"
            name_center_sinogram = "Centered_sinogram_"

        left = self.sinogram[1][0]
        right = self.sinogram[1][-1]
        bottom = self.sinogram[2][0]
        top = self.sinogram[2][-1]
        bottom = -20
        top = np.abs(bottom)
        result = result[:, np.abs(self.sinogram[2][:-1]) <= top]
        self.f_gaussian_mean = plt.figure()
        plt.plot(self.sinogram[1][:-1], gaussian_r[:, 1], "-.", color="black")
        plt.imshow(result.T, 'hot', interpolation=None, origin="lower", aspect="auto",
                   extent=[left, right, bottom, top])
        plt.xlabel("$Phi(^\circ)$")
        plt.ylabel("$S(mm)$")
        plt.colorbar()
        self.f_gaussian_mean.savefig(os.path.join(self.path_data_validation,str(time_point), "{}{}".format(name_sinogram, slice_)))
        plt.close(self.f_gaussian_mean)
        angle = np.degrees(np.arcsin((gaussian_r[:, 1].max() + gaussian_r[:, 1].min()) / 30))

        f = plt.figure()


        self.centered_sinogram[:, np.abs(self.sinogram[2]) > 22.5] = 0
        # self.centered_sinogram = self.centered_sinogram[:, np.abs(self.sinogram[2]) <= top]
        plt.imshow(self.centered_sinogram.T, 'hot', interpolation=None, origin="lower", aspect="auto",
                   extent=[left, right, bottom, top])
        plt.colorbar()
        plt.plot(self.sinogram[1][:-1], np.zeros(len(self.sinogram[1][:-1])), "-.", color="black")
        plt.xlabel("$Phi(^\circ)$")
        plt.ylabel("$S(mm)$")
        f.savefig(os.path.join(self.path_data_validation, str(time_point), "{}{}".format(name_center_sinogram, slice_)))
        plt.close(f)
        #

    @staticmethod
    def gaussian_fit(x, a, b, c, d):
        return (1 / c * np.sqrt(2 * np.pi)) * np.exp(-((x - b) ** 2) / (2 * c ** 2))

    def finalResults(self, random_activity_cut=None):
        if random_activity_cut is None:
            random_activity_cut = [False, 5]
        print(self.Average_activity/ 37000)
        if random_activity_cut[0]:
            act = np.array([5])
            # for v in range(len(act)):
            #     act_cond = (self.Average_activity < act[v]*37000)
            #     self.sf_i_j = np.sum(self.c_rs_ij_t[act_cond], axis=0) / np.sum(self.C_tot_ij_t[act_cond], axis=0)
            #     print(f"act {act[v]}:{np.sum(self.sf_i_j)}")

            # act_cond = self.Average_activity / 37000 < random_activity_cut[1]
            act_cond =  (self.Average_activity < random_activity_cut[1]*37000)
            self.sf_i_j = np.sum(self.c_rs_ij_t[act_cond], axis=0) / np.sum(self.C_tot_ij_t[act_cond], axis=0)
            print(f"ACT {random_activity_cut[1]}: Scatter fraction{np.sum(np.sum(self.c_rs_ij_t[act_cond], axis=0), axis=0) / np.sum(np.sum(self.C_tot_ij_t[act_cond], axis=0), axis=0)}")
            self.Rrandom_i_j = self.Rtot_i_j - (
                    self.Rtrue_i_j / (1 - np.tile(self.sf_i_j, (self.Rtrue_i_j.shape[0], 1))))
            self.SeventRate_i_j = self.Rtot_i_j - self.Rtrue_i_j - self.Rrandom_i_j #- self.r_int
            self.RNECi_j = self.Rtrue_i_j ** 2 / self.Rtot_i_j

        fig_proj= plt.figure()
        i_ = int(self.cr_interpolated_t.shape[2] / 2)
        i_ = int(10)
        j_ = 1
        s_forinterpolation = np.linspace(-30, 30, len(self.cr_interpolated_t[j_, i_]))
        s_cr = np.linspace(-21,21, len(self.cr_t[j_, i_]))
        pixel_r_finder = np.abs(s_forinterpolation - 7)
        pixel_l_finder = np.abs(s_forinterpolation + 7)
        pixel_r = self.cr_interpolated_t[j_, i_][pixel_r_finder == pixel_r_finder.min()][0]  # sum along S
        pixel_l = self.cr_interpolated_t[j_, i_][pixel_l_finder == pixel_l_finder.min()][0]  # sum along S
        plt.plot(s_forinterpolation, self.cr_interpolated_t[j_, i_], ".-", label="Data Interpolated")
        plt.plot(s_cr, self.cr_t[j_, i_], "--", label="Real Data")
        plt.plot(7, pixel_r, ".", label="$C_{L,ij}$", color="r")
        plt.plot(-7, pixel_l, ".", label="$C_{R,ij}$", color="b")
        n_points = 10
        plt.plot(-7 * np.ones(n_points), np.linspace(0, 1.2 * self.cr_t[j_, i_].max(), n_points), "-", color="b")
        plt.plot(7 * np.ones(n_points), np.linspace(0, 1.2 * self.cr_t[j_, i_].max(), n_points), "-", color="r")
        plt.xlabel("$Distance\:to\:center\:S\:(mm)$")
        plt.ylabel("$Counts$")
        plt.xlim(-20, 20)
        plt.ylim(0, 1.2 * self.cr_t[j_, i_].max())

        plt.grid(True)
        plt.legend()

        fig_proj.savefig(os.path.join(self.path_data_validation, "fig_proj.png"), dpi=300, pad_inches=.1,
                    bbox_inches='tight')

        Average_activity_uci = self.Average_activity[::-1] / (1*10**6)
        fig_nec = plt.figure()
        markers = ["-s", "-p", "-P", "-*", "-o"]
        plt.plot(Average_activity_uci, np.sum(self.Rtot_i_j, axis=1)[::-1], "--s", label="$R_{tot_{ij}} (Total)$", markersize=5)
        plt.errorbar(Average_activity_uci, np.sum(self.Rtot_i_j, axis=1)[::-1], np.sqrt(np.sum(self.Rtot_i_j, axis=1)[::-1]),
                         fmt='none', capsize=5, markersize =5)
        plt.plot(Average_activity_uci, np.sum(self.Rtrue_i_j, axis=1)[::-1], "-o", label="$R_{t_{ij}}(Trues)$")
        plt.errorbar(Average_activity_uci, np.sum(self.Rtrue_i_j, axis=1)[::-1],
                     np.sqrt(np.sum(self.Rtrue_i_j, axis=1)[::-1]),
                     fmt='none', capsize=5, markersize=5)
        plt.plot(Average_activity_uci, np.sum(self.Rrandom_i_j, axis=1)[::-1], ":*", label="$R_{r_{ij}}(Randoms)$")
        plt.errorbar(Average_activity_uci, np.sum(self.Rrandom_i_j, axis=1)[::-1],
                     np.sqrt(np.sum(self.Rrandom_i_j, axis=1)[::-1]),
                     fmt='none', capsize=5, markersize=5)
        plt.plot(Average_activity_uci, np.sum(self.SeventRate_i_j, axis=1)[::-1], "--P",
                 label="$R_{s_{ij}} (Scattered)$")
        plt.errorbar(Average_activity_uci, np.sum(self.SeventRate_i_j, axis=1)[::-1],
                     np.sqrt(np.sum(self.SeventRate_i_j, axis=1)[::-1]),
                     fmt='none', capsize=5, markersize=5)
        plt.plot(Average_activity_uci, np.sum(self.RNECi_j, axis=1)[::-1], "o-", label="$R_{NEC_{ij}} (Noise)$")
        plt.errorbar(Average_activity_uci, np.sum(self.RNECi_j, axis=1)[::-1],
                     np.sqrt(np.sum(self.RNECi_j, axis=1)[::-1]),
                     fmt='none', capsize=5, markersize=5)
        # a = np.arange(Average_activity_uci.min(), Average_activity_uci.max(), 1)
        # a_line = np.interp(a, Average_activity_uci, np.sum(self.Rtrue_i_j, axis=1)[::-1])
        # plt.plot(a, a_line, ".")
        plt.xlabel("$Average \, activity \, (MBq)$")
        plt.ylabel("$cps$")
        plt.legend()
        plt.grid(which='both')
        plt.ylim(0, 1.05 * np.sum(self.Rtot_i_j, axis=1)[::-1].max())
        # log scale

        # Or if you want different settings for the grids:
        plt.grid(which='minor', alpha=0.2, linestyle='--')
        plt.grid(which='major', alpha=0.5, linestyle=':')
        plt.grid(True)
        plt.minorticks_on()
        plt.title("$CTW: \, 40  \, ns$", fontsize=16)
        plt.savefig(os.path.join(self.path_data_validation, "Fig_nec.png"), dpi=300, pad_inches=.1,
                        bbox_inches='tight')


        max_rt = np.sum(self.Rtrue_i_j, axis=1)[::-1].max()
        at_max = Average_activity_uci[np.sum(self.Rtrue_i_j, axis=1)[::-1] == max_rt]
        max_rnec = np.sum(self.RNECi_j, axis=1)[::-1].max()
        anec_max = Average_activity_uci[np.sum(self.RNECi_j, axis=1)[::-1] == max_rnec]

        print("R_t at peak: {} Bq".format(max_rt))
        print("A_t at R_t peak: {} uci".format(at_max))
        print("R_nec peak: {} Bq".format(max_rnec))
        print("A_nec peak: {} uCi".format(anec_max))

        print("Average activity: {}".format(Average_activity_uci[7]))
        print("rnec: {}".format(np.sum(self.RNECi_j, axis=1)[::-1][7]))
        print("rt: {}".format(np.sum(self.Rtrue_i_j, axis=1)[::-1][7]))
        plt.show()

#
# def plt_configure():
#     # fsize = 14
#     # tsize = 14
#     #
#     # tdir = 'in'
#     #
#     # major = 3.0
#     # minor = 1.0
#     #
#     # style = 'seaborn-dark-palette'
#     # plt.style.use(style)
#     # plt.rcParams['text.usetex'] = True
#     # # plt.rcParams['text.font.size'] = 10
#     # plt.rcParams['font.size'] = fsize
#     # plt.rcParams['legend.fontsize'] = tsize
#     # plt.rcParams['xtick.direction'] = tdir
#     # plt.rcParams['ytick.direction'] = tdir
#     # plt.rcParams['xtick.major.size'] = major
#     # plt.rcParams['xtick.minor.size'] = minor
#     # plt.rcParams['ytick.major.size'] = major
#     # plt.rcParams['ytick.minor.size'] = minor
#     # sizeOfFont = 12
#     # fontProperties = {'family':'sans-serif','sans-serif':['Helvetica'],
#     #     'weight' : 'normal', 'size' : sizeOfFont}
#     # ticks_font = font_manager.FontProperties(family='Helvetica', style='normal',
#     #     size=sizeOfFont, weight='normal', stretch='normal')
#     # a = plt.gca()
#     # a.set_xticklabels(a.get_xticks(), fontProperties)
#     # a.set_yticklabels(a.get_yticks(), fontProperties)
#
#     fsize = 18
#     tsize = 18
#
#     tdir = 'in'
#
#     major = 5.0
#     minor = 3.0
#
#     # style = "seaborn-v0_8-paper"
#     style = "seaborn-paper"
#     # plt.style.use(")
#     plt.style.use(style)
#     plt.rcParams['text.usetex'] = True
#     # mpl.use('pgf')
#     plt.rcParams.update({'font.size': 12})
#     plt.rcParams['font.size'] = fsize
#     plt.rcParams['legend.fontsize'] = 13
#     plt.rcParams["axes.labelsize"] = 16
#     # plt.rcParams["xtick.labelsize"] = 'medium'
#     #
#     # plt.rcParams["ytick.labelsize"] = 'medium'
#     plt.rcParams['xtick.labelsize'] = 16
#     plt.rcParams["ytick.labelsize"] = 16
#     # plt.rc('axes', labelsize=MEDIUM_SIZE)
#     # plt.rcParams['xtick.direction'] = tdir
#     # plt.rcParams['ytick.direction'] = tdir
#     plt.rcParams['xtick.major.size'] = major
#     plt.rcParams['xtick.minor.size'] = minor
#     plt.rcParams['ytick.major.size'] = major
#     plt.rcParams['ytick.minor.size'] = minor

def plt_configure():
    fsize = 15
    tsize = 18

    tdir = 'in'

    major = 5.0
    minor = 3.0

    style = 'seaborn-paper'
    plt.style.use(style)
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = fsize
    # plt.rcParams['legend.fontsize'] = tsize
    # plt.rcParams['axes.titlesize'] = tsize
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['font.size'] = fsize
    plt.rcParams['legend.fontsize'] = 13
    plt.rcParams['xtick.direction'] = tdir
    plt.rcParams['ytick.direction'] = tdir
    plt.rcParams['xtick.major.size'] = major
    plt.rcParams['xtick.minor.size'] = minor
    plt.rcParams['ytick.major.size'] = major
    plt.rcParams['ytick.minor.size'] = minor
    plt.rcParams["axes.labelsize"] = 16
    # plt.rcParams["xtick.labelsize"] = 'medium'
    #
    # plt.rcParams["ytick.labelsize"] = 'medium'
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams["ytick.labelsize"] = 14





if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    filepath = filedialog.askopenfilename()
    plt_configure()
    nec = NECAnalysis(filepath, generate_files=False)
    nec.finalResults(random_activity_cut=[True, 5])
    # plt.rcParams['text.usetex'] = True
    # matplotlib.rc('text.latex', preamble=r'\usepackage{cmbright}')
