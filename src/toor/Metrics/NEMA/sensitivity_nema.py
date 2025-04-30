import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import pylab as plt

from EasyPETLinkInitializer.EasyPETDataReader import binary_data
import tkinter as tk
from scipy.optimize import curve_fit
from Geometry import SetParametricCoordinates, MatrixGeometryCorrection
from EasyPETLinkInitializer.Preprocessing import Sinogram


class SystemSensibility:
    def __init__(self, file_path=None, source_activity=5, one_file=True, n_turns_per_position=1):

        self.file_path = file_path
        self.listMode_per_file = []
        self.folders = None
        self._value = ''
        self.id = 0
        self.counts_per_acquisition = None
        self.bq_per_slice = None
        self.path_data_validation = None
        self.sinogram = None
        self.one_file = one_file
        self.n_turns_per_position = n_turns_per_position
        self.center_method = "Gaussian"
        self.source_activity = source_activity * 37000
        self.source_branching = 0.9060
        self.physical_center = 68
        self.motor_center = 68
        self.half_axial_fov = 72.68 / 2
        self.half_motor_axial_fov = 99 / 2
        self.bq_background = 0
        self._x_position = None
        self.counts_per_acquisition = None
        self.bq_per_slice = None
        self._energyWindowMin = 0
        self._energyWindowMax = 0
        self._applyEnergyCutBool = False
        # self._x_position - self.half_motor_axial_fov - (self.motor_center - self.physical_center)
        self.acquisition_time = None
        self._timeToRemove = 0
        self.Fov = 50
        self._applyFoVCutBool = False
        self.readFiles(self.file_path)

    @property
    def energyWindowMin(self):
        return self._energyWindowMin

    @property
    def energyWindowMax(self):
        return self._energyWindowMax

    def setEnergyWindow(self, min_value=None, max_value=None):
        self._energyWindowMax = max_value
        self._energyWindowMin = min_value
        self._applyEnergyCutBool = True

    def generateFiles(self):
        self._generateDataFromOneMichelogram()
        if not self.acquisitionInfo["Acquisition method"] == "Simulation":
            self._generateYDataBackground()
        # self._generate_y_data_from_one_file()

    def readFiles(self, file_name=None):
        if file_name is None:
            file_name = self.file_path
        try:
            [self.listMode, self.Version_binary, self.header, self.dates, self.otherinfo, self.acquisitionInfo,
             self.stringdata, self.systemConfigurations_info, self.energyfactor_info,
             self.peakMatrix_info] = binary_data().open(file_name)
            self.crystals_geometry = [self.systemConfigurations_info['array_crystal_x'],
                                      self.systemConfigurations_info['array_crystal_y']]

        except FileNotFoundError:
            raise FileNotFoundError

        return self.listMode, self.acquisitionInfo

    def setFoVCut(self, fov):
        self.Fov = fov
        self._applyFoVCutBool= True

    def applyFoVCut(self):
        if self._applyFoVCutBool:
            # plt.plot(np.unique(np.round(self.listMode[:,6],5)))
            # plt.plot(self.listMode[:,6])
            angle = np.degrees(np.arcsin(self.Fov /(2* self.systemConfigurations_info["distance_between_motors"])))
            print(f"Angle to cut{angle}")

            self.listMode = self.listMode[np.abs(self.listMode[:, 5]) <= angle]
            diff = np.diff(self.listMode[:, 6])
            bot_positions = int(360 / self.header[1])
            number_of_turns = len(self.acquisitionInfo["Turn end index"])
            self._timeToRemove = np.mean(diff[np.argsort(diff)][::-1][number_of_turns:bot_positions + number_of_turns])*bot_positions
            self.acquisition_time -= self._timeToRemove

    def _load_y_data(self):
        path_folder = os.path.dirname(os.path.dirname(self.file_path))
        self.counts_per_acquisition = np.loadtxt(os.path.join(path_folder, "counts_per_acquisition"))
        self.bq_per_slice = np.loadtxt(os.path.join(path_folder, "bq_per_acquisition"))
        self.path_data_validation = os.path.join(path_folder, "Sensibility Validation")

    def setOutputPath(self, name=None):
        path_folder = os.path.dirname(os.path.dirname(self.file_path))
        self.path_data_validation = os.path.join(path_folder, "Sensibility Validation", str(name))
        if not os.path.isdir(self.path_data_validation):
            os.makedirs(self.path_data_validation)

    def _generateYDataBackground(self):
        name_background_file = "Easypet Scan 24 Oct 2022 - 10h 46m 55s"
        name_background_file = "Easypet Scan 27 Jun 2022 - 10h 32m 52s"

        name_background_file = "Easypet Scan 25 Jan 2023 - 09h 31m 47s"
        path_file = os.path.join(os.path.dirname(os.path.dirname(self.file_path)), name_background_file,
                                 "{}.easypet".format(name_background_file))
        self.readFiles(path_file)
        self._acquisition_time_background = self.listMode[-1, 6] - self.listMode[0, 6]
        self._generateDataFromOneMichelogram(background_calculation=True)
        # np.savetxt(os.path.join(os.path.dirname(self.file_path), "bq_background"),
        #            np.array(self.bq_background))
        # np.savetxt(os.path.join(os.path.dirname(self.file_path), "bq_background"),
        #            self.counts_per_acquisition)       #

    def x_position(self):
        return self._x_position

    def rewriteXPosition(self, arr):
        self._x_position = np.array(arr)

    def _energyCut(self, apply=True):
        if apply:
            self.energy_window = [self._energyWindowMin, self.energyWindowMax]
            index_ea = np.where(
                (self.listMode[:, 0] < self.energy_window[0]) | (self.listMode[:, 1] < self.energy_window[0]))
            index_eb = np.where(
                (self.listMode[:, 0] > self.energy_window[1]) | (self.listMode[:, 1] > self.energy_window[1]))
            union_indexes_intersection = np.union1d(index_ea, index_eb)
            self.listMode = np.delete(self.listMode, union_indexes_intersection, axis=0)

    def _generateDataFromOneMichelogram(self, background_calculation=False):

        time_indexes = self.acquisitionInfo["Turn end index"]
        if self.acquisitionInfo["Acquisition method"] == "Simulation":
            simulation_file = True
            swap_sideAtoB = False

        else:
            simulation_file = False
            swap_sideAtoB = True

        if swap_sideAtoB:
            a = np.copy(self.listMode[:, 2])
            b = np.copy(self.listMode[:, 3])
            ea = np.copy(self.listMode[:, 0])
            eb = np.copy(self.listMode[:, 1])
            self.listMode[:, 2] = b
            self.listMode[:, 3] = a
            self.listMode[:, 0] = eb
            self.listMode[:, 1] = ea

        if background_calculation:
            print("Background Calculation")
            self.acquisition_time = self._acquisition_time_background

        else:
            # if self.n_turns_per_position == 1:
            #     self.acquisition_time = (self.listMode[time_indexes[self.n_turns_per_position], 6] -
            #                              self.listMode[0, 6]) # só é válido até corrigir a derivada
            # else:
            self.acquisition_time = (self.listMode[time_indexes[self.n_turns_per_position - 1], 6] -
                                     self.listMode[0, 6])
            self.acquisition_time = 478.55


        if not background_calculation:
            if not self.acquisitionInfo["Acquisition method"] == "Simulation":
                self.listMode = self.listMode[time_indexes[int(self.n_turns_per_position * 11)]:time_indexes[
                    -int(self.n_turns_per_position * 23)]]
            self.applyFoVCut()
            self._energyCut(self._applyEnergyCutBool)

        print(f"Acquisition_time: {self.acquisition_time} ")
        # self.listMode = self.listMode[self.listMode[:, 2] > 10]
        # self.listMode = self.listMode[self.listMode[:, 3] > 10]
        # self.listMode = self.listMode[self.listMode[:, 2] < 54]
        # self.listMode = self.listMode[self.listMode[:, 3] < 54]
        listmode = self.listMode
        print("Listmode shape: {}".format(listmode.shape))
        path_geometric_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            'system_configurations', 'x_{}__y_{}'.format(
                self.crystals_geometry[0], self.crystals_geometry[1]))
        MatrixCorrection = MatrixGeometryCorrection(operation='r', file_path=path_geometric_file)
        geometry_file = MatrixCorrection.coordinates
        crystals_geometry = self.crystals_geometry
        height = self.systemConfigurations_info["crystal_pitch_x"]
        crystal_width = self.systemConfigurations_info["crystal_pitch_y"]
        reflector_y = 2 * self.systemConfigurations_info["reflector_interior_A_y"]

        geometry_file[:, 1] = np.tile(np.round(np.arange(0, crystals_geometry[1] * crystal_width + 2 * reflector_y,
                                                         crystal_width + reflector_y) - (
                                                       crystal_width + reflector_y) *
                                               (crystals_geometry[1] - 1) / 2, 3), crystals_geometry[0] * 2)
        if not self.acquisitionInfo["Acquisition method"] == "Simulation":
            geometry_file[:crystals_geometry[0] * crystals_geometry[1], 1] *= -1

        # valid_turns = self._breakListModeIntoParts()
        z = np.repeat(np.arange(0, crystals_geometry[0] * height, height), crystals_geometry[1])

        geometry_file[0:crystals_geometry[0] * crystals_geometry[1], 2] = (z + 1)
        ## add 1.5 for 2019 aqusitions
        # geometry_file[32:64, 2] = z + 2.5
        geometry_file[
        crystals_geometry[0] * crystals_geometry[1]:crystals_geometry[0] * crystals_geometry[1] * 2,
        2] = (z + 1)
        #
        bins_x = len(np.unique(listmode[:, 4]))
        bins_y = 200
        axial_cuts = self.crystals_geometry[0] * 2

        parametric = SetParametricCoordinates(listMode=listmode, geometry_file=geometry_file,
                                              crystal_width=self.systemConfigurations_info[
                                                  "crystal_pitch_x"],
                                              crystal_height=self.systemConfigurations_info[
                                                  "crystal_pitch_y"],
                                              distance_between_motors=self.systemConfigurations_info[
                                                  "distance_between_motors"],
                                              distance_crystals=self.systemConfigurations_info[
                                                  "distance_between_crystals"],
                                              crystal_depth=self.systemConfigurations_info['crystal_length'],
                                              simulation_files=simulation_file, transform_into_positive=False)

        sinoClass = Sinogram(listMode=listmode, parametric=parametric)
        sinoClass.calculate_s_phi()
        sinoClass.updateLimits()
        sinoClass.calculateMichelogram(f2f_or_reb="reb", bins_x=bins_x, bins_y=bins_y)

        s = sinoClass.s
        phi = sinoClass.phi

        self.sinogram = sinoClass.michelogram

        number_s_steps = len(self.sinogram[2])
        s_step_size = (s.max() + np.abs(s.min())) / number_s_steps
        number_of_s_bins = int(np.ceil(s.max() / s_step_size))

        angles_axial = np.zeros(self.sinogram[0].shape[2])
        s_forinterpolation = np.arange(-30, 30, 0.5)
        print("________")
        counts = np.zeros(self.sinogram[0].shape[2])

        for i in range(self.sinogram[0].shape[2]):
            angles_axial[i] = self._center_sinogram(number_s_steps, number_of_s_bins, slice_=i,
                                                    _background_calculation=background_calculation)

            counts[i] = np.sum(self.centered_sinogram)

        if background_calculation:
            self.bq_background = counts / self.acquisition_time

        else:
            self.bq_per_slice = counts / self.acquisition_time

    def _center_sinogram(self, number_of_s_bins, bin_range, slice_=0, _background_calculation=False):
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

                    p0 = np.array([max_value, mean_posix[0], 10, np.min(result[i, :])])
                    popt, pcov = curve_fit(SystemSensibility.gaussian_fit, self.sinogram[2][:-1], result[i, :],
                                           method='lm', p0=p0)
                    # print(max_value)
                    # print((1 / popt[2] * np.sqrt(2 * np.pi)))
                    # print("_______")

                except RuntimeError as e:

                    popt = np.ones(4)
                    s_y = self.sinogram[2][:-1]
                    mean_posix = s_y[result[i, :] == result[i, :].max()][0]
                    popt[1] = mean_posix
                gaussian_r[i] = np.array(popt)

            diff_vector = np.abs(popt[1] - self.sinogram[2][:-1])
            try:
                loc_max = int(np.where(diff_vector == diff_vector.min())[0])
            except TypeError:
                print("Error locating maximum")
                failed_maxim += 1
                loc_max = 0
            # plt.plot(self.sinogram[2][:-1], result[i, :], 'b+:', label='data')
            # plt.plot(self.sinogram[2][:-1], SystemSensibility.gaussian_fit(self.sinogram[2][:-1], *popt), 'ro:',
            #                                label='fit')
            # plt.show()

            # gaussian_r = np.array(gaussian_r)
            SystemSensibility.statsFit(self.sinogram[2][:-1], result[i, :], popt)
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

        result = result[:, np.abs(self.sinogram[2][:-1]) <= top]
         # = plt.figure()
        self.f_gaussian_mean, ax = plt.subplots(1,1)
        plt.plot(self.sinogram[1][:-1], gaussian_r[:, 1], "-.", color="black")
        image = plt.imshow(result.T, 'hot', interpolation="gaussian", origin="lower", aspect="auto",
                   extent=[left, right, bottom, top])
        plt.xlabel("$Phi(^\circ)$")
        plt.ylabel("$S(mm)$")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.02)

        cbar = plt.colorbar(image, cax=cax)
        cbar.ax.get_yaxis().labelpad = 0
        cbar.ax.set_ylabel('$Counts$', rotation=90)
        cbar.outline.set_visible(False)

        plt.savefig(os.path.join(self.path_data_validation, "{}{}".format(name_sinogram, slice_)), dpi=100, bbox_inches='tight')
        plt.close(self.f_gaussian_mean)
        angle = np.degrees(np.arcsin((gaussian_r[:, 1].max() + gaussian_r[:, 1].min()) / 30))
        bottom = -20
        top = np.abs(bottom)

        f = plt.figure()
        # left = self.sinogram[1][0]
        # right = self.sinogram[1][-1]
        # bottom = self.sinogram[2][0]
        # top = self.sinogram[2][-1]

        self.centered_sinogram[:, np.abs(self.sinogram[2]) > 10] = 0
        self.centered_sinogram = self.centered_sinogram[:, np.abs(self.sinogram[2]) <= top]
        plt.imshow(self.centered_sinogram.T, 'hot', interpolation="gaussian", origin="lower", aspect="auto",
                   extent=[left, right, bottom, top])
        plt.colorbar()
        plt.plot(self.sinogram[1][:-1], np.zeros(len(self.sinogram[1][:-1])), "-.", color="black")
        plt.xlabel("$Phi \, (^\circ)$")
        plt.ylabel("$S \, (mm)$")
        plt.savefig(os.path.join(self.path_data_validation, "{}{}".format(name_center_sinogram, slice_)))
        plt.close(f)

        return angle

    @staticmethod
    def gaussian_fit_(x, a, b, c, d):
        return (1 / c * np.sqrt(2 * np.pi)) * np.exp(-((x - b) ** 2) / (2 * c ** 2))

    @staticmethod
    def gaussian_fit(x, a, b, c, d):
        return a * np.exp(-((x - b) ** 2 / (2 * c))) + d

    @staticmethod
    def statsFit(x, y, popt):
        residuals = y - SystemSensibility.gaussian_fit(x, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        # print("Residuals sum of squares: {}".format(ss_res))
        # print("Total sum of squares: {}".format(ss_tot))
        # print("R_2: {}".format(r_squared))
        # print("___________________________")
        return ss_res, ss_tot, r_squared

    def sensibilityCharts(self):
        plt.figure()
        try:
            self._x_position = self.sinogram[3]
            self._x_position = self.sinogram[3][:-1]
            si = (self.bq_per_slice - self.bq_background) / self.source_activity
            # si = (self.bq_per_slice) / self.source_activity
            sai = 100 * si / self.source_branching
            np.savetxt(os.path.join(self.path_data_validation, "sai{}".format(self._value)),
                       sai)
            np.savetxt(os.path.join(self.path_data_validation, "si{}".format(self._value)),
                       si)

            np.savetxt(os.path.join(self.path_data_validation, "x_position{}".format(self._value)),
                       self._x_position)
            np.savetxt(os.path.join(self.path_data_validation, "bq_per_slice{}".format(self._value)), self.bq_per_slice)
            # np.savetxt(os.path.join(self.path_data_validation, "bq_background{}".format(self._value)), self.bq_background)
        except TypeError:
            self._x_position = np.loadtxt(os.path.join(self.path_data_validation, "x_position"))
            sai = np.loadtxt(os.path.join(self.path_data_validation, "sai"))
            si = np.loadtxt(os.path.join(self.path_data_validation, "si"))
            self.bq_per_slice = np.loadtxt(os.path.join(self.path_data_validation, "bq_per_slice"))
            # self.bq_background = np.loadtxt(os.path.join(self.path_data_validation, "bq_background"))
            # self._load_y_data()


        plt.figure()
        plt.plot(self.x_position(), si, ".")
        plt.savefig(os.path.join(self.path_data_validation, "sensitivity{}.png".format(self._value)))


        plt.figure()
        ind = self.x_position().argsort()
        sai = sai[ind]

        x_sort = np.sort(self.x_position())
        max_abs_ind = np.where(sai == np.max(sai))
        print("desvio")
        print(x_sort[max_abs_ind])
        x_sort = x_sort - x_sort[max_abs_ind]
        ax = plt.plot(x_sort, sai, ".--", color="black")
        # plt.legend(frameon=False, loc=2)
        plt.xlabel('$Axial  \, Position \, (mm)$', labelpad=10)
        plt.ylabel('$Absolute \, sensitivity \, (\%)$', labelpad=10)
        # plt.xlim(0, 117)
        plt.tick_params(top=True, right=True, which='both', direction='in')
        # plt.xlim(-37, +37)

        plt.grid(which='both')

        # Or if you want different settings for the grids:
        plt.grid(which='minor', alpha=0.2, linestyle='--')
        plt.grid(which='major', alpha=0.5, linestyle=':')
        plt.grid(True)
        plt.savefig(os.path.join(self.path_data_validation, "Absolute sensivity {}.png".format(self._value)), dpi=300,
                    pad_inches=.1,
                    bbox_inches='tight')

        condition = np.abs(self._x_position) <= self.half_axial_fov
        print("Total system sensitivity: {} ".format(np.round(np.sum(si) / len(self.bq_per_slice), 4)))
        print("Total system absolute sensibility : {} %".format(np.round(np.sum(sai) / len(self.bq_per_slice),4)))
        condition = np.abs(self._x_position) <= 35
        print("Total system sensitivity Mice: {} ".format(
            np.round(np.sum(si[condition]) / len(self.bq_per_slice[condition]), 4)))

        print("Total system absolute sensibility mice: {} %".format(np.round(np.sum(sai[condition]) / len(self.bq_per_slice[condition]),4)))



        print("Total system absolute sensibility : {} %".format(np.sum(sai) / len(self.bq_per_slice)))


def plt_configure():
    # fsize = 14
    # tsize = 14
    #
    # tdir = 'in'
    #
    # major = 3.0
    # minor = 1.0
    #
    # style = 'seaborn-dark-palette'
    # plt.style.use(style)
    # plt.rcParams['text.usetex'] = True
    # # plt.rcParams['text.font.size'] = 10
    # plt.rcParams['font.size'] = fsize
    # plt.rcParams['legend.fontsize'] = tsize
    # plt.rcParams['xtick.direction'] = tdir
    # plt.rcParams['ytick.direction'] = tdir
    # plt.rcParams['xtick.major.size'] = major
    # plt.rcParams['xtick.minor.size'] = minor
    # plt.rcParams['ytick.major.size'] = major
    # plt.rcParams['ytick.minor.size'] = minor
    # sizeOfFont = 12
    # fontProperties = {'family':'sans-serif','sans-serif':['Helvetica'],
    #     'weight' : 'normal', 'size' : sizeOfFont}
    # ticks_font = font_manager.FontProperties(family='Helvetica', style='normal',
    #     size=sizeOfFont, weight='normal', stretch='normal')
    # a = plt.gca()
    # a.set_xticklabels(a.get_xticks(), fontProperties)
    # a.set_yticklabels(a.get_yticks(), fontProperties)

    fsize = 18
    tsize = 18

    tdir = 'in'

    major = 5.0
    minor = 3.0

    # style = "seaborn-v0_8-paper"
    style = "seaborn-paper"
    # plt.style.use(")
    plt.style.use(style)
    plt.rcParams['text.usetex'] = True
    # mpl.use('pgf')
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['font.size'] = fsize
    plt.rcParams['legend.fontsize'] = 13
    plt.rcParams["axes.labelsize"] = 16
    # plt.rcParams["xtick.labelsize"] = 'medium'
    #
    # plt.rcParams["ytick.labelsize"] = 'medium'
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams["ytick.labelsize"] = 16
    # plt.rc('axes', labelsize=MEDIUM_SIZE)
    # plt.rcParams['xtick.direction'] = tdir
    # plt.rcParams['ytick.direction'] = tdir
    plt.rcParams['xtick.major.size'] = major
    plt.rcParams['xtick.minor.size'] = minor
    plt.rcParams['ytick.major.size'] = major
    plt.rcParams['ytick.minor.size'] = minor

plt_configure()

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
    # file_path = filedialog.askopenfilename()
    file_path = "/media/crispim/Storage/Simulations/6-December-2022_14h32_64turn_0p005s_1p80bot_0p23top_range108_ListMode/6-December-2022_14h32_64turn_0p005s_1p80bot_0p23top_range108_ListMode.easypet"
    plt_configure()

    # listMode = sens.listMode
    # value = 1
    # value *= 10 ** -9
    # diff_time = np.abs(listMode[:, 6] - listMode[:, 7])
    # _all_events = diff_time[diff_time < value]
    # sens.listMode = sens.listMode[diff_time < value]
    # sens._value = "{}s".format(value)
    fov_variation = True
    energy_variation = True
    time_variation = False
    fov_size = np.arange(10, 45, 5)
    output_paths = []
    coin_windows = [30, 40000]
    from EasyPETLinkInitializer.SimulationData import SimulationStatusData
    if time_variation:

        for coin_window in coin_windows:
            sens = SystemSensibility(file_path)
            sens.setOutputPath(f"time_{coin_window}_ps")
            output_paths.append(sens.path_data_validation)
            status = SimulationStatusData(listMode=sens.listMode, study_file=sens.path_data_validation )

            sens.listMode = status.applyCoincidenceWindow(sens.listMode, coin_window)  # ps
            # sens.setTimeWindow(time)
            sens.setFoVCut(20)
            # sens.generateFiles()
            sens.sensibilityCharts()
        f_c = 0
        fig_ctr, ax_ctr = plt.subplots(1, 1)

        for file in output_paths:
            print("_________________________")
            print(file)
            x_sort = np.loadtxt(os.path.join(file, "x_position"))
            sai = np.loadtxt(os.path.join(file, "sai"))
            max_abs_ind = np.where(sai == np.max(sai))
            print("Maximum: {}".format(np.max(sai)))
            print(np.sum(sai))
            # print(x_sort[max_abs_ind])
            x_sort = x_sort - x_sort[max_abs_ind]
            plt.plot(x_sort[1:-1], sai[1:-1], ".--", label="$\Delta t$ = {} ps".format(coin_windows[f_c]))
            f_c += 1
        plt.legend()
        plt.xlabel("$Position (mm)$")
        plt.ylabel("$Absolute Sensitivity$")
        # plt.tick_params(top=True, right=True, which='both', direction='in')
        # plt.xlim(-37, +37)
        plt.minorticks_on()

        plt.grid(which='both')

        # Or if you want different settings for the grids:
        plt.grid(which='minor', alpha=0.2, linestyle='--')
        plt.grid(which='major', alpha=0.5, linestyle=':')
        plt.grid(True)
        # fig_ctr.savefig(os.path.join(os.path.dirname(file), "Absolute sensivity CTR.png"), dpi=300)

        plt.savefig(os.path.join(os.path.dirname(file), "Absolute sensivity CTR.png"), dpi=300,
                    pad_inches=.1,
                    bbox_inches='tight')


    if fov_variation:

        for fov in fov_size:
            print("FOV: {}".format(fov))
            sens = SystemSensibility(file_path)
            sens.setOutputPath(f"fov_{fov}_mm")
            output_paths.append(sens.path_data_validation)
            sens.setFoVCut(fov)
            # sens.generateFiles()
            sens.sensibilityCharts()

        i = 0
        plt.figure()
        for file in output_paths[1:]:
            x_sort = np.loadtxt(os.path.join(file, "x_position"))
            sai = np.loadtxt(os.path.join(file, "sai"))
            max_abs_ind = np.where(sai == np.max(sai))
            print(file)
            print("Maximum: {}".format(np.max(sai)))
            print("Minimum: {}".format(np.min(sai)))
            # print(x_sort[max_abs_ind])
            x_sort = x_sort - x_sort[max_abs_ind]
            # ax = plt.plot(x_sort[1:-1], sai[1:-1], ".--", label="$D_{fov}$" + "$: {} \,  mm$".format(fov_size[1:][i]))
            ax = plt.plot(x_sort[1:-1], sai[1:-1], ".--", label="$D_{fov}$" + ": {} mm".format(fov_size[1:][i]))
            i += 1
        plt.legend()

        plt.xlabel('$Axial  \, Position \, (mm)$', labelpad=10)
        plt.ylabel('$Absolute \, sensitivity \, (\%)$', labelpad=10)
        # plt.xlim(0, 117)
        plt.minorticks_on()

        # plt.tick_params(top=True, right=True, which='both', direction='in')
        # plt.xlim(-37, +37)
        plt.grid(which='both')

        # Or if you want different settings for the grids:
        plt.grid(which='minor', alpha=0.2, linestyle='--')
        plt.grid(which='major', alpha=0.5, linestyle=':')

        plt.savefig(os.path.join(os.path.dirname(file), "Absolute sensivity fov variation.png"), dpi=300,
                    pad_inches=.1,
                    bbox_inches='tight')
    if energy_variation:
        pico = 511
        energy_cut = np.round(np.array([
                      [0, 1500],
                      [pico - pico*0.3, pico + pico*0.3],
                      [pico - pico*0.5, pico + pico*0.5]]), 0)
        energy_cut.astype(int)
        output_paths = []
        i = 0
        for energy in energy_cut:
            sens = SystemSensibility(file_path)
            sens.setOutputPath(f"energy_{energy[0]} to {energy[1]}_kev")
            output_paths.append(sens.path_data_validation)
            sens.setEnergyWindow(energy[0], energy[1])
            sens.setFoVCut(20)
            # sens.generateFiles()
            sens.sensibilityCharts()
            i += 1

        # x_sort = np.sort(sens.x_position())
        # x_sort = np.arange(0,63)
        # markers = ["--s", "-o",  "o-"]
        i = 0
        plt.figure()
        df = pd.DataFrame()
        for file in output_paths:
            x_sort = np.loadtxt(os.path.join(file, "x_position"))
            sai = np.loadtxt(os.path.join(file, "sai"))

            max_abs_ind = np.where(sai == np.max(sai))
            print(file)
            print("Maximum: {}".format(np.max(sai)))
            print("Minimum: {}".format(np.min(sai)))
            print(x_sort[max_abs_ind])
            x_sort = x_sort - x_sort[max_abs_ind]
            # label_ = f"${energy_cut[i, 0]} \, - \, {energy_cut[i, 1]} \,keV$"
            label_ = f"{energy_cut[i, 0]} - {energy_cut[i, 1]} keV"
            df[label_ + "Absolute sensitivity"] = sai.tolist()
            if i == 0:
                # label_ = "$No\, energy \, window\, applied$"
                label_ = "No energy window applied"

            ax = plt.plot(x_sort[1:-1], sai[1:-1], ".--", label=label_ )
            i += 1
        df["x_position"] = x_sort.tolist()
        print(df.to_string())
        plt.legend()

        plt.xlabel('$Axial  \, Position \, (mm)$', labelpad=10)
        plt.ylabel('$Absolute \, sensitivity \, (\%)$', labelpad=10)
        # plt.xlim(0, 117)
        # plt.tick_params(top=True, right=True, which='both', direction='in')
        # plt.xlim(-37, +37)
        plt.minorticks_on()

        plt.grid(which='both')

        # Or if you want different settings for the grids:
        plt.grid(which='minor', alpha=0.2, linestyle='--')
        plt.grid(which='major', alpha=0.5, linestyle=':')
        plt.grid(True)

        plt.savefig(os.path.join(os.path.dirname(file), "Absolute sensivity energy variation.png"), dpi=300,
                    pad_inches=.1,
                    bbox_inches='tight')
