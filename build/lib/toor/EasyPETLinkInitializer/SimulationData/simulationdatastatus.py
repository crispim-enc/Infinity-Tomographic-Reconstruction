# *******************************************************
# * FILE: simulationdatastatus.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from EasyPETLinkInitializer.EasyPETDataReader import binary_data
import tkinter as tk
from tkinter import filedialog
from scipy.stats import t
from Corrections.PET.DecayCorrection import DecayCorrection


# def plt_configure():
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
#     plt.rcParams['font.size'] = fsize
#     # plt.rcParams['legend.fontsize'] = 12
#     # plt.rcParams["axes.labelsize"] = 'medium'
#     # plt.rcParams["xtick.labelsize"] = 'medium'
#     # plt.rcParams["ytick.labelsize"] = 'medium'
#     plt.rcParams['xtick.direction'] = tdir
#     plt.rcParams['ytick.direction'] = tdir
#     plt.rcParams['xtick.major.size'] = major
#     plt.rcParams['xtick.minor.size'] = minor
#     plt.rcParams['ytick.major.size'] = major
#     plt.rcParams['ytick.minor.size'] = minor
#     plt.rcParams["grid.linestyle"] = (3, 5)

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

    # plt.rcParams["gridOn"] = True


class SimulationStatusData:
    def __init__(self, listMode=None, acquisitionInfo=None, crystal_geometry=[32, 2], study_file=None, save_spectrum_file=False):
        if listMode is None:
            [listMode, Version_binary, header, dates, otherinfo, acquisitionInfo,
             stringdata, systemConfigurations_info, energyfactor_info,
             peakMatrix] = binary_data().open(study_file)
        self.listMode = listMode
        self.acquisitionInfo = acquisitionInfo
        self.study_file = study_file
        self.save_spectrum_file = save_spectrum_file
        self.crystal_geometry = crystal_geometry

        self.path_data_validation = os.path.join(os.path.dirname(self.study_file), "Data_Validation")
        if not os.path.isdir(self.path_data_validation):
            os.makedirs(self.path_data_validation)

    def individualEnergySpectrum(self):
        crystal_geometry = self.crystal_geometry
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

    def validation_status(self):
        crystal_geometry = self.crystal_geometry
        # self.listMode = self.applyCoincidenceWindow(self.listMode, 40000)

        f_energy_corrected, (ax_EA) = plt.subplots(1, 1)
        f_energy_corrected_b, (ax_EB) = plt.subplots(1, 1)
        # f_energy_corrected.suptitle('Energy Cut (keV)', fontsize=16)
        # f_individuals
        f_ids, ((ax_idA, ax_idB)) = plt.subplots(1, 2)
        # f_ids.suptitle('Crystal ID', fontsize=16)
        f_motor_c, ((ax_top_c, ax_bot_c)) = plt.subplots(1, 2)
        # f_motor_c.suptitle('Motors detection angles (.easypet)', fontsize=16)
        f_time, ax_time = plt.subplots(1, 1)

        u_bot_c, indices_bot = np.unique(self.listMode[:, 4], return_index=True)
        u_c, indices = np.unique(self.listMode[:, 5], return_index=True)
        # u_time, indices_time = np.unique(self.listMode[:, 6], return_index=True)
        base_color_1 = np.array([75.85, 0, 24.15]) / 100
        base_color_2 = np.array([0.8, 31.47, 67.73]) / 100
        hist_values = ax_EA.hist(self.listMode[:, 0],
                                 int(self.listMode[:, 0].max()), [0, self.listMode[:, 0].max()], edgecolor='None',
                                 alpha=0.85)
        ax_EA.plot(hist_values[1][:-1], hist_values[0], "-", color=base_color_2)
        SimulationStatusData._applygradient(hist_values, base_color_1, base_color_2)
        ax_EA.set_xlim(0, 1400)
        ax_EA.grid(True)
        ax_EA.set_xlabel("$Energy \, (keV)$")
        ax_EA.set_ylabel("$Prompts$")

        # Energy B
        hist_values = ax_EB.hist(self.listMode[:, 1],
                                 int(self.listMode[:, 1].max()), [0, self.listMode[:, 1].max()], edgecolor='None',
                                 alpha=0.85)
        ax_EB.plot(hist_values[1][:-1], hist_values[0], "-", color=base_color_2)
        SimulationStatusData._applygradient(hist_values, base_color_1, base_color_2)
        ax_EB.set_xlim(0, 1400)
        ax_EB.grid(True)
        ax_EB.set_xlabel("$Energy \, (keV)$")
        ax_EB.set_ylabel("$Prompts$")

        ax_EB.hist(self.listMode[:, 1], 1200, [0, 1200])

        fig_2d, ax_2d = plt.subplots(1, 1)
        ax_2d.hist2d(self.listMode[:, 2], self.listMode[:, 3], bins=[int(crystal_geometry[0] * crystal_geometry[1]),
                                                                     int(crystal_geometry[0] * crystal_geometry[1])],
                     cmap="jet")
        ax_2d.set_xlabel('$Crystal \, ID_A$')
        ax_2d.set_ylabel('$Crystal \, ID_B$')

        hist_idA = ax_idA.hist(self.listMode[:, 2], crystal_geometry[0] * crystal_geometry[1],
                            [np.min(self.listMode[:, 2]), np.max(self.listMode[:, 2])], density=True )
        SimulationStatusData._applygradient(hist_idA, base_color_1, base_color_2)
        ax_idA.set_xlabel("$Crystal \, ID_A$")
        ax_idA.set_ylabel("$Probability$")

        hist_idB = ax_idB.hist(self.listMode[:, 3], crystal_geometry[0] * crystal_geometry[1],
                               [np.min(self.listMode[:, 3]), np.max(self.listMode[:, 3])], density=True)
        SimulationStatusData._applygradient(hist_idB, base_color_1, base_color_2)
        ax_idB.set_xlabel("$Crystal \, ID_B$")
        ax_idB.set_ylabel("$Probability$")
        ax_idB.tick_params('y', labelleft=False)

        ax_bot_c.hist(self.listMode[:, 4], u_bot_c, density=True)
        ax_bot_c.set_xlabel("$Rear \, motor \, angle \,(º)$")
        ax_bot_c.set_ylabel("Probability \, of \, detection")
        ax_top_c.hist(self.listMode[:, 5], u_c)
        ax_time.hist(self.listMode[:, 6], 1000)
        # plt.show()

        # ax_EA.hist(self.listMode[:, 0], 1200, [0, 1200])
        # ax_EA.set_xlabel("KeV")
        # ax_EA.set_ylabel("Counts")
        # ax_EB.hist(self.listMode[:, 1], 1200, [0, 1200])
        # ax_idA.hist(self.listMode[:, 2], len(u_idA) + 1, [np.min(self.listMode[:, 2]), np.max(self.listMode[:, 2]) + 1])
        # ax_idB.hist(self.listMode[:, 3], len(u_idB) + 1, [np.min(self.listMode[:, 3]), np.max(self.listMode[:, 3]) + 1])
        # ax_bot_c.hist(self.listMode[:, 4], u_bot_c)
        # ax_top_c.hist(self.listMode[:, 5], u_c)
        # ax_time.hist(self.listMode[:, 6], u_time)

        plt.close(f_energy_corrected)
        plt.close(f_energy_corrected_b)
        plt.close(f_ids)
        plt.close(f_time)
        plt.close(f_motor_c)
        # plt.close(f_spectrum)
        # plt.close(f_motor)
        # plt.close((f_histogram2d))
        with PdfPages(os.path.join(self.path_data_validation, "DataValidation.pdf")) as pdf:
            # Pictures .easypet
            # Energy Histograms

            # f_motor_c.savefig(os.path.join(self.path_data_validation, "Motors easypet.png"))
            pdf.savefig(f_energy_corrected)
            pdf.savefig(f_ids)
            pdf.savefig(f_time)
            pdf.savefig(f_motor_c)
            pdf.savefig(fig_2d)

        # self.coincidenceWindowCharts()
        # pdf.savefig(f_spectrum)
        # pdf.savefig(f_motor)
        # pdf.savefig(f_histogram2d)

    # def activityMeanCalculation(self):

    @staticmethod
    def _applygradient(hist_values, base_color_1, base_color_2):
        alpha = 1
        N, bins, patches = hist_values
        colors = np.linspace(base_color_1, base_color_2, len(patches))
        bm = bins.max()
        bins_norm = bins / bm

        i = 0
        for bin_norm, patch in zip(bins_norm, patches):
            # grad = np.sin(np.pi * n_loops * bin_norm) / 15 + .04

            color = (colors[i][0], colors[i][1], colors[i][2], alpha)
            patch.set_facecolor(color)
            i += 1

    def applyCoincidenceWindow(self,listmode, value):
        value *= 10 ** -12 #convert to s
        diff_time = np.abs(listmode[:, 6] - listmode[:, 7])
        _all_events = diff_time[diff_time < value]
        listmode = listmode[diff_time < value]
        return listmode

    def coincidenceWindowCharts(self):

        decay_correction_class = DecayCorrection(listMode=self.listMode, acquisition_info=self.acquisitionInfo,
                                                 correct_decay=False)
        # decay_correction_class.list_mode_decay_correction()
        initial_activity = decay_correction_class.activity_on_subject_at_scanning_time()
        print("Initial activity: {} Bq".format(initial_activity))
        # uci
        print("Initial activity: {} uCi".format(initial_activity / 37000))
        time_indexes = self.acquisitionInfo["Turn end index"]
        # add 0 to the beginning
        time_indexes = np.insert(time_indexes, 0, 0)
        fig, ax = plt.subplots(1, 1)
        increment = 13
        Average_activity = np.zeros(len(time_indexes) // increment)

        coinc_window = np.logspace(0, 5, 60) * 10 ** -12
        n_counts_vector = np.zeros((len(time_indexes) // increment, coinc_window.shape[0]))
        el = 0
        for j in range(0, len(time_indexes) - increment, increment):
            # for j in range(0, 1, 1):

            listmode = self.listMode[time_indexes[j]:time_indexes[j + increment]]

            A0_frame = decay_correction_class.apply_decay(initial_activity, listmode[0, 6],
                                                          decay_correction_class.decay_half_life)
            Average_activity[el] = decay_correction_class.average_activity(A0_frame,
                                                                           listmode[-1, 6] -
                                                                           listmode[0, 6],
                                                                         decay_correction_class.decay_half_life)
            print("Frame: {}".format(el))
            print("Initial time: {} s".format(listmode[0, 6]))
            print("Final time: {} s".format(listmode[-1, 6]))
            print("A0 frame: {} Bq".format(A0_frame))
            print("A0 frame: {} uCi".format(A0_frame / 37000))
            print("Average activity: {} Bq".format(Average_activity[el]))
            print("Average activity: {} uCi".format(Average_activity[el] / 37000))
            print("-----------------------------------")

            # listmode = self.listMode
            diff_time = np.abs(listmode[:, 6] - listmode[:, 7])
            # coinc_window = np.array([1, 2.5, 5, 10,25,50,100,1000,2000,3000,4000])*10**-12

            n_counts = np.zeros(coinc_window.shape)
            n_en1 = np.zeros(coinc_window.shape)
            i = 0
            for value in coinc_window:
                _all_events = diff_time[diff_time < value]
                n_counts[i] = len(_all_events) / (listmode[-1, 6] - listmode[0, 6])
                # n_en1[i] = diff_time[self.listMode[:,0]>300]
                i += 1
            n_counts_vector[el] = n_counts
            ax.plot(coinc_window, n_counts, label="{} $\mu Ci$".format(int(np.round(Average_activity[el] / 37000, 0))))
            ax.fill_between(x=coinc_window,
                            y1=n_counts,
                            alpha=0.15)
            el += 1
        ax.plot(40 * np.ones(10) * (10 ** (-9)), np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 10), "--",
                label="$easyPETBased \, operation  \,mode$")
        ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.set_xlabel('$CTW, (s)$')
        ax.set_ylabel('$Bq$')
        ax.legend()

        # plt.close(fig)
        # n_counts_vector /=n_counts_vector.max()
        fig_norm, ax_norm = plt.subplots(1, 1)
        for i in range(n_counts_vector.shape[0]):
            n_counts = n_counts_vector[i]
            ax_norm.plot(coinc_window, n_counts / n_counts.max(),
                         label="{} $\mu Ci$".format(int(np.round(Average_activity[i] / 37000, 0))))
            ax_norm.fill_between(x=coinc_window,
                                 y1=n_counts / n_counts.max(),
                                 alpha=0.15)
        ax_norm.set_xscale('log')
        # ax.set_yscale('log')
        ax_norm.set_xlabel('$CTW \, (s)$')
        ax_norm.set_ylabel('$Bq/Bq$')
        ax_norm.plot(40 * np.ones(10) * (10 ** (-9)), np.linspace(ax_norm.get_ylim()[0], ax_norm.get_ylim()[1], 10),
                     "--",
                     label="$easyPETBased \, operation  \,mode$")

        ax_norm.legend(loc=0)

        fig_norm_plan, ax_norm_plan = plt.subplots(1, 1)
        fig_per, ax_per = plt.subplots(1, 1)
        for i in range(n_counts_vector.shape[0]):
            n_counts = n_counts_vector[i]
            n_counts_part = n_counts[(coinc_window > 10 ** -10) & (coinc_window < 10 ** -8)]
            n_counts_loc_max_plan = np.where(np.diff(n_counts_part) == np.diff(n_counts_part).min())[0][0]
            n_counts_max_plan = n_counts_part[n_counts_loc_max_plan]
            n_counts_max_loc_total = np.where(n_counts == n_counts_max_plan)[0][0]
            ax_norm_plan.plot(coinc_window, n_counts / n_counts_max_plan,
                              label="{} MBq".format(int(np.round(Average_activity[i]/(1*10**6), 0))))
            ax_norm_plan.fill_between(x=coinc_window, y1=n_counts / n_counts_max_plan, alpha=0.15)
            ax_per.plot(coinc_window[n_counts_max_loc_total:],
                        100 * np.abs(n_counts[n_counts_max_loc_total:] - n_counts_max_plan) / n_counts_max_plan,
                        label="{} $\mu Ci$".format(int(np.round(Average_activity[i] / 37000, 0))))
            ax_per.fill_between(x=coinc_window[n_counts_max_loc_total:],
                                y1=100 * np.abs(
                                    n_counts[n_counts_max_loc_total:] - n_counts_max_plan) / n_counts_max_plan,
                                alpha=0.15)
        ax_norm_plan.set_xscale('log')
        # ax.set_yscale('log')
        ax_norm_plan.set_xlabel('$CTW \, (s)$')
        ax_norm_plan.set_ylabel('$R_{trues + scattered}/R_{prompts}$')
        ax_norm_plan.plot(40 * np.ones(10) * (10 ** (-9)),
                          np.linspace(ax_norm_plan.get_ylim()[0], ax_norm_plan.get_ylim()[1], 10),
                          "--",
                          label="easyPETBased \n operation mode")
        # add a fill region  between 214 and 300 ps
        ax_norm_plan.fill_between(x=[100*10 ** -12, 300* 10 ** -12], y1=[ax_norm_plan.get_ylim()[1], ax_norm_plan.get_ylim()[1]], alpha=0.15, label="State-of-the-art \n electronics", color="blue")
        #legend top left
        ax_norm_plan.legend(loc=2)

        ax_per.set_xscale('log')
        ax_per.set_xlabel('$CTW \, (s)$')
        ax_per.set_ylabel('$R_{trues + scattered}/R_{prompts} \, (\%)$')
        ax_per.plot(40 * np.ones(10) * (10 ** (-9)), np.linspace(ax_per.get_ylim()[0], ax_per.get_ylim()[1], 10),
                    "--",
                    label="$easyPETBased \, operation  \,mode$")



        ax_per.legend()

        fig.savefig(os.path.join(os.path.dirname(self.study_file), 'Coincidence window.pdf'), dpi=300)
        fig_norm.savefig(os.path.join(os.path.dirname(self.study_file), 'normalized to max.pdf'), dpi=300)
        fig_norm_plan.savefig(os.path.join(os.path.dirname(self.study_file), 'normalized to plateu.pdf'), dpi=300, bbox_inches='tight')
        fig_per.savefig(os.path.join(os.path.dirname(self.study_file), 'percetagem randoms.pdf'), dpi=300)

        # plt.show()
        # plt.plot(self.listMode[:,7])

    @staticmethod
    def sin_fit(x, a, b, c, d):
        return a * np.cos(b * x + d) + c

    @staticmethod
    def gaussian_fit(x, a, b, c, d):
        # (1/(c*np.sqrt(2*np.pi))) * np.exp(-((x - b) ** 2)/(2*c**2)) + d
        return a * np.exp(-((x - b) ** 2) / (2 * c ** 2)) + d

    @staticmethod
    def geometric_p(x, p):
        return p * (1 - p) ** (x - 1)

    @staticmethod
    def fit_function(x, n, loc, scale):
        return t.pdf(x, n, loc, scale)
        # return binom.pmf(x, n, p)


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    plt_configure()
    # mpl.use("pgf")
    file_path = filedialog.askopenfilename()

    s = SimulationStatusData(study_file=file_path)
    # number_of_positions_top_lost = 20/0.225
    # time_gain = 0.005*number_of_positions_top_lost
    # diff = np.diff(s.listMode[:,4])
    # diff[diff!=0] = 1
    # cumsum = np.cumsum(diff) * time_gain
    #
    # new_time = s.listMode[1:,6]-cumsum
    # plt.plot(new_time, label="new_time")
    # plt.plot(s.listMode[:,6], label="old_time")
    # plt.legend()
    # plt.show()
    # s.validation_status()
    # s.coincidenceWindowCharts()
    # s.angle_correction()
