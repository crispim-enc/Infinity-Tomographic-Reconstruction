import tkinter as tk
from tkinter import filedialog
from os import listdir, path
from os.path import isfile, join
import numpy as np
from array import array
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar, exp
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.style as style
import re
import pandas

pandas.set_option('display.max_columns', 500)


# Sort a list  [im1 , im2, im11] instead of [im1, im11, im2]

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


class FWHMEstimator:
    def __init__(self, file_name, size_file_m, pixel_size, pixel_size_axial):
        self.file_name = file_name
        self.size_file_m = size_file_m
        self.pixel_size = pixel_size
        self.pixel_size_axial = pixel_size_axial
        self.size_file = self.size_file_m[0] * self.size_file_m[1] * self.size_file_m[2]
        self.__readFiles()

    def __readFiles(self):
        output_file = open(self.file_name, 'rb')  # define o ficheiro que queres ler
        a = array('f')  # define quantos bytes le de cada vez (float32)
        a.fromfile(output_file, self.size_file)  # lê o ficheiro binário (fread)
        output_file.close()  # fecha o ficheiro
        volume = np.array(a)  # não precisas
        # sensitivity_matrix = sensitivity_matrix.reshape((60,60,37), order='F')
        # sensitivity_matrix = sensitivity_matrix.reshape((120,120,73), order='F')
        self.volume = volume.reshape((self.size_file_m[0], self.size_file_m[1], self.size_file_m[2]),
                                     order='f')  # transforma em volume
        self._determine_linear_array()

    def _determine_linear_array(self):
        max_value = np.max(self.volume)
        ind = np.where(self.volume == max_value)
        self.ind = ind
        # axial_0d =
        # axial_0d =
        # axial_0d =
        self.fwhm = np.zeros((len(self.volume.shape)))
        self.fwtm = np.zeros((len(self.volume.shape)))
        self.mean = np.zeros((len(self.volume.shape)))
        self.sigma = np.zeros((len(self.volume.shape)))
        self.amplitude = np.zeros((len(self.volume.shape)))
        self.popt_all = [None] * 3
        self.x = [None] * 3
        self.y = [None] * 3
        self.image_axial = self.volume[:, :, ind[2]]
        self.image_axial = self.image_axial[:, :, 0]
        self.image_coronal = self.volume[ind[0], :, :]
        self.image_coronal = self.image_coronal[0, :, :]
        self.images = [self.image_axial, self.image_axial, self.image_coronal]
        for sh in range(len(self.volume.shape)):
            # for sh in range(1):
            x = np.arange(self.volume.shape[sh])
            if sh == 0:
                y = self.volume[:, ind[1], ind[2]]
                y = y[:, 0]
                pixel_size_to_use = self.pixel_size

            elif sh == 1:
                y = self.volume[ind[0], :, ind[2]]
                y = y[0, :]
                pixel_size_to_use = self.pixel_size
            elif sh == 2:
                y = self.volume[ind[0], ind[1], :]
                y = y[0, :]
                pixel_size_to_use = self.pixel_size_axial

            mean = ind[sh][0] * pixel_size_to_use
            x = x * pixel_size_to_use
            sigma = self.volume.shape[sh] * pixel_size_to_use / 4
            n = len(x)  # the number of data
            # mean = sum(x * y) / n  # note this correctiony[0,:]
            # sigma = sum(y * (x - mean) ** 2) / n  # note this correction
            sigma_squared = sigma ** 2
            try:
                popt, pcov = curve_fit(FWHMEstimator.gaus, x, y, p0=[max_value, mean, sigma_squared])
            except RuntimeError:
                popt = [1, 1, 1]
            # print("FIT RESULTS\n _______________\n")
            # print("Amplitude: {}".format(popt[0]))
            # print("Mean: {}".format(popt[1]))
            # print("Sigma: {}".format(np.sqrt(popt[2])))
            fwhm = 2.35482 * np.sqrt(popt[2])
            fwtm = 4.29193 * np.sqrt(popt[2])
            # print("FWHM: {}".format(fwhm))
            # print("FWTM: {}".format(fwtm))
            self.popt_all[sh] = popt
            self.x[sh] = x
            self.y[sh] = y
            self.fwhm[sh] = fwhm
            self.fwtm[sh] = fwtm
            self.mean[sh] = popt[1]
            self.sigma[sh] = np.sqrt(sigma)
            self.amplitude[sh] = popt[0]

    @staticmethod
    def gaus(x, a, x0, sigma_squared):
        return a * np.exp(-((x - x0) ** 2 / (2 * sigma_squared)))


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    matplotlib.rcParams['font.family'] = "Gill Sans MT"
    file_path = filedialog.askopenfilename()
    file_folder = path.dirname(file_path)
    file_folder = path.join(file_folder, "static_image")
    # onlyfiles = [f for f in listdir(file_folder) if isfile(join(file_folder, f))]
    onlyfiles = [path.join(file_folder, _) for _ in listdir(file_folder) if _.endswith(".T")]
    # onlyfiles = natural_sort(onlyfiles)
    angles_of_files = [file.split("Identifier")[1] for file in onlyfiles]
    # angles_of_files = [np.round(float(angle.split(".T")[0]),3) for angle in angles_of_files]

    # angles_of_files = np.arange(len(angles_of_files))
    onlyfiles = [x for _, x in sorted(zip(angles_of_files, onlyfiles), key=lambda pair: pair[0])]
    angles_of_files.sort()

    file_sizes = np.array([[None, None, None]] * len(onlyfiles))
    pixel_size_list = [None] * len(onlyfiles)
    c = 0
    for file in onlyfiles:
        for element in range(3):
            file_sizes[c, element] = int(file.split(")")[0].split("(")[1].split(",")[element])
        c += 1
        # file_name = "G:\\Easypet Scan 22 Jun 2021 - 17h 40m 32s\\static_image\\Easypet Scan 22 Jun 2021 - 17h 40m 32s_ IMAGE (110, 110, 162)Identifie0.5700000000000001r.T"
    # file_size = [110,110,162]
    # file_size = [147,147,213]
    file_sizes = list(file_sizes)
    pixel_size = 0.5
    reflector_y = 0.28
    number_of_crystals = [32, 2]
    size_of_crystal = 2  ## mm
    # pixel_size_axial = (number_of_crystals[0]*size_of_crystal+(number_of_crystals[0]-1)*reflector_y)/file_sizes[2]

    # min_angle = -2
    # max_angle = 2

    # angles = np.arange(min_angle,max_angle,increment)
    angles = np.array(angles_of_files)
    try:
        increment = angles[1] - angles[0]
    except TypeError:
        increment = 1.0
        angles = np.arange(len(angles_of_files))
    fwhm_total = np.zeros((len(onlyfiles), 3))
    fwtm_total = np.zeros((len(onlyfiles), 3))
    mean_total = np.zeros((len(onlyfiles), 3))
    sigma_total = np.zeros((len(onlyfiles), 3))
    amplitude_total = np.zeros((len(onlyfiles), 3))
    counter = 0
    list_ax_gauss = [[None, None, None]] * len(onlyfiles)
    list_ax_images = [[None, None, None]] * len(onlyfiles)
    fig_ax_gauss = [None] * len(onlyfiles)
    title_images = ['0º Axial', '90º Axial', '90º Coronal']
    images = [None] * len(onlyfiles)
    x = [None] * len(onlyfiles)
    y = [None] * len(onlyfiles)
    popt = [None] * len(onlyfiles)
    maximum_location = [None] * len(onlyfiles)
    parameters = [None] * len(onlyfiles)
    for file in onlyfiles:
        file_size = file_sizes[counter]
        pixel_size_axial = (number_of_crystals[0] * size_of_crystal + (number_of_crystals[0] - 1) * reflector_y) / \
                           file_size[2]
        print(file)
        print("______________")
        file_name = path.join(file_folder, file)
        estimator = FWHMEstimator(file_name, file_size, pixel_size, pixel_size_axial)
        fwhm_total[counter] = estimator.fwhm
        fwtm_total[counter] = estimator.fwtm
        mean_total[counter] = estimator.mean
        sigma_total[counter] = estimator.sigma
        amplitude_total[counter] = estimator.amplitude
        images[counter] = estimator.images
        x[counter] = estimator.x
        y[counter] = estimator.y
        popt[counter] = estimator.popt_all
        maximum_location[counter] = "{}{}{}".format(estimator.ind[0], estimator.ind[1], estimator.ind[2])
        parameters[counter] = "{}".format(angles_of_files[counter].split(".T")[0])

        counter += 1
        # if counter>=1:
        #     break
        print("______________")
    score = np.arange(0, len(onlyfiles))
    table_score = np.zeros((len(onlyfiles), 7))
    # table_score[:, 0] = angles_of_files
    table_score[:, 1] = (fwhm_total[:, 0] + fwhm_total[:, 1]) / 2
    table_score[:, 2] = np.mean(fwhm_total, axis=1)
    table_score[:, 3] = np.mean(amplitude_total, axis=1)
    table_score[:, 4] = np.abs(fwhm_total[:, 0] - fwhm_total[:, 1])
    # table_score[:, 5] =  np.array(maximum_location)

    table_panda = pandas.DataFrame(table_score,
                                   columns=["Parameter", 'Mean axial FWHM', 'Mean FWHM', "Mean Amplitude",
                                            "Diff FWHM axial", "Maximum location", "Score"])
    keys = [('Mean axial FWHM', True), ('Mean FWHM', True), ("Mean Amplitude", False), ("Diff FWHM axial", True)]
    # ascend_list = [True, True, False, True]
    table_panda["Maximum location"] = maximum_location
    table_panda["Parameter"] = parameters
    for key, ascend in keys:
        table_panda = table_panda.sort_values(by=[key], ascending=ascend)
        table_panda["Score"] += score
    # table_panda = table_panda.sort_values(by=["Score"])
    table_panda = table_panda.round(4)

    print(table_panda)
    # mean_axial_fwhm = np.sort(score, axis = None)
    # min_axial_fwhm = np.abs(fwhm_total[:, 0] - fwhm_total[:, 1])/2
    # min_coronal_fwhm
    # Plots

    # ax_fwhm.set_x
    # plt.bar( angles, amplitude_total[:, 0], width=0.005)
    # plt.show()
    with PdfPages(path.join(file_folder, "General_results_2.pdf")) as pdf:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('tight')
        ax.axis('off')
        the_table = ax.table(cellText=table_panda.values, colLabels=table_panda.columns, loc='center')
        pdf.savefig(fig, bbox_inches='tight')
        for c in range(len(onlyfiles)):

            counter = c
            fig_ax_gauss[counter], ((list_ax_images[counter]), (list_ax_gauss[counter])) = plt.subplots(2, 3)
            fig_ax_gauss[counter].suptitle('Par: {}'.format(parameters[counter]), fontsize=16)
            for i in range(len(file_size)):
                if i == 2:
                    list_ax_images[counter][i].imshow(images[counter][i].T, cmap="jet")
                else:
                    list_ax_images[counter][i].imshow(images[counter][i], cmap="jet")
                list_ax_images[counter][i].set_title(title_images[i], fontsize=10, weight='bold')
                list_ax_images[counter][i].text(5, 15, "FWHM: {} mm".format(np.round(fwhm_total[counter][i], 2)),
                                                color="white")
                list_ax_images[counter][i].text(5, 25, "FWTM: {} mm".format(np.round(fwtm_total[counter][i], 2)),
                                                color="white")
                list_ax_images[counter][i].text(5, 35,
                                                "Amplitude: {} mm".format(np.round(amplitude_total[counter][i], 2)),
                                                color="white")

                list_ax_gauss[counter][i].plot(x[counter][i], y[counter][i], 'b+:', label='data')
                list_ax_gauss[counter][i].plot(x[counter][i],
                                               FWHMEstimator.gaus(x[counter][i], *popt[counter][i]), 'ro:',
                                               label='fit')
                # list_ax_gauss[counter][i].legend()
                # plt.title('Fit')
                list_ax_gauss[counter][i].set_xlabel('Number of voxels N')
                list_ax_gauss[counter][i].set_ylabel('Intensity')
            pdf.savefig(fig_ax_gauss[c])
            plt.close(fig_ax_gauss[c])
        style.use('seaborn-poster')
        # style.use('ggplot')
        f_ax, (ax_fwhm) = plt.subplots(1, 1)
        f_ax_fwtm, (ax_fwtm) = plt.subplots(1, 1)
        f_ax_amplitude, (ax_amplitude) = plt.subplots(1, 1)
        f_ax_mean, (ax_mean) = plt.subplots(1, 1)

        # f_ax_mean, (ax_amplitude) = plt.subplots(1, 1)
        plot_fwhm_0_transverse = ax_fwhm.bar(angles, fwhm_total[:, 0], width=increment / 4, label='0º Transverse')
        ax_fwhm.bar(angles + increment / 3, fwhm_total[:, 1], width=increment / 4, color="darkturquoise",
                    label='90º Transverse')
        ax_fwhm.bar(angles + 2 * increment / 3, fwhm_total[:, 2], width=increment / 4, color="darkblue",
                    label='90º Coronal')
        ax_fwhm.get_children()[2].set_color = "yellow"
        ax_fwhm.set_xlabel('Correction Angle', fontsize=15, weight='bold', )
        ax_fwhm.set_ylabel('FWHM', fontsize=15, weight='bold')
        ax_fwhm.set_title('FWHM', fontsize=20, weight='bold')
        ax_fwhm.legend()

        ax_amplitude.bar(angles, amplitude_total[:, 0], width=increment / 4, label='0º Transverse')
        ax_amplitude.bar(angles + increment / 3, amplitude_total[:, 1], width=increment / 4, color="darkturquoise",
                         label='90º Transverse')
        ax_amplitude.bar(angles + 2 * increment / 3, amplitude_total[:, 2], width=increment / 4, color="darkblue",
                         label='90º Coronal')
        ax_amplitude.set_xlabel('Correction Angle', fontsize=15, weight='bold', )
        ax_amplitude.set_ylabel('Pixel Amplitude', fontsize=15, weight='bold')
        ax_amplitude.set_title('Amplitude', fontsize=20, weight='bold')
        ax_amplitude.legend()

        ax_fwtm.bar(angles, fwtm_total[:, 0], width=increment / 4, label='0º Transverse')
        ax_fwtm.bar(angles + increment / 3, fwtm_total[:, 1], width=increment / 4, color="darkturquoise",
                    label='90º Transverse')
        ax_fwtm.bar(angles + 2 * increment / 3, fwtm_total[:, 2], width=increment / 4, color="darkblue",
                    label='90º Coronal')
        ax_fwtm.set_xlabel('Correction Angle', fontsize=15, weight='bold', )
        ax_fwtm.set_ylabel('FWTM', fontsize=15, weight='bold')
        ax_fwtm.set_title('FWTM', fontsize=20, weight='bold')
        ax_fwtm.legend()

        ax_mean.bar(angles, mean_total[:, 0], width=increment / 4, label='0º Transverse')
        ax_mean.bar(angles + increment / 3, mean_total[:, 1], width=increment / 4, color="darkturquoise",
                    label='90º Transverse')
        ax_mean.bar(angles + 2 * increment / 3, mean_total[:, 2], width=increment / 4, color="darkblue",
                    label='90º Coronal')
        ax_mean.set_xlabel('Correction Angle', fontsize=15, weight='bold', )
        ax_mean.set_ylabel('Mean', fontsize=15, weight='bold')
        ax_mean.set_title('Mean', fontsize=20, weight='bold')
        ax_mean.legend()

        pdf.savefig(f_ax)
        pdf.savefig(f_ax_fwtm)
        pdf.savefig(f_ax_amplitude)
        pdf.savefig(f_ax_mean)

        # for i in range(crystal_geometry[0] * crystal_geometry[1]):
        # list_figures_crystal_energy_side_A[i], list_axes_crystal_energy_side_A[i] = plt.subplots(1, 1)
        # list_figures_crystal_energy_side_B[i], list_axes_crystal_energy_side_B[i] = plt.subplots(1, 1)
        # events_A = self.listMode[self.listMode[:, 2] == i + 1, 0]
        # events_B = self.listMode[self.listMode[:, 3] == i + 1, 1]
        # u_id_EA, indices_id_EA = np.unique(events_A, return_index=True)
        # u_id_EB, indices_id_EB = np.unique(events_B, return_index=True)

        # list_axes_crystal_energy_side_A[i].set_xlabel("KeV")
        # list_axes_crystal_energy_side_A[i].set_ylabel("Counts")
        # list_axes_crystal_energy_side_A[i].set_title("Crystal side A {}".format(i))
        # pdf.savefig(list_figures_crystal_energy_side_A[i])
        # plt.close(list_figures_crystal_energy_side_A[i])
        #
        # list_axes_crystal_energy_side_B[i].hist(events_B, u_id_EB, [0, 1200])
        # list_axes_crystal_energy_side_B[i].set_xlabel("KeV")
        # list_axes_crystal_energy_side_B[i].set_ylabel("Counts")
        # list_axes_crystal_energy_side_B[i].set_title("Crystal side B {}".format(i))
        # pdf.savefig(list_figures_crystal_energy_side_B[i])
        # plt.close(list_figures_crystal_energy_side_B[i])

    # plt.plot(x, y, 'b+:', label='data')
    # plt.plot(x, FWHMEstimator.gaus(x, *popt), 'ro:', label='fit')
    # plt.legend()
    # plt.title('Fit')
    # plt.xlabel('Number of voxels N')
    # plt.ylabel('Intensity')
#
# x = ar(range(10))
# y = ar([0,1,2,3,4,5,4,3,2,1])
#


#
