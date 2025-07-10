# *******************************************************
# * FILE: FWHMdicom.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

import os.path
import tkinter as tk
from tkinter import filedialog
from os import path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch
import pandas

pandas.set_option('display.max_columns', 500)


class FWHMEstimator:
    def __init__(self, _volume=None, voxelsize=None):
        self.volume = _volume
        if voxelsize is None:
            voxelsize = [1, 1, 1]
        voxelsize = np.array([float(s) for s in voxelsize])
        self.voxelSize = voxelsize
        self.fwhm = np.zeros((len(self.volume.shape)))
        self.fwtm = np.zeros((len(self.volume.shape)))
        self.r_2 = np.zeros((len(self.volume.shape)))
        self.sigma = np.zeros((len(self.volume.shape)))
        self.amplitude = np.zeros((len(self.volume.shape)))
        self.popt_all = [None, None, None]
        self.x = [None, None, None]
        self.y = [None, None, None]
        self.x_left_fwhm = [None, None, None]
        self.y_left_fwhm = [None, None, None]
        self.x_right_fwhm = [None, None, None]
        self.y_right_fwhm = [None, None, None]

        self.x_left_fwtm = [None, None, None]
        self.y_left_fwtm = [None, None, None]
        self.x_right_fwtm = [None, None, None]
        self.y_right_fwtm = [None, None, None]

        self._y_max_value = [None, None, None]
        self._x_max_value = [None, None, None]
        volumeTemp = ndimage.median_filter(self.volume, size=3)

        ind = np.where(volumeTemp == volumeTemp.max())
        print(ind)
        n_mm_mask = 8
        n_voxels = np.array(n_mm_mask/voxelsize, dtype=np.int32)
        extent = np.array([[ind[0][0] - n_voxels[0],
                          ind[0][0] + n_voxels[0]],
                          [ind[1][0] - n_voxels[1],
                          ind[1][0] + n_voxels[1]],
                          [ind[2][0] - n_voxels[2],
                          ind[2][0] + n_voxels[2]]])

        extent[extent < 0] = 0
        for i in range(3):
            a = extent[i]
            a[a >= self.volume.shape[i]] = self.volume.shape[i]

        maxTempNoFilter = self.volume[extent[0][0]:extent[0][1],
                                      extent[1][0]:extent[1][1],
                                      extent[2][0]:extent[2][1]].max()
        ind = np.where(self.volume == maxTempNoFilter)
        print(ind)
        # ind = ind[0]
        # if len(ind[0]) > 1:
        #     ind = list(ind)
        #     ind[0] = np.array([ind[0][0]])
        #     ind[1] = np.array([ind[1][0]])
        #     ind[2] = np.array([ind[2][0]])
        #     ind = tuple(ind)
        #remove baseline per axial slice

        # ind = np.argmax(volumeTemp)
        self.ind = ind
        self.image_axial = self.volume[:, :, ind[2]]
        self.image_axial = self.image_axial[:, :, 0]
        # remove baseline
        self.image_axial = self.image_axial - self.image_axial.min()
        self.image_coronal = self.volume[:, ind[1], :]
        self.image_coronal = self.image_coronal[:, 0, :]
        self.image_coronal = self.image_coronal - self.image_coronal.min()
        self.images = [self.image_axial, self.image_axial, self.image_coronal]

        # min_per_axial_slice = self.volume.min(axis=2)
        # self.volume = self.volume - min_per_axial_slice[:, :, np.newaxis]
        # filter the volume
        # self.volume = ndimage.median_filter(self.volume, size=3)
        self.max_value = self.volume[ind][0]

    def applyAlg(self):
        resolution_radial = 1*3
        resolution_tangencial = 1*3
        resolution_axial = 2*3
        for sh in range(len(self.volume.shape)):
            print("sh: {}".format(sh))
            # for sh in range(1):
            x = np.arange(self.volume.shape[sh])-self.volume.shape[sh]/2
            y = 0
            if sh == 0:
                number_of_profiles_y = int((resolution_radial)/self.voxelSize[1])
                number_of_profiles_z = int((resolution_radial)/self.voxelSize[2])
                y = self.volume[:, self.ind[1][0]-number_of_profiles_y: self.ind[1][0]+number_of_profiles_y, self.ind[2][0]-number_of_profiles_z:self.ind[2][0]+number_of_profiles_z]
                y = np.sum(np.sum(y, axis=2), axis=1)
                # y = y[:, 0]
            elif sh == 1:
                number_of_profiles_x = int((resolution_tangencial) / self.voxelSize[0])
                number_of_profiles_z = int((resolution_tangencial) / self.voxelSize[2])
                y = self.volume[self.ind[0][0]-number_of_profiles_x: self.ind[0][0] + number_of_profiles_x, :, self.ind[2][0]-number_of_profiles_z:self.ind[2][0]+number_of_profiles_z]
                # y = np.sum(np.sum(y, axis=0), axis=1)
                # y = self.volume[:, self.ind[1][0] - number_of_profiles_y: self.ind[1][0] + number_of_profiles_y,
                #     self.ind[2][0] - number_of_profiles_z:self.ind[2][0] + number_of_profiles_z]
                y = np.sum(np.sum(y, axis=2), axis=0)
                # y = y[0, :]

            elif sh == 2:
                number_of_profiles_x = int((resolution_axial) / self.voxelSize[0])
                number_of_profiles_y = int((resolution_axial) / self.voxelSize[1])
                y = self.volume[self.ind[0][0]-number_of_profiles_x:self.ind[0][0]+number_of_profiles_x, self.ind[1][0]-number_of_profiles_y:self.ind[1][0]+number_of_profiles_y, :]
                y = np.sum(np.sum(y, axis=1), axis=0)
                # y = y[0, :]


            # mean = self.ind[sh][0] * pixel_size_to_use
            pixel_size_to_use = self.voxelSize[sh]

            mean = x[self.ind[sh][0]] * pixel_size_to_use
            x = x * pixel_size_to_use
            sigma = self.volume.shape[sh] * pixel_size_to_use / 8
            # n = len(x)  # the number of data
            x_pixels_FHWM_left = x[:self.ind[sh][0]+1]
            y_pixels_FWHM_left = y[:self.ind[sh][0]+1]
            x_pixels_FHWM_right = x[self.ind[sh][0]:]
            y_pixels_FWHM_right = y[self.ind[sh][0]:]
            try:
                x_at_max = x[self.ind[sh][0]-1:self.ind[sh][0]+2]
                y_at_max = y[self.ind[sh][0]-1:self.ind[sh][0]+2]
                _x_inter = np.linspace(x_at_max.min(), x_at_max.max(), 100)
                _y_inter = np.interp(_x_inter, x_at_max, y_at_max)

                popt_parabole, pcov_parabole = curve_fit(FWHMEstimator.quadraticFit, _x_inter, _y_inter)
                max_value_to_FWHM = FWHMEstimator.quadraticFit(_x_inter, *popt_parabole).max()
                x_max_value = _x_inter[FWHMEstimator.quadraticFit(_x_inter, *popt_parabole) == max_value_to_FWHM]

                # popt_parabole, pcov_parabole = curve_fit(FWHMEstimator.quadraticFit, x_at_max, y_at_max)
                # max_value_to_FWHM = FWHMEstimator.quadraticFit(x_at_max, *popt_parabole).max()
                # x_max_value = x_at_max[FWHMEstimator.quadraticFit(x_at_max, *popt_parabole) == max_value_to_FWHM]
                # plt.plot(x_at_max, y_at_max, '.')
                # plt.plot(_x_inter, FWHMEstimator.quadraticFit(_x_inter, *popt_parabole), '--', label='fit')
                # plt.legend()
                # plt.show()

            except RuntimeError:
                popt_parabole = [1, 1, 1]
                max_value_to_FWHM = self.max_value
                x_at_max = 0
            # plt.plot(_x_inter, _y_inter)
            # plt.show()
            # mean = sum(x * y) / n  # note this correctiony[0,:]
            # sigma = sum(y * (x - mean) ** 2) / n  # note this correction
            sigma_squared = sigma ** 2
            try:
                # popt, pcov = curve_fit(FWHMEstimator.gaussian_fit, x, y, p0=[mean, sigma])
                popt, pcov = curve_fit(FWHMEstimator.gaussian_fit, x, y,
                                       p0=[self.max_value, mean, sigma_squared, np.min(y)])
                # popt_par = curve_fit
            except RuntimeError:
                popt = [1, 1, 1, 1]

            ss_res, ss_tot, r_squared = FWHMEstimator.statsFit(x, y, popt)

            # plt.plot(x, FWHMEstimator.gaussian_fit(x, *popt), 'ro:',
            #                                label='fit')
            # plt.show()
            try:
                x_left_fwhm , y_left_fwhm, x_inter_left_fwhm, y_inter_left_fwhm = FWHMEstimator._getXPointFWHM(x_pixels_FHWM_left, y_pixels_FWHM_left,
                                                                         max_value_to_FWHM/2)
            except ValueError:
                x_left_fwhm = np.inf
                y_left_fwhm = np.inf
            try:
                x_right_fwhm, y_right_fwhm, x_inter_left_fwhm, y_inter_left_fwhm = FWHMEstimator._getXPointFWHM(x_pixels_FHWM_right, y_pixels_FWHM_right,
                                                                          max_value_to_FWHM/2)
            except ValueError:
                x_right_fwhm = np.inf
                y_right_fwhm = np.inf

            try:
                x_left_fwtm, y_left_fwtm, x_inter_left_fwtm, y_inter_left_fwtm = FWHMEstimator._getXPointFWHM(x_pixels_FHWM_left, y_pixels_FWHM_left,
                                                                        max_value_to_FWHM/ 10)
            except ValueError:
                x_left_fwtm = np.inf
                y_left_fwtm = np.inf

            try:
                x_right_fwtm, y_right_fwtm, x_inter_right_fwtm, y_inter_right_fwtm = FWHMEstimator._getXPointFWHM(x_pixels_FHWM_right, y_pixels_FWHM_right,
                                                                          max_value_to_FWHM / 10)
            except ValueError:
                x_right_fwtm = np.inf
                y_right_fwtm = np.inf

            fwhm = np.abs(x_right_fwhm - x_left_fwhm)
            fwtm = np.abs(x_right_fwtm - x_left_fwtm)
            # plt.plot(x, y, 'b+:', label='data')
            # plt.plot(x_left_fwhm, y_left_fwhm, 'o:', label='left')
            # plt.plot(x_right_fwhm, y_right_fwhm, 'o:', label='right')
            # plt.legend()
            # plt.show()
            fwhm_gauss = 2.35482 * np.sqrt(popt[2])
            fwtm_gauss = 4.29193 * np.sqrt(popt[2])
            print("FIT RESULTS\n _______________\n")
            print("Amplitude: {}".format(popt[0]))
            print("Mean: {}".format(popt[1]))
            print("Sigma: {}".format(np.sqrt(popt[2])))
            print("R_2: {}".format(r_squared))
            # print("fwhm_gauss: {}".format(fwhm_gauss))
            print("fwhm: {}".format(fwhm))
            print("___________________________")

            # fwhm = 2.35482 * (popt[2])
            # fwtm = 4.29193 * (popt[2])
            # print("FWHM: {}".format(fwhm))
            # print("FWTM: {}".format(fwtm))

            self.popt_all[sh] = popt
            self.x[sh] = x
            self.y[sh] = y
            self.fwhm[sh] = fwhm
            self.fwtm[sh] = fwtm
            self.r_2[sh] = r_squared
            self.sigma[sh] = np.sqrt(sigma)
            self.amplitude[sh] = popt[0]
            self.x_left_fwhm[sh] = x_left_fwhm
            self.y_left_fwhm[sh] = y_left_fwhm
            self.x_right_fwhm[sh] = x_right_fwhm
            self.y_right_fwhm[sh] = y_right_fwhm

            self.x_left_fwtm[sh] = x_left_fwtm
            self.y_left_fwtm[sh] = y_left_fwtm
            self.x_right_fwtm[sh] = x_right_fwtm
            self.y_right_fwtm[sh] = y_right_fwtm

            self._y_max_value[sh] = max_value_to_FWHM
            self._x_max_value[sh] = x_max_value

    @staticmethod
    def _getXPointFWHM(x_pixels_FHWM_left, y_pixels_FWHM_left, valueatcurve):
        y_pixels_FWHM_left_abs = np.abs(y_pixels_FWHM_left - valueatcurve)
        y_min_left_loc = np.where(y_pixels_FWHM_left_abs == y_pixels_FWHM_left_abs.min())[0][0]
        # x_min_left_loc = x_min_left_loc
        extent = np.array([y_min_left_loc-1,
                            y_min_left_loc+2])

        extent[extent < 0] = 0
        extent[extent > y_pixels_FWHM_left.shape[0]-2] = y_pixels_FWHM_left.shape[0]+1


        x_left_to_inter = x_pixels_FHWM_left[extent[0]:extent[1]]
        y_left_to_inter = y_pixels_FWHM_left[extent[0]:extent[1]]
        x_new_left_inter = np.linspace(x_left_to_inter.min(), x_left_to_inter.max(), 100)
        y_new_left_inter = np.interp(x_new_left_inter, x_left_to_inter, y_left_to_inter)
        # plt.plot(x_new_left_inter, y_new_left_inter, 'o')
        # plt.plot(x_pixels_FHWM_left, y_pixels_FWHM_left, ".")
        # plt.show()
        diff = np.abs(y_new_left_inter - valueatcurve)
        x_left_fwhm = x_new_left_inter[diff == diff.min()]
        y_left_fwhm = y_new_left_inter[diff == diff.min()]
        return x_left_fwhm[0], y_left_fwhm[0], x_new_left_inter, y_new_left_inter

    @staticmethod
    def gaussian_fit(x, a, x0, sigma_squared, d):
        return a * np.exp(-((x - x0) ** 2 / (2 * sigma_squared))) + d

    @staticmethod
    def quadraticFit(x, a, b, c):
        return a*x**2 + b*x + c

    @staticmethod
    def statsFit(x, y, popt):
        residuals = y - FWHMEstimator.gaussian_fit(x, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        # print("Residuals sum of squares: {}".format(ss_res))
        # print("Total sum of squares: {}".format(ss_tot))

        return ss_res, ss_tot, r_squared

    @staticmethod
    def gauss(x, b, c):
        return (1 / c * np.sqrt(2 * np.pi)) * np.exp(-((x - b) ** 2) / (2 * c ** 2))


def plt_configure():
    fsize = 12
    tsize = 10

    tdir = 'in'

    major = 3.0
    minor = 1.0

    style = 'seaborn-dark-palette'
    plt.style.use(style)
    plt.rcParams['text.usetex'] = True
    # plt.rcParams['text.font.size'] = 10
    plt.rcParams['font.size'] = fsize
    plt.rcParams['legend.fontsize'] = tsize
    plt.rcParams['xtick.direction'] = tdir
    plt.rcParams['ytick.direction'] = tdir
    plt.rcParams['xtick.major.size'] = major
    plt.rcParams['xtick.minor.size'] = minor
    plt.rcParams['ytick.major.size'] = major
    plt.rcParams['ytick.minor.size'] = minor
    # plt.rcParams["grid.linestyle"] = (3, 7)


if __name__ == "__main__":
    from scipy import ndimage
    from ImageReader.DICOM.filesreader import DicomReader

    # from toor.NEMA.sensitivity_nema import plt_configure
    plt_configure()
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()

    d = DicomReader(_file_init=file_path)
    d.readDirectory()
    volumes = d.volumes()
    plt.imshow(np.sum(np.sum(volumes[:,:,:,:], axis=3), axis=2))
    plt.show()
    volumes = volumes[:, :, :,
              np.where(volumes.sum(axis=tuple(range(volumes.ndim - 1))) != 0)[0]]
    dicomHeader = d.dicomHeaders
    voxelSize = d.dicomHeaders[0].PixelSpacing
    voxelSize.append(d.dicomHeaders[0].SliceThickness)
    # voxelSize = [0.4, 0.4, 1.15]
    # totalPositions = dicomHeader[0].NumberOfTimeSlices
    totalPositions = volumes.shape[3]
    fwhm_total = np.zeros((totalPositions, 3))
    fwtm_total = np.zeros((totalPositions, 3))
    R2_total = np.zeros((totalPositions, 3))
    sigma_total = np.zeros((totalPositions, 3))
    amplitude_total = np.zeros((totalPositions, 3))
    # vectorPosition = np.zeros((totalPositions, 3))
    counter = 0
    list_ax_gauss = [[None, None, None]] * totalPositions
    list_ax_images = [[None, None, None]] * totalPositions
    fig_ax_gauss = [None] * totalPositions
    title_images = ['Radial \, cut \, (mm)', 'Tangencial \, cut \, (mm)', 'Axial \, cut \, (mm)']
    images = [None] * totalPositions
    x = [None for i in range(totalPositions)]
    y = [None for i in range(totalPositions)]
    popt = [None for i in range(totalPositions)]
    maximum_location = [None for i in range(totalPositions)]
    maximum_location_float = [0 for i in range(totalPositions)]
    parameters = [None for i in range(totalPositions)]
    x_left_fwhm_total = [None for i in range(totalPositions)]
    y_left_fwhm_total = [None for i in range(totalPositions)]
    x_right_fwhm_total = [None for i in range(totalPositions)]
    y_right_fwhm_total = [None for i in range(totalPositions)]
    x_left_fwtm_total = [None for i in range(totalPositions)]
    y_left_fwtm_total = [None for i in range(totalPositions)]
    x_right_fwtm_total = [None for i in range(totalPositions)]
    y_right_fwtm_total = [None for i in range(totalPositions)]
    x_max_value_total = [None for i in range(totalPositions)]
    y_max_value_total = [None for i in range(totalPositions)]



    for counter in range(totalPositions):
        volume = volumes[:, :, :, counter]
        print(counter)
        print("______________")
        # file_name = path.join(file_folder, file)
        try:
            estimator = FWHMEstimator(_volume=volume, voxelsize=voxelSize)
            estimator.applyAlg()
            fwhm_total[counter] = estimator.fwhm
            fwtm_total[counter] = estimator.fwtm
            R2_total[counter] = estimator.r_2
            sigma_total[counter] = estimator.sigma
            amplitude_total[counter] = estimator.amplitude
            images[counter] = estimator.images
            x[counter] = estimator.x
            y[counter] = estimator.y
            popt[counter] = estimator.popt_all
            mX = (estimator.ind[0][0] - volume.shape[0] / 2) * voxelSize[0]
            mY = (estimator.ind[1][0] - volume.shape[1] / 2) * voxelSize[1]
            mZ = (estimator.ind[2][0] - volume.shape[2] / 2) * voxelSize[2]
            maximum_location[counter] = "X:{:.2f}  Y:{:.2f} Z:{:.2f}".format(mX, mY, mZ)
            maximum_location_float[counter] = np.array([mX, mY, mZ])
            parameters[counter] = "{}".format(counter)
            x_left_fwhm_total[counter] = estimator.x_left_fwhm
            y_left_fwhm_total[counter] = estimator.y_left_fwhm
            x_right_fwhm_total[counter] = estimator.x_right_fwhm
            y_right_fwhm_total[counter] = estimator.y_right_fwhm

            x_left_fwtm_total[counter] = estimator.x_left_fwtm
            y_left_fwtm_total[counter] = estimator.y_left_fwtm
            x_right_fwtm_total[counter] = estimator.x_right_fwtm
            y_right_fwtm_total[counter] = estimator.y_right_fwtm
            x_max_value_total[counter] = estimator._x_max_value
            y_max_value_total[counter] = estimator._y_max_value
        except Exception as e:
            print(e)
            pass
        counter += 1
        # if counter>=1:
        #     break
        print("______________")
    abs_maximum_location_float = np.abs(np.array(maximum_location_float))

    score = np.arange(0, totalPositions)
    table_score = np.zeros((totalPositions, 8))
    # table_score[:, 0] = angles_of_files
    table_score[:, 1] = fwhm_total[:, 0]
    table_score[:, 2] = fwhm_total[:, 1]
    table_score[:, 3] = fwhm_total[:, 2]
    table_score[:, 4] = fwtm_total[:, 0]
    table_score[:, 5] = fwtm_total[:, 1]
    table_score[:, 6] = fwtm_total[:, 2]
    # table_score[:, 4] = np.mean(amplitude_total, axis=1)
    # table_score[:, 5] =  np.array(maximum_location)

    table_panda = pandas.DataFrame(table_score,
                                   columns=["Maximum location", 'Radial FWHM', 'Tangencial FWHM', "Axial FWHM",
                                            'Radial FWTM', 'Tangencial FWTM', "Axial FWTM", "R2"])
    keys = [('Mean axial FWHM', True), ('Mean FWHM', True), ("Mean Amplitude", False), ("Diff FWHM axial", True)]
    # ascend_list = [True, True, False, True]
    table_panda["Maximum location"] = maximum_location
    table_panda["R2"] = R2_total
    # for key, ascend in keys:
    #     table_panda = table_panda.sort_values(by=[key], ascending=ascend)
    #     table_panda["Score"] += score
    # table_panda = table_panda.sort_values(by=["Score"])
    table_panda = table_panda.round(4)


    print(table_panda)
    table_panda.to_csv(path.join(os.path.dirname(file_path), "results_panda"),index=False)

    # mean_axial_fwhm = np.sort(score, axis = None)
    # min_axial_fwhm = np.abs(fwhm_total[:, 0] - fwhm_total[:, 1])/2
    # min_coronal_fwhm
    # Plots
    #
    # ax_fwhm.set_x
    # plt.bar(angles, amplitude_total[:, 0], width=0.005)
    # plt.show()
    with PdfPages(path.join(os.path.dirname(file_path), "FWHMResults_.pdf")) as pdf:

        # indexes_volumes = np.zeros(20)
        # indexes_volumes[0:10] = np.arange(0, 10).astype(np.int32)
        # indexes_volumes[10:20] = np.arange(20, 30).astype(np.int32)
        _volumes = np.mean(volumes[:, :, :, :20], axis=3)
        # # _volumes = self._volumes[:, :, :, 4]
        # plt.imshow(np.max(_volumes, axis=1))
        fig_initial, ax_initial = plt.subplots()
        # ax_initial.set_title("{}".format(d.dicomHeaders[0].ReconstructionMethod))
        volumeto_print= np.max(_volumes[:, :, 4:-4], axis=1)
        ax_initial.imshow(volumeto_print/volumeto_print.max(), cmap="hot",
               extent=(-_volumes.shape[2] * voxelSize[2] / 2,
                       _volumes.shape[2] * voxelSize[2] / 2,
                       -_volumes.shape[0] * voxelSize[0] / 2,
                       _volumes.shape[0] * voxelSize[0] / 2), vmin=0.05, vmax=0.90, interpolation="Gaussian")
        ax_initial.set_ylabel('$Radial\, FoV \,(mm) \,$')
        ax_initial.set_xlabel('$Axial\, FoV \,(mm) \,$')
        pdf.savefig(fig_initial)
        plt.close(fig_initial)
        # # plt.imshow(volume[-20])
        # plt.show()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        the_table = ax.table(cellText=table_panda.values, colLabels=table_panda.columns, loc='center')
        pdf.savefig(fig, bbox_inches='tight')
        break_ = 10
        direction = 0
        labels = [["Radial CAFoV", "Tangencial CAFoV", "Axial CAFoV" ],
                  ["Radial 1/4", "Tangencial 1/4","Axial 1/4"],
                  ["Radial 1/4", "Tangencial 1/4" ,"Axial 1/4"],
                  ["Radial 1/4", "Tangencial 1/4" ,"Axial 1/4"],
                 ]
        for i in range(len(labels)):
            fig_fhwm, ax_fwhm = plt.subplots()
            ax_fwhm.plot(abs_maximum_location_float[i*break_:break_*(i+1)-1, direction], fwhm_total[i*break_:break_*(i+1)-1, 0], ".--", label=labels[i][0])
            ax_fwhm.plot(abs_maximum_location_float[i*break_:break_*(i+1)-1, direction], fwhm_total[i*break_:break_*(i+1)-1, 1], "s--", label=labels[i][1])
            ax_fwhm.plot(abs_maximum_location_float[i*break_:break_*(i+1)-1, direction], fwhm_total[i*break_:break_*(i+1)-1, 2], "^--", label=labels[i][2])
            ax_fwhm.legend()
            ax_fwhm.set_xlabel("$Radial \, Distance \, (mm)$")
            ax_fwhm.set_ylabel("$FWHM \, (mm)$")
            max_value_x = 20
            max_value_y = 1.5
            ax_fwhm.set_ylim(0, max_value_y)
            major_ticks_x = np.arange(0, max_value_x, max_value_x / 5)
            minor_ticks_x = np.arange(0, max_value_x, max_value_x / 20)

            major_ticks_y = np.arange(0, max_value_y, max_value_y/5)
            minor_ticks_y = np.arange(0, max_value_y, max_value_y/20)

            ax_fwhm.set_xticks(major_ticks_x)
            ax_fwhm.set_xticks(minor_ticks_x, minor=True)
            ax_fwhm.set_yticks(major_ticks_y)
            ax_fwhm.set_yticks(minor_ticks_y, minor=True)

            # And a corresponding grid
            ax_fwhm.grid(which='both')

            # Or if you want different settings for the grids:
            ax_fwhm.grid(which='minor', alpha=0.2, linestyle='--')
            ax_fwhm.grid(which='major', alpha=0.5, linestyle=':')
            ax_fwhm.grid(True)
            pdf.savefig(fig_fhwm)
            plt.close(fig_fhwm)

            fig_fwtm, ax_fwtm = plt.subplots()
            ax_fwtm.plot(abs_maximum_location_float[i * break_:break_ * (i + 1) - 1, direction],
                         fwtm_total[i * break_:break_ * (i + 1) - 1, 0], ".--", label=labels[i][0] )
            ax_fwtm.plot(abs_maximum_location_float[i * break_:break_ * (i + 1) - 1, direction],
                         fwtm_total[i * break_:break_ * (i + 1) - 1, 1], "s--", label=labels[i][1])
            ax_fwtm.plot(abs_maximum_location_float[i * break_:break_ * (i + 1) - 1, direction],
                         fwtm_total[i * break_:break_ * (i + 1) - 1, 2], "^--", label=labels[i][2])
            ax_fwtm.legend()
            ax_fwtm.set_xlabel("$Radial \, Distance \, (mm)$")
            ax_fwtm.set_ylabel("$FWTM \, (mm)$")

            ax_fwtm.minorticks_on()
            # And a corresponding grid
            ax_fwtm.grid(which='both')

            # Or if you want different settings for the grids:
            ax_fwtm.grid(which='minor', alpha=0.2, linestyle='--')
            ax_fwtm.grid(which='major', alpha=0.5, linestyle=':')
            ax_fwtm.grid(True)
            pdf.savefig(fig_fwtm)
            plt.close(fig_fwtm)


        # joined plots
        labels = [["$Radial$", "$Tangencial$", "$Axial$"],
                  ["$Radial$", "$Tangencial$", "$Axial$"],
                  ["$Radial$", "$Tangencial$", "$Axial$"],
                  ["$Radial$", "$Tangencial$", "$Axial$"],
                  ]
        labels = [["Radial", "Tangencial", "Axial"],
                  ["Radial", "Tangencial", "Axial"],
                  ["Radial", "Tangencial", "Axial"],
                  ["Radial", "Tangencial", "Axial"],
                  ]
        y_label = ["$Resolution \, at \, axial \, center \,(mm)$",
                   "$Resolution \, at \, 1/4  axial\, (mm)$",
                   "$Resolution \, at \, 1/4  axial\, (mm)$",
                   "$Resolution \, at \, 1/4  axial\, (mm)$"]

        # y_label = ["Resolution  at  axial  center(mm)",
        #            "Resolution  at  1/4  axial (mm)",
        #            "Resolution at  1/4  axial (mm)",
        #            "Resolution at  1/4  axial (mm)"]
        colors_ = plt.rcParams['axes.prop_cycle'].by_key()['color']
        cut= 0
        for i in range(len(labels)):
            fig_fhwm, ax_fwhm = plt.subplots()
            ax_fwhm.plot(abs_maximum_location_float[i * break_:break_ * (i + 1) - 1 - cut, direction],
                         fwhm_total[i * break_:break_ * (i + 1) - cut - 1, 0], ".--", label=labels[i][0] +" FWHM")
            ax_fwhm.plot(abs_maximum_location_float[i * break_:break_ * (i + 1) - cut - 1, direction],
                         fwhm_total[i * break_:break_ * (i + 1) - cut - 1 , 1], "s--", label=labels[i][1] +" FWHM")
            ax_fwhm.plot(abs_maximum_location_float[i * break_:break_ * (i + 1) - 1 - cut, direction],
                         fwhm_total[i * break_:break_ * (i + 1) - cut - 1, 2], "^--", label=labels[i][2] +" FWHM")

            ax_fwhm.set_xlabel("$Radial \, Distance \, (mm)$")
            ax_fwhm.set_ylabel(ylabel=y_label[i])
            ax_fwhm.minorticks_on()
            # max_value_x = 20
            # max_value_y = 1.5
            # ax_fwhm.set_ylim(0, max_value_y)
            # major_ticks_x = np.arange(0, max_value_x, max_value_x / 5)
            # minor_ticks_x = np.arange(0, max_value_x, max_value_x / 20)
            #
            # major_ticks_y = np.arange(0, max_value_y, max_value_y / 5)
            # minor_ticks_y = np.arange(0, max_value_y, max_value_y / 20)
            #
            # ax_fwhm.set_xticks(major_ticks_x)
            # ax_fwhm.set_xticks(minor_ticks_x, minor=True)
            # ax_fwhm.set_yticks(major_ticks_y)
            # ax_fwhm.set_yticks(minor_ticks_y, minor=True)

            # And a corresponding grid
            ax_fwhm.grid(which='both')

            # Or if you want different settings for the grids:
            ax_fwhm.grid(which='minor', alpha=0.2, linestyle='--')
            ax_fwhm.grid(which='major', alpha=0.5, linestyle=':')
            ax_fwhm.grid(True)


            ax_fwhm.plot(abs_maximum_location_float[i * break_:break_ * (i + 1) - 1 - cut, direction],
                         fwtm_total[i * break_:break_ * (i + 1) - 1 - cut, 0], ".-", label=labels[i][0]+" FWTM", color=colors_[0])
            ax_fwhm.plot(abs_maximum_location_float[i * break_:break_ * (i + 1) - 1 - cut, direction],
                         fwtm_total[i * break_:break_ * (i + 1) - 1- cut, 1], "s-", label=labels[i][1]+" FWTM", color=colors_[1])
            ax_fwhm.plot(abs_maximum_location_float[i * break_:break_ * (i + 1) - 1 - cut, direction],
                         fwtm_total[i * break_:break_ * (i + 1) - 1 -cut, 2], "^-", label=labels[i][2]+" FWTM", color=colors_[2])
            ax_fwhm.legend()
            pdf.savefig(fig_fhwm)
            plt.close(fig_fhwm)

        for c in range(totalPositions):

            counter = c
            fig_ax_gauss[counter], ((list_ax_images[counter]), (list_ax_gauss[counter])) = plt.subplots(2, 3)
            # fig_ax_gauss[counter].suptitle('Maximum at: {}'.format(maximum_location[counter]), fontsize=16)

            for i in range(3):
                list_ax_gauss[counter][i].grid(True)
                if i == 0:
                    list_ax_gauss[counter][i].set_ylabel('$Intensity$')
                    list_ax_images[counter][i].set_ylabel('$mm$')
                    list_ax_images[counter][i].imshow(images[counter][i].T, cmap="hot",
                                                      extent=(-images[counter][i].shape[0] * voxelSize[0] / 2,
                                                              images[counter][i].shape[0] * voxelSize[0] / 2,
                                                              -images[counter][i].shape[1] * voxelSize[1] / 2,
                                                              images[counter][i].shape[1] * voxelSize[1] / 2))
                    list_ax_images[counter][i].set_xlim(maximum_location_float[counter][0] - 4,
                                                        maximum_location_float[counter][0] + 4)
                    list_ax_images[counter][i].set_ylim(-maximum_location_float[counter][1] - 4,
                                                        -maximum_location_float[counter][1] + 4)
                elif i == 2:
                    list_ax_images[counter][i].imshow(images[counter][i], cmap="hot",
                                                      extent=( -images[counter][i].shape[1] * voxelSize[2] / 2,
                                                              images[counter][i].shape[1] * voxelSize[2] / 2,
                                                              -images[counter][i].shape[0] * voxelSize[1] / 2,
                                                              images[counter][i].shape[0] * voxelSize[1] / 2,
                                                             ))

                    list_ax_images[counter][i].set_xlim(maximum_location_float[counter][2] - 4,
                                                        maximum_location_float[counter][2] + 4)
                    list_ax_images[counter][i].set_ylim(-maximum_location_float[counter][0] - 4,
                                                        -maximum_location_float[counter][0] + 4)

                    # list_ax_gauss[counter][i].plot(x[counter][i], y[counter][i], '+:', label='data')
                    # list_ax_gauss[counter][i].plot(x[counter][i],
                    #                                FWHMEstimator.gaussian_fit(x[counter][i],
                    #                                                           *popt[counter][i]), '.-',
                    #                                label='fit')
                else:
                    list_ax_images[counter][i].imshow(images[counter][i], cmap="hot",
                                                      extent=(-images[counter][i].shape[1] * voxelSize[0] / 2,
                                                              images[counter][i].shape[1] * voxelSize[0] / 2,
                                                              -images[counter][i].shape[0] * voxelSize[1] / 2,
                                                              images[counter][i].shape[0] * voxelSize[1] / 2))
                    list_ax_images[counter][i].set_xlim(maximum_location_float[counter][1] - 4,
                                                        maximum_location_float[counter][1] + 4)
                    list_ax_images[counter][i].set_ylim(-maximum_location_float[counter][0] - 4,
                                                        -maximum_location_float[counter][0] + 4)
                list_ax_gauss[counter][i].plot(x[counter][i], y[counter][i], '.', label='data')
                list_ax_gauss[counter][i].fill_between(x=x[counter][i],
                                 y1=FWHMEstimator.gaussian_fit(x[counter][i],  *popt[counter][i]),
                                 # y2=[0] * len(df),
                                 color="#b030b0",
                                 alpha=0.15)


                list_ax_gauss[counter][i].minorticks_on()
                list_ax_gauss[counter][i].grid(which='both')
                arrow = FancyArrowPatch((x_left_fwhm_total[counter][i], y_left_fwhm_total[counter][i]),
                                        (x_right_fwhm_total[counter][i], y_right_fwhm_total[counter][i]),
                                        arrowstyle='<->', mutation_scale=1)

                list_ax_gauss[counter][i].add_patch(arrow)
                list_ax_gauss[counter][i].plot(x_left_fwhm_total[counter][i],  y_left_fwhm_total[counter][i], "^", markersize=3, color="red")
                list_ax_gauss[counter][i].plot(x_right_fwhm_total[counter][i],  y_right_fwhm_total[counter][i], "^", markersize=3, color="red")
                list_ax_gauss[counter][i].text((x_left_fwhm_total[counter][i]+x_right_fwhm_total[counter][i])/2, 1.1*y_left_fwhm_total[counter][i], "$FWHM$", fontsize=5, ha="center")

                list_ax_gauss[counter][i].plot(x_left_fwtm_total[counter][i], y_left_fwtm_total[counter][i], "^",
                                               markersize=3, color="green")
                list_ax_gauss[counter][i].plot(x_right_fwtm_total[counter][i], y_right_fwtm_total[counter][i], "^",
                                               markersize=3, color="green")

                list_ax_gauss[counter][i].text((x_left_fwtm_total[counter][i] + x_right_fwtm_total[counter][i]) / 2,
                                               1.1 * y_left_fwtm_total[counter][i], "$FWTM$", fontsize=5, ha="center")

                arrow = FancyArrowPatch((x_left_fwtm_total[counter][i], y_left_fwtm_total[counter][i]),
                                        (x_right_fwtm_total[counter][i], y_right_fwtm_total[counter][i]),
                                        arrowstyle='<->', mutation_scale=1)

                list_ax_gauss[counter][i].add_patch(arrow)

                list_ax_gauss[counter][i].plot(x_max_value_total[counter][i], y_max_value_total[counter][i], "v",
                                               markersize=3, color="#FF5733")
                # list_ax_gauss[counter][i].annotate('', xy=(x_left_fwhm_total[counter][i],
                #                                                    y_left_fwhm_total[counter][i]),
                #                                    xytext = (x_right_fwhm_total[counter][i],  y_right_fwhm_total[counter][i]),
                #                                    arrowprops=dict(arrowstyle='<->'),
                #                                    fontsize=6)
                # xytext=((x_right_fwhm_total[counter][i]+x_left_fwhm_total[counter][i])/2,
                #                                                            y_left_fwhm_total[counter][i]+y_left_fwhm_total[counter][i]*0.1)

                # Or if you want different settings for the grids:
                list_ax_gauss[counter][i].grid(which='minor', alpha=0.2, linestyle='--')
                list_ax_gauss[counter][i].grid(which='major', alpha=0.5, linestyle=':')


                list_ax_gauss[counter][i].set_xlim(maximum_location_float[counter][i]-4, maximum_location_float[counter][i]+4)
                # list_ax_gauss[counter][i].set_ylim(maximum_location_float[counter][i]-4, maximum_location_float[counter][i]+4)

                # list_ax_gauss[counter][i].plot(x[counter][i],
                #                                FWHMEstimator.gaussian_fit(x[counter][i],
                #                                                           *popt[counter][i]), '--',
                #                                label='fit')

                # list_ax_images[counter][i].set_title(, fontsize=10, weight='bold')
                list_ax_gauss[counter][i].set_xlabel(f'${title_images[i]}$')
                # list_ax_images[counter][i].text(-images[counter][i].shape[0] * voxelSize[1] / 2 + 5, -15,
                #                                 "FWHM: {} mm".format(np.round(fwhm_total[counter][i], 2)),
                #                                 color="white", fontsize=8)
                # list_ax_images[counter][i].text(-images[counter][i].shape[0] * voxelSize[1] / 2 + 5, -20,
                #                                 "FWTM: {} mm".format(np.round(fwtm_total[counter][i], 2)),
                #                                 color="white", fontsize=8)
                # list_ax_images[counter][i].text(5, 35,
                #                                 "Amplitude: {} mm".format(np.round(amplitude_total[counter][i], 2)),
                #                                 color="white")

                # list_ax_gauss[counter][i].text(3, 0.85*y[counter][i].max(),
                #                                "$R^2$: {}".format(np.round(R2_total[counter][i], 2), fontsize=8),
                #                                color="black")
                # list_ax_gauss[counter][i].legend()
                # plt.title('Fit')

            # plt.tight_layout()
            pdf.savefig(fig_ax_gauss[c])
            plt.close(fig_ax_gauss[c])
        # style.use('seaborn-poster')

        # # style.use('ggplot')
        # f_ax, (ax_fwhm) = plt.subplots(1, 1)
        # f_ax_fwtm, (ax_fwtm) = plt.subplots(1, 1)
        # f_ax_amplitude, (ax_amplitude) = plt.subplots(1, 1)
        # f_ax_mean, (ax_mean) = plt.subplots(1, 1)
        #
        # # f_ax_mean, (ax_amplitude) = plt.subplots(1, 1)
        # plot_fwhm_0_transverse = ax_fwhm.bar(angles, fwhm_total[:, 0], width=increment / 4, label='0º Transverse')
        # ax_fwhm.bar(angles + increment / 3, fwhm_total[:, 1], width=increment / 4, color="darkturquoise",
        #             label='90º Transverse')
        # ax_fwhm.bar(angles + 2 * increment / 3, fwhm_total[:, 2], width=increment / 4, color="darkblue",
        #             label='90º Coronal')
        # ax_fwhm.get_children()[2].set_color = "yellow"
        # ax_fwhm.set_xlabel('Correction Angle', fontsize=15, weight='bold', )
        # ax_fwhm.set_ylabel('FWHM', fontsize=15, weight='bold')
        # ax_fwhm.set_title('FWHM', fontsize=20, weight='bold')
        # ax_fwhm.legend()
        #
        # ax_amplitude.bar(angles, amplitude_total[:, 0], width=increment / 4, label='0º Transverse')
        # ax_amplitude.bar(angles + increment / 3, amplitude_total[:, 1], width=increment / 4, color="darkturquoise",
        #                  label='90º Transverse')
        # ax_amplitude.bar(angles + 2 * increment / 3, amplitude_total[:, 2], width=increment / 4, color="darkblue",
        #                  label='90º Coronal')
        # ax_amplitude.set_xlabel('Correction Angle', fontsize=15, weight='bold', )
        # ax_amplitude.set_ylabel('Pixel Amplitude', fontsize=15, weight='bold')
        # ax_amplitude.set_title('Amplitude', fontsize=20, weight='bold')
        # ax_amplitude.legend()
        #
        # ax_fwtm.bar(angles, fwtm_total[:, 0], width=increment / 4, label='0º Transverse')
        # ax_fwtm.bar(angles + increment / 3, fwtm_total[:, 1], width=increment / 4, color="darkturquoise",
        #             label='90º Transverse')
        # ax_fwtm.bar(angles + 2 * increment / 3, fwtm_total[:, 2], width=increment / 4, color="darkblue",
        #             label='90º Coronal')
        # ax_fwtm.set_xlabel('Correction Angle', fontsize=15, weight='bold', )
        # ax_fwtm.set_ylabel('FWTM', fontsize=15, weight='bold')
        # ax_fwtm.set_title('FWTM', fontsize=20, weight='bold')
        # ax_fwtm.legend()
        #
        # ax_mean.bar(angles, mean_total[:, 0], width=increment / 4, label='0º Transverse')
        # ax_mean.bar(angles + increment / 3, mean_total[:, 1], width=increment / 4, color="darkturquoise",
        #             label='90º Transverse')
        # ax_mean.bar(angles + 2 * increment / 3, mean_total[:, 2], width=increment / 4, color="darkblue",
        #             label='90º Coronal')
        # ax_mean.set_xlabel('Correction Angle', fontsize=15, weight='bold', )
        # ax_mean.set_ylabel('Mean', fontsize=15, weight='bold')
        # ax_mean.set_title('Mean', fontsize=20, weight='bold')
        # ax_mean.legend()

        # pdf.savefig(f_ax)
        # pdf.savefig(f_ax_fwtm)
        # pdf.savefig(f_ax_amplitude)
        # pdf.savefig(f_ax_mean)
