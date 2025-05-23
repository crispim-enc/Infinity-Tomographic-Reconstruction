import os
import numpy as np

import tkinter as tk
from tkinter import filedialog
# mpl.rcParams.update(mpl.rcParamsDefault)

import matplotlib.pyplot as plt
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})
from scipy.optimize import curve_fit
from scipy import ndimage
from EasyPETLinkInitializer.EasyPETDataReader import binary_data
from Geometry import SetParametricCoordinates, MatrixGeometryCorrection
from skimage import filters
from EasyPETLinkInitializer.SimulationData import SimulationStatusData


class DataTest:
    def __init__(self, filename=None):
        self.study_file = filename
        self.study_file_original = os.path.join(os.path.dirname(self.study_file),
                                                "{} Original data.easypetoriginal".format(
                                                    os.path.basename(self.study_file).split(".")[0]))


        [self.listMode, self.Version_binary, self.header, self.dates, self.otherinfo, self.acquisitionInfo,
         self.stringdata, self.systemConfigurations_info, self.energyfactor_info, self.peakMatrix] = binary_data().open(self.study_file)
        try:
            self.peakMatrix = np.array(self.peakMatrix.split(" "))
            self.peakMatrix = self.peakMatrix[self.peakMatrix != ""].astype(np.float32)
        except AttributeError:
            self.peakMatrix = np.array(self.peakMatrix)

        self.number_of_crystals_per_module = self.systemConfigurations_info["array_crystal_x"] * \
                                             self.systemConfigurations_info["array_crystal_y"]
        self.listMode_original = binary_data().open_original_data(self.study_file_original)

        path_folder = os.path.dirname(os.path.dirname(self.study_file))
        self.path_data_validation = os.path.join(path_folder, "Sinodata")
        if not os.path.isdir(self.path_data_validation):
            os.makedirs(self.path_data_validation)

        self.crystals_geometry = [self.systemConfigurations_info["array_crystal_x"],
                                  self.systemConfigurations_info["array_crystal_y"]]

        self._spectrumA = None
        self._spectrumB = None
        self._ratioA = None
        self._ratioB = None

        print(self.header)

    def write_pdf_stats(self):
        """"""

    def calculateSpectrumA(self):
        spectrumA = self.listMode_original[:, 0] + self.listMode_original[:, 1]
        self._spectrumA = spectrumA
        return spectrumA

    def calculateSpectrumB(self):
        spectrumB = self.listMode_original[:, 2] + self.listMode_original[:, 3]
        self._spectrumB = spectrumB
        return spectrumB

    @property
    def spectrumA(self):
        return self._setSpectrumA

    @property
    def spectrumB(self):
        return self._setSpectrumB

    def calculateRatioA(self):
        spectrumA = self.calculateSpectrumA()
        ratioA = (self.listMode_original[spectrumA != 0, 0] - self.listMode_original[spectrumA != 0, 1]) / spectrumA[spectrumA != 0]
        self._ratioA = ratioA
        return ratioA

    def calculateRatioB(self):
        spectrumB = self.calculateSpectrumB()
        ratioB = (self.listMode_original[spectrumB != 0, 3] - self.listMode_original[spectrumB != 0, 2]) / spectrumB[spectrumB != 0]
        self._ratioB = ratioB
        return ratioB

    @property
    def ratioA(self):
        return self._ratioA

    @property
    def ratioB(self):
        return self._ratioB


    def original_data_analysis(self):
        f_histogram2d, (ax_histogram2d_A, ax_histogram2d_B) = plt.subplots(1, 2)
        f_spectrum, ((ax_spectrumA, ax_spectrumB), (ax_ratioA, ax_ratioB)) = plt.subplots(2, 2)
        f_motor, ((ax_top, ax_bot)) = plt.subplots(1, 2)
        f_a, ((ax_a1, ax_a2), (ax_a3, ax_a4)) = plt.subplots(2, 2)
        self.listMode_original[:, 0:4] = self.listMode_original[:, 0:4] // 4

        spectrumA = self.listMode_original[:, 0] + self.listMode_original[:, 1]
        spectrumB = self.listMode_original[:, 2] + self.listMode_original[:, 3]
        ratioA = (self.listMode_original[spectrumA != 0, 0] - self.listMode_original[spectrumA != 0, 1]) / spectrumA[
            spectrumA != 0]
        ratioB = (self.listMode_original[spectrumB != 0, 3] - self.listMode_original[spectrumB != 0, 2]) / spectrumB[
            spectrumB != 0]

        nBins = int(750)
        threshold = 0.99
        [n, bins, patches] = ax_ratioA.hist(ratioA, nBins, [-threshold, threshold])
        peak_amplitude = np.digitize(self.peakMatrix[:self.number_of_crystals_per_module], bins, right=True)
        ax_ratioA.plot(self.peakMatrix[:self.number_of_crystals_per_module], n[peak_amplitude - 1], '.', markersize=1)
        [n, bins, patches] = ax_ratioB.hist(ratioB, nBins, [-threshold, threshold])
        peak_amplitude = np.digitize(
            self.peakMatrix[self.number_of_crystals_per_module: self.number_of_crystals_per_module * 2], bins,
            right=True)
        ax_ratioB.plot(self.peakMatrix[self.number_of_crystals_per_module: self.number_of_crystals_per_module * 2],
                       n[peak_amplitude - 1], '.', markersize=1)
        [n, bins, patches] = ax_spectrumA.hist(spectrumA, nBins, [0, 8096 / 4])
        [n, bins, patches] = ax_spectrumB.hist(spectrumB, nBins, [0, 8096 / 4])

        # HistogrM 2D
        # Z_A, X, Y = np.histogram2d(ratioA, spectrumA, 750, range=[[-0.95, 0.95], [200, 6000]])
        a = self.listMode_original[spectrumA != 0, 0]
        b = self.listMode_original[spectrumA != 0, 1]
        # spectrumA = spectrumA[spectrumA != 0]
        # spectrumB = spectrumB[spectrumB != 0]
        # ratioA= ratioA[spectrumB != 0]
        # ratioB= ratioB[spectrumA != 0]
        # Z_A, X, Y = np.histogram2d(ratioA, spectrumA, 750, range=[[np.nanmin(ratioA), np.nanmax(ratioB)], [np.nanmin(spectrumA), np.nanmax(spectrumA)]])
        Z_A, X, Y = np.histogram2d(spectrumA, spectrumB , 200, range=[[-threshold, threshold], [0, 8096 / 4]])
        Z_B, X, Y = np.histogram2d(ratioA, b, 750, range=[[-threshold, threshold], [0, 8096 / 4]])

        ax_histogram2d_A.matshow(Z_A, cmap="jet")
        ax_histogram2d_B.matshow(Z_B, cmap="jet")

        ### MOTORS
        u, indices = np.unique(self.listMode_original[:, 5], return_index=True)
        u_bot, indices_bot = np.unique(self.listMode_original[:, 4], return_index=True)

        ax_top.hist(self.listMode_original[:, 4], u_bot)
        ax_top.set_title("TOP")
        ax_bot.hist(self.listMode_original[:, 5], u)
        ax_bot.set_title("BOT")

        ### AS

        u_a1, indices_a1 = np.unique(self.listMode_original[:, 0], return_index=True)
        u_a2, indices_a2 = np.unique(self.listMode_original[:, 1], return_index=True)
        u_a3, indices_a3 = np.unique(self.listMode_original[:, 2], return_index=True)
        u_a4, indices_a4 = np.unique(self.listMode_original[:, 3], return_index=True)
        ax_a1.hist(self.listMode_original[:, 0], u_a1)
        ax_a1.set_title("A1")
        ax_a2.hist(self.listMode_original[:, 1], u_a2)
        ax_a2.set_title("A2")
        ax_a3.hist(self.listMode_original[:, 2], u_a3)
        ax_a3.set_title("A3")
        ax_a4.hist(self.listMode_original[:, 3], u_a4)
        ax_a4.set_title("A4")

    def hist_per_turn_original(self):
        time_indexes = self.acquisitionInfo["Turn end index original data"]
        time_indexes.insert(0, 0)

    def cumulative_odd_even_turn(self):
        time_indexes = self.acquisitionInfo["Turn end index original data"]
        listMode_turn_par = None
        listMode_turn_impar = None
        bot = self.listMode_original[:, 5]
        top = self.listMode_original[:, 4]
        # top_par = top[(self.listMode[:
        self.events_per_turn = np.zeros((len(time_indexes)))
        listMode_turn_impar_temp = None
        listMode_turn_par_temp = None
        for t in range(0, len(time_indexes) - 2, 2):

            listMode_turn_impar_temp = self.listMode_original[time_indexes[t]:time_indexes[t + 1]]
            listMode_turn_par_temp = self.listMode_original[time_indexes[t + 1]:time_indexes[t + 2]]
            if listMode_turn_par is None:
                listMode_turn_par = listMode_turn_par_temp
                listMode_turn_impar = listMode_turn_impar_temp
            else:
                listMode_turn_par = np.vstack((listMode_turn_par, listMode_turn_par_temp))
                listMode_turn_impar = np.vstack((listMode_turn_impar, listMode_turn_impar_temp))
            # print("turn: {} counts: {}".format(t, len(listMode_turn_impar_temp)))
            self.events_per_turn[t] = len(listMode_turn_impar_temp)
            self.events_per_turn[t + 1] = len(listMode_turn_par_temp)
            # print("turn: {} counts: {}".format(t + 1, len(listMode_turn_par_temp)))

        return listMode_turn_impar, listMode_turn_par, top, bot

    def events_per_turn_hist(self):
        f_events, (ax_events_per_turn) = plt.subplots(1, 1)
        ax_events_per_turn.bar(np.arange(0, len(self.events_per_turn)), self.events_per_turn)

    def sinogram_bot_top_original(self, i=0):
        # plt.figure(i)
        time_indexes = self.acquisitionInfo["Turn end index original data"]
        # self.listMode_original = self.listMode_original[self.listMode_original[:, 5] % 2 == 0]
        listMode_original_temp1 = self.listMode_original[time_indexes[0]:time_indexes[3]]
        listMode_original_temp2 = self.listMode_original[time_indexes[-4]:time_indexes[-1]]
        # plt.figure(14)
        # sinotemp1 = plt.hist2d(listMode_original_temp1[:,5], listMode_original_temp1[:,4], bins=[len(np.unique(listMode_original_temp1[:,5])),
        #                                                    int(self.header[5] * self.header[4] / self.header[3])])
        #
        # plt.figure(15)
        # sinotemp2 = plt.hist2d(listMode_original_temp2[:, 5], listMode_original_temp2[:, 4],
        #                        bins=[len(np.unique(listMode_original_temp2[:, 5])),
        #                              int(self.header[5] * self.header[4] / self.header[3])])
        # self.listMode_original = np.vstack((listMode_original_temp1, listMode_original_temp2))
        # self.listMode_original = self.listMode_original[np.abs(self.listMode_original[:,5]) < 3200,: ]
        # self.listMode_original[self.listMode_original[:, 5] % 128 == 64,4] += 4
        bot = self.listMode_original[:, 5]
        top = self.listMode_original[:, 4]
        listMode_original = self.listMode_original[self.listMode_original[:, 5] % 128 == 0]
        # self.listMode_original[self.listMode_original[:, 5] % 128 == 0,4] += 4
        bot_e = listMode_original[:, 5]
        top_e = listMode_original[:, 4]
        listMode_original = self.listMode_original[self.listMode_original[:, 5] % 128 == 64]
        bot_o = listMode_original[:, 5]
        top_o = listMode_original[:, 4]
        fig = plt.figure(0)

        self.sinogram = plt.hist2d(bot, top, bins=[200, int(self.header[5] * self.header[4] / self.header[3])])
        self.sinogram = plt.hist2d(bot, top, bins=[200, 1000])
        # hist_1 = plt.hist2d(bot_o, top_o, bins=[100, np.unique(top)],  cmap="jet")
        # plt.figure(2)
        # hist_2 = plt.hist2d(bot_e, top_e, bins=[100, np.unique(top)], cmap="jet")
        # plt.figure(3)
        # plt.bar(hist_1[2][:-1], hist_1[0][5])
        # plt.bar(hist_2[2][:-1], hist_2[0][5])
        from EasyPETLinkInitializer.SimulationData import SimulationStatusData
        result = self.sinogram[0]
        # result = ndimage.median_filter(self.sinogram[0], size=4)
        gaussian_r = np.zeros((result.shape[0], 4))

        for i in range(result.shape[0]):
            # plt.plot(self.sinogram[2][:-1], result[i,:])
            try:
                max_value = result[i, :].max()
                s_y = self.sinogram[2][:-1]
                mean_posix = s_y[result[i, :] == result[i, :].max()]

                p0 = np.array([300, mean_posix[0], 170, 100])
                popt, pcov = curve_fit(SimulationStatusData.gaussian_fit, s_y, result[i, :],
                                       method='lm', p0=p0)
                print(max_value)
                print((1 / popt[2] * np.sqrt(2 * np.pi)))
                print("_______")
                # popt, pcov = curve_fit(SimulationStatusData.gaussian_fit, self.sinogram[2][:-1], result[i,:],method='lm')

                # popt, pcov = curve_fit(SimulationStatusData.gaussian_fit, self.sinogram[2][:-1], result[i,:], method='lm', p0=[1, mean_posix[0],result[i,:].max()])

                # popt = t.fit(result[i,:], loc=mean_posix[0], scale=result[i,:].max())

            except RuntimeError:
                popt = np.ones((4)) * 1000
            # gaussian_r[i] = popt
            gaussian_r[i] = np.array(popt)

        # gaussian_r = np.array(gaussian_r)
        print(gaussian_r)
        dif = np.diff(gaussian_r[:, 1])

        #
        print("Max: {}".format(gaussian_r[0].max()))
        print("Min: {}".format(gaussian_r[0].min()))

        plt.plot(self.sinogram[1][:-1], gaussian_r[:, 1], "go")
        # plt.bar(self.sinogram[2][:-1], result[0, :])

        for i in range(len(gaussian_r)):
            fig_V = plt.figure(i + 1)
            plt.plot(self.sinogram[2][:-1], SimulationStatusData.gaussian_fit(self.sinogram[2][:-1], *gaussian_r[i]),
                     "r--")
            plt.plot(self.sinogram[2][:-1], self.sinogram[0][i, :])
            plt.savefig(os.path.join(self.path_data_validation, "Topvarr{}".format(i)))
            plt.close(fig_V)

        self.listMode_original[self.listMode_original[:, 5] % 128 == 64, 4] -= np.mean(
            gaussian_r[::2, 1] - gaussian_r[1::2, 1])
        bot = self.listMode_original[:, 5]
        top = self.listMode_original[:, 4]
        plt.figure()
        self.sinogram = plt.hist2d(bot, top, bins=[200, 1000])
        # print(bot.min())
        # print(bot.max())
        # print(top.min())
        # print(top.max())
        print(len(bot))
        print(len(np.unique(bot)))
        print(len(top[(1 + bot / 5) % 2 == 0]))
        top_odd = top[(bot / 5) % 2 == 0]
        bot_odd = bot[(bot / 5) % 2 == 0]
        print(len(np.unique(bot_odd)))

        # top[bot/64 % 2 == 0] -= 32252
        n, bins = np.histogram(bot, len(np.unique(bot)))
        # plt.plot(np.diff(np.unique(bot)),n[:-1], ".")
        # plt.figure(16)
        # plt.plot(np.unique(bot)[:-1],np.diff(np.unique(bot)), ".")
        # plt.figure(3)
        # self.sinogram = plt.hist2d(bot_odd, top_odd, bins=[len(np.unique(bot_odd)),
        #                                            int(self.header[5] * self.header[4] / self.header[3])])
        # result = ndimage.median_filter(self.sinogram[0], size=3)
        #
        # bot_t = np.repeat(np.unique(bot_odd), len(self.sinogram[2])-1)
        # top_t = np.tile(self.sinogram[2][:-1], len(self.sinogram[1])-1)
        #
        # hist_shap = np.reshape(self.sinogram[0], self.sinogram[0].shape[0]*self.sinogram[0].shape[1])
        # hist_shap = np.reshape(result, self.sinogram[0].shape[0]*self.sinogram[0].shape[1])
        #
        # bot_t = bot_t[hist_shap!=0]
        # top_t = top_t[hist_shap!=0]
        # bot_t = np.repeat(bot_t,hist_shap[hist_shap!=0].astype(int))
        # top_t = np.repeat(top_t,hist_shap[hist_shap!=0].astype(int))
        #
        # # plt.figure(4)
        #
        # # t = np.repeat(self.sinogram[1], self.sinogram[])
        # # s, m = DataTest.reject_outliers(top_t)
        # # top_odd = top_odd[s < m]
        # # bot_odd = bot_odd[s < m][4, top_t.max()-top_t.min(),0,top_t/2]
        # # plt.scatter(bot_t, top_t)
        # # plt.figure(5)
        # # plt.scatter(np.radians(bot_t*self.header[2] / self.header[1]), np.radians(self.header[5]/2 -top_t * self.header[4] / self.header[3]))
        # # p0 =np.array([4000,bot_t.max()-bot_t.min(),0,top_t.max()/2])
        # # popt, pcov = curve_fit(DataTest.sin_fit, np.radians(bot_t*self.header[2] / self.header[1]), np.radians(self.header[5]/2 -top_t * self.header[4] / self.header[3]))
        # # # popt, pcov = curve_fit(DataTest.sin_fit, bot_t, top_t)
        # # #
        # # plt.plot(np.radians(bot_t*self.header[2] / self.header[1]), DataTest.sin_fit(np.radians(bot_t*self.header[2] / self.header[1]), *popt), 'r--')
        #
        # # popt, pcov = curve_fit(DataTest.x_5, bot_t, top_t)
        # # plt.plot(bot_t, DataTest.x_5(bot_t, *popt), 'g.')
        # #
        # # popt, pcov = curve_fit(DataTest.custom, bot_t, top_t)
        # # plt.plot(bot_t, DataTest.custom(bot_t, *popt), 'y-')
        # plt.figure(7)
        #
        # top_even = top[(bot / 10) % 2 != 0]
        # bot_even = bot[(bot / 10) % 2 != 0]
        # print(len(np.unique(bot_even)))
        #
        # self.sinogram_l = plt.hist2d(bot_even, top_even, bins=[len(np.unique(bot_even)),
        #                                                    int(self.header[5] * self.header[4] / self.header[3])])

        # plt.imshow((self.sinogram[0]-self.sinogram_l[0]).T,  cmap="hot", aspect="auto", origin="lower")
        # from scipy import ndimage, misc
        # plt.figure(10)
        # result = ndimage.median_filter(self.sinogram_l[0], size=1)
        # plt.imshow(result.T, cmap="hot", aspect="auto", origin="lower")
        #
        # plt.figure(11)
        # result = ndimage.median_filter(self.sinogram[0], size=1)
        # plt.imshow(result.T, cmap="hot", aspect="auto", origin="lower")
        # # plt.plot(bot_even, np.ones(len(bot_even))*np.max(top_even)/2)

    @staticmethod
    def reject_outliers(data, m=1.):
        d = np.abs(data - np.median(data, axis=1))
        mdev = np.median(d)
        s = d / mdev if mdev else 0.
        return s, m

    def sinogram_odd_even_original(self, listMode_turn_impar, listMode_turn_par, bot, top):

        plt.figure(5)
        self.sinogram_par = plt.hist2d(listMode_turn_par[:, 5], listMode_turn_par[:, 4], bins=[len(np.unique(bot)),
                                                                                               int(self.header[5] *
                                                                                                   self.header[4] /
                                                                                                   self.header[3])],
                                       cmap="jet")
        plt.figure(6)
        self.sinogram_impar = plt.hist2d(listMode_turn_impar[:, 5], listMode_turn_impar[:, 4],
                                         bins=[len(np.unique(bot)),
                                               int(self.header[5] * self.header[4] / self.header[3])],
                                         cmap="jet")

    def angle_correction(self):

        listmode = self.listMode
        listmode = listmode[listmode[:, 0] > 400]
        listmode = listmode[listmode[:, 0] < 550]
        listmode = listmode[listmode[:, 1] > 400]
        listmode = listmode[listmode[:, 1] < 550]
        # listmode = listmode[(listmode[:, 2] % 6 > 1) | (listmode[:, 2] % 6 < 3) , :]
        # listmode = listmode[listmode[:, 3] % 6 == 2, :]
        # #
        # listmode = listmode[
        #     np.ceil(np.array(np.round((listmode[:, 4]), 4)) / (self.header[3] / 32)) % 128 - 56 == 64]
        id_ = np.degrees(np.arctan((((listmode[:, 3] - 1) % 2) * 2.35 - 1.175) / 60))
        bot = listmode[:, 4]
        top = -listmode[:, 5] + id_

        # - np.random.randint(0,8,len(bot))*0.9

        s = np.sin(np.radians(top)) * (60 / 2)
        # s[(listmode[:,3] % 2)-(listmode[:,2] % 2) == 0] += 1.175
        # s[(listmode[:,3] % 2 + 1)-(listmode[:,2] % 2 + 1) == 0] -= 1.175
        # s = np.abs(s)
        phi = 90 + bot + top
        # phi = bot + top
        phi = phi % 360
        # top = s
        # bot = phi
        s = np.round(s, 6)
        phi = np.round(phi, 6)

        # self.sinogram = plt.hist2d(bot, top, bins=[len(np.unique(bot)),
        #                                            int(header[5] * header[4] / header[3])])
        plt.figure(1)
        # self.sinogram = plt.hist2d(phi, s, bins=[int(len(np.unique(bot))), int(len(np.unique(top)))])
        # plt.show()
        # plt.figure()
        #
        # plt.hist2d(phi,s, bins=[int(len(np.unique(bot))), int(len(np.unique(top)))])
        #
        # plt.figure()
        self.sinogram = plt.hist2d(bot, top, bins=[int(len(np.unique(bot))), int(len(np.unique(top)))])
        # plt.show()

        val = filters.threshold_otsu(self.sinogram[0])
        mask = self.sinogram[0] >= val
        # print(mask)

        foot_print = np.array([[0, 0, 0.2, 0, 0],
                               [0, 0, 0.5, 0, 0],
                               [0, 0, 1.0, 0, 0],
                               [0, 0, 0.5, 0, 0],
                               [0, 0, 0.2, 0, 0]])
        # result = ndimage.median_filter(self.sinogram[0], size=4)
        # result = ndimage.median_filter(mask*self.sinogram[0] , size=1, footprint=foot_print)
        result = ndimage.median_filter(self.sinogram[0], size=1)
        plt.figure(3)
        plt.imshow(result)
        # result =result.T
        # result = self.sinogram[0] *mask# footprint=np.array([[0,0.8,0],
        # footprint=np.array([[0,*0.8,0],
        # [0,1,0],
        #  [0,0.8,0]])

        # result[result[:,:]==np.max(result, axis=1)] =0
        bot_t = np.repeat(self.sinogram[1][:-1], len(self.sinogram[2]) - 1)
        top_t = np.tile(self.sinogram[2][:-1], len(self.sinogram[1]) - 1)
        gaussian_r = [None] * result.shape[0]

        for i in range(result.shape[0]):
            # plt.plot(self.sinogram[2][:-1], result[i,:])
            try:
                max_value = result[i, :].max()
                s_y = self.sinogram[2][:-1]
                mean_posix = s_y[result[i, :] == result[i, :].max()]

                p0 = np.array([int(0), mean_posix[0], 10, int(0)])
                popt, pcov = curve_fit(SimulationStatusData.gaussian_fit, self.sinogram[2][:-1], result[i, :],
                                       method='lm', p0=p0)
                print(max_value)
                print((1 / popt[2] * np.sqrt(2 * np.pi)))
                print("_______")
                # popt, pcov = curve_fit(SimulationStatusData.gaussian_fit, self.sinogram[2][:-1], result[i,:],method='lm')

                # popt, pcov = curve_fit(SimulationStatusData.gaussian_fit, self.sinogram[2][:-1], result[i,:], method='lm', p0=[1, mean_posix[0],result[i,:].max()])

                # popt = t.fit(result[i,:], loc=mean_posix[0], scale=result[i,:].max())

            except RuntimeError:
                popt = np.ones((4)) * 1000
            gaussian_r[i] = popt

        gaussian_r = np.array(gaussian_r)
        print(gaussian_r)
        # plt.figure(4)
        # # plt.plot(self.sinogram[2][:-1], self.sinogram[0][0,:])
        # plt.bar(self.sinogram[2][:-1], result[0, :])
        # plt.plot(self.sinogram[2][:-1], SimulationStatusData.gaussian_fit(self.sinogram[2][:-1], *gaussian_r[0]), "r--")

        top_fit = gaussian_r[:, 1]
        phi_fit = 90 + self.sinogram[1][:-1] + top_fit
        phi_fit = phi_fit % 360
        gaussian_r = np.sin(np.radians(gaussian_r)) * (60 / 2)
        # self.sinogram[2][:-1] = self.sinogram[2][gaussian_r[:,1] < np.abs(self.sinogram[2][:-1]).max()]
        # y_sino = self.sinogram[1][:-1]
        # y_sino = y_sino[gaussian_r[:,1] < np.abs(self.sinogram[2][:-1]).max()]
        # self.sinogram[1] = y_sino
        plt.figure(5)
        self.sinogram = plt.hist2d(phi, s, bins=[int(len(np.unique(s))), int(len(np.unique(s)))])
        bot_t = np.repeat(self.sinogram[1][:-1], len(self.sinogram[2]) - 1)
        top_t = np.tile(self.sinogram[2][:-1], len(self.sinogram[1]) - 1)

        index_gaussian = np.where(np.abs(gaussian_r[:, 1]) < np.abs(self.sinogram[2][:-1]).max())
        gaussian_r = gaussian_r[np.abs(gaussian_r[:, 1]) < np.abs(self.sinogram[2][:-1]).max()]
        phi_fit = phi_fit[index_gaussian]
        # result[np.min(np.abs(gaussian_r[:,1]-top_t), axis=1)]

        # hist_shap = np.reshape(self.sinogram[0], self.sinogram[0].shape[0] * self.sinogram[0].shape[1])
        hist_shap = np.reshape(self.sinogram[0], self.sinogram[0].shape[0] * self.sinogram[0].shape[1])

        bot_t = bot_t[hist_shap != 0]
        top_t = top_t[hist_shap != 0]
        bot_t = np.repeat(bot_t, hist_shap[hist_shap != 0].astype(int))
        top_t = np.repeat(top_t, hist_shap[hist_shap != 0].astype(int))
        # plt.figure(2)
        # plt.scatter(bot_t, top_t)
        # popt, pcov = curve_fit(SimulationStatusData.sin_fit,self.sinogram[1][:-1], gaussian_r[:,1],
        #                        p0=np.array([np.abs((gaussian_r[:,1].max()-gaussian_r[:,1].min())/2),np.pi/360, 0]))

        popt, pcov = curve_fit(SimulationStatusData.sin_fit, phi_fit, gaussian_r[:, 1],
                               p0=np.array(
                                   [np.abs((gaussian_r[:, 1].max() - gaussian_r[:, 1].min()) / 2), np.pi / 180, 0, 50]))
        print(np.sqrt(np.diag(pcov)))

        amplitude = SimulationStatusData.sin_fit(bot_t, *popt)
        amplitude_angle = np.degrees(np.arcsin(amplitude / 30))
        # np.sin(np.radians(top)) * (60 / 2)
        print("Angle to correct: {}".format((amplitude_angle.max() - np.abs(amplitude_angle.min())) / 2))
        # popt, pcov = curve_fit(DataTest.sin_fit, bot_t, top_t)
        #
        plt.plot(phi_fit, gaussian_r[:, 1], "go")
        # plt.plot(self.sinogram[1][:-1], np.zeros(len(gaussian_r[:,1])), "y--")
        plt.plot(bot_t,
                 SimulationStatusData.sin_fit(bot_t, *popt), 'r--')
        print("Angle to correct: {}".format(
            np.degrees(np.arcsin((gaussian_r[:, 1].max() + gaussian_r[:, 1].min()) / 30)) / 2))
        print(np.degrees(np.arcsin(gaussian_r[:, 1].max() / 30)))
        print(np.degrees(np.arcsin(gaussian_r[:, 1].min() / 30)))
        plt.show()

    def return_gaussian_vector(self, result, phi_fit):
        result = result[0]
        gaussian_r = [None] * result.shape[0]
        for i in range(result.shape[0]):
            # plt.plot(self.sinogram[2][:-1], result[i,:])
            try:
                max_value = result[i, :].max()
                s_y = self.sinogram[2][:-1]
                mean_posix = s_y[result[i, :] == result[i, :].max()]

                p0 = np.array([result[i, :].max(), mean_posix[0], 2.5, int(0)])
                popt, pcov = curve_fit(SimulationStatusData.gaussian_fit, self.sinogram[2][:-1], result[i, :],
                                       method='lm', p0=p0)
                # print(max_value)
                # print((1 / popt[2] * np.sqrt(2 * np.pi)))
                print("_______")
                residuals = result[i, :] - SimulationStatusData.gaussian_fit(self.sinogram[2][:-1], *popt)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((result[i, :] - np.mean(result[i, :])) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                print(r_squared)
                # popt, pcov = curve_fit(SimulationStatusData.gaussian_fit, self.sinogram[2][:-1], result[i,:],method='lm')

                # popt, pcov = curve_fit(SimulationStatusData.gaussian_fit, self.sinogram[2][:-1], result[i,:], method='lm', p0=[1, mean_posix[0],result[i,:].max()])

                # popt = t.fit(result[i,:], loc=mean_posix[0], scale=result[i,:].max())

            except RuntimeError:
                popt = np.ones((4)) * 1000
            gaussian_r[i] = popt

        gaussian_r = np.array(gaussian_r)

        bot_t = np.repeat(self.sinogram[1][:-1], len(self.sinogram[2]) - 1)
        top_t = np.tile(self.sinogram[2][:-1], len(self.sinogram[1]) - 1)

        # index_gaussian = np.where(np.abs(gaussian_r[:, 1]) < np.abs(self.sinogram[2][:-1]).max())
        # gaussian_r = gaussian_r[np.abs(gaussian_r[:, 1]) < np.abs(self.sinogram[2][:-1]).max()]
        # phi_fit = phi_fit[index_gaussian]
        # result[np.min(np.abs(gaussian_r[:,1]-top_t), axis=1)]

        # hist_shap = np.reshape(self.sinogram[0], self.sinogram[0].shape[0] * self.sinogram[0].shape[1])
        hist_shap = np.reshape(self.sinogram[0], self.sinogram[0].shape[0] * self.sinogram[0].shape[1])

        bot_t = bot_t[hist_shap != 0]
        top_t = top_t[hist_shap != 0]
        bot_t = np.repeat(bot_t, hist_shap[hist_shap != 0].astype(int))
        top_t = np.repeat(top_t, hist_shap[hist_shap != 0].astype(int))
        # plt.figure(2)
        # plt.scatter(bot_t, top_t)
        # popt, pcov = curve_fit(SimulationStatusData.sin_fit,self.sinogram[1][:-1], gaussian_r[:,1],
        #                        p0=np.array([np.abs((gaussian_r[:,1].max()-gaussian_r[:,1].min())/2),np.pi/360, 0]))
        try:
            popt, pcov = curve_fit(SimulationStatusData.sin_fit, self.sinogram[1][:-1], gaussian_r[:, 1],
                                   p0=np.array(
                                       [np.abs((gaussian_r[:, 1].max() - gaussian_r[:, 1].min()) / 2), np.pi / 180, 0,
                                        0]))
        except RuntimeError:
            popt = np.ones((4)) * 1000
        # print(np.sqrt(np.diag(pcov)))

        amplitude = SimulationStatusData.sin_fit(bot_t, *popt)

        return phi_fit, bot_t, gaussian_r, popt, amplitude

    def sinogram_from_parametric(self):

        # matplotlib.rc('text.latex', preamble=r'\usepackage{cmbright}')
        # sens.plt_configure()
        # plt.style.use('ggplot')

        time_indexes = self.acquisitionInfo["Turn end index"]
        listmode = self.listMode
        # listmode = listmode[time_indexes[10]:time_indexes[170]]
        print("Tempo Frame:{}".format(listmode[-1, 6] - listmode[0, 6]))
        print("Tempo INICIO:{}".format(listmode[0, 6]))
        print("Tempo FIM:{}".format(listmode[-1, 6]))
        print(len(time_indexes))
        listMode_turn_par = None
        angle_top = 0
        listmode[:, 5] = (listmode[:, 5] +angle_top)
        # listmode[:, 4] = -listmode[:, 4]
                          # - np.degrees(np.arctan(angle_top / 60)))
        print("angulo:{}".format(np.degrees(np.arctan(angle_top / 60))))
        listmode = listmode[listmode[:, 0] > 400]
        listmode = listmode[listmode[:, 0] < 600]
        listmode = listmode[listmode[:, 1] > 400]
        listmode = listmode[listmode[:, 1] < 600]
        a = np.copy(listmode[:, 2])
        b = np.copy(listmode[:, 3])
        ea = np.copy(listmode[:, 0])
        eb = np.copy(listmode[:, 1])
        listmode[:, 2] = b
        listmode[:, 3] = a
        listmode[:, 0] = eb
        listmode[:, 1] = ea
        listmode= listmode
        dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        plt.figure()
        plt.hist2d(listmode[:,4], listmode[:,5], bins=(len(np.unique(listmode[:,4])), len(np.unique(listmode[:,5]))))

        for i in range(1):

            #     np.ceil(np.array(np.round((listmode[:, 4]), 4)) / (self.header[3] / 32)) % 128 - 56 == 64]
            listmode[:, 4] = np.round(listmode[:, 4], 5)
            # listmode[:, 5] = -np.round(listmode[:, 5], 5)
            # listmode[:, 4] *= -1
            # listmode[:, 5] *= -1

            # listmode[:,5] -= 0.3
            # listmode[:,4] += 0.3
            path_matrix = os.path.join(os.path.dirname(dirname), "system_configurations", "x_{}__y_{}".format(
                                                                self.crystals_geometry[0], self.crystals_geometry[1]))
            print(path_matrix)
            MatrixCorrection = MatrixGeometryCorrection(operation='r',
                                                        file_path=path_matrix)
            geometry_file = MatrixCorrection.coordinates
            crystals_geometry = self.crystals_geometry
            height = self.systemConfigurations_info["crystal_pitch_x"]
            crystal_width = self.systemConfigurations_info["crystal_pitch_y"]
            reflector_y = 2 * self.systemConfigurations_info["reflector_interior_A_y"]
            # reflector_y = self.systemConfigurations_info["reflector_interior_A_y"]#simula
            # geometry_file[:crystals_geometry[0] * crystals_geometry[1], 1] += 1
            # geometry_file[:, 1] = np.tile(np.round(np.arange(0,crystals_geometry[1]-1,0.8)-2.4,3),crystals_geometry[0])
            geometry_file[:, 1] = np.tile(np.round(np.arange(0, crystals_geometry[1] * crystal_width + 2 * reflector_y,
                                                             crystal_width + reflector_y) - (
                                                               crystal_width + reflector_y) *
                                                   (crystals_geometry[1] - 1) / 2, 3), crystals_geometry[0] * 2)


            geometry_file[crystals_geometry[0] * crystals_geometry[1]:, 1] *= -1
            # g
            # if not self.acquisitionInfo["Type of subject"] == "Simulation":

            # geometry_file[crystals_geometry[0] * crystals_geometry[1]:, 2] = -geometry_file[crystals_geometry[0] * crystals_geometry[1]:, 2]+np.max(geometry_file[crystals_geometry[0] * crystals_geometry[1]:,2 ])
            # geometry_file[:, 1] *= -1
            # geometry_file[:crystals_geometry[0] * crystals_geometry[1], 1] += 0.3
            # geometry_file[crystals_geometry[0] * crystals_geometry[1]:, 1] += 0.3
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
                                                  simulation_files=False, transform_into_positive=False)
            xi = parametric.xi
            yi = parametric.yi
            zi = parametric.zi

            xf = parametric.xf
            yf = parametric.yf
            zf = parametric.zf
            # p1 = np.array([xi, yi])
            p1 = np.column_stack((xi, yi))
            p2 = np.column_stack((xf, yf))
            # p2 = np.array([xf, yf])
            p3 = np.copy(p1) * 0

            # phi = phi%360
            v1 = p2 - p3
            v2 = p1 - p3
            n1 = (np.sqrt(v1[:, 0] ** 2 + v1[:, 1] ** 2))
            n2 = np.sqrt(v2[:, 0] ** 2 + v2[:, 1] ** 2)

            abcissa = (xf - xi)
            # declive = (yf - yi)[abcissa != 0] / abcissa[abcissa!=0]
            declive = (yf - yi) / abcissa
            phi = np.degrees(np.arctan(declive)) + 90
            phi[np.sign(xi - xf) == 1] += 180
            # phi[np.sign(xi-xf) == 1] *= -1

            # phi = phi[np.sign(xi-xf) == 1]
            # phi[phi==360] = 0
            # declive=declive[phi<=180]

            # phi[abcissa == 0] = 90
            s = (np.cross(v1, v2) / n1)
            s = np.cross(v1, v2) /n1

            print(s.max() + s.min())
            # s += 0.18
            # phi and s for z under 6 mm
            # phi = phi[np.abs(zi-zf) < 6]
            # s = s[np.abs(zi-zf) < 6]
            #
            # zi = zi[np.abs(zi-zf) < 6]
            # # zf = zf[np.abs(zi-zf) < 6]
            # #
            # mask =(zi>6) & (zi<30)
            # phi = phi[mask]
            # s = s[mask]
            # plt.figure()
            # plt.hist(phi, 500)
            # plt.figure()
            # plt.hist(s, 500)
            # plt.title("S")
            plt.figure()
            from skimage.transform import iradon_sart
            bins = 200
            # self.sinogram = plt.hist2d(phi, s, bins=[bins, 200], range=[[0, 360], [-30, 30]], cmap='jet', )
            bins_y = int(len(np.unique(s)) / 2**6)
            bins_y = int(200)
            # bins_y = int(self.header[5]/(self.header[3]/self.header[4])/2)
            print(bins_y)
            self.sinogram = plt.hist2d(phi, s, bins=[bins, bins_y ], range=[[0, 360], [s.min(), s.max()]], cmap='jet', )
            # save in outputs s and phi
            np.save(os.path.join(self.path_data_validation, "s"), s)
            np.save(os.path.join(self.path_data_validation, "phi"), phi)
            np.save(os.path.join(self.path_data_validation, "top"), listmode[:, 5])
            np.save(os.path.join(self.path_data_validation, "bot"),listmode[:, 4])
            print(self.path_data_validation)
            # self.sinogram = np.array(self.sinogram[0:3])
            # self.sinogram[0] = self.sinogram[0][1::2,:]
            # self.sinogram[1] = self.sinogram[1][::2]
            # self.sinogram[2] = self.sinogram[2][::2]
            [phi_fit, bot_t, gaussian_r, popt, amplitude] = self.return_gaussian_vector(self.sinogram, phi)

            # theta = np.linspace(0, 360, num=len(self.sinogram[1])-1)  # len(phi_AngVector)
            theta = self.sinogram[1][:-1]  # len(phi_AngVector)
            # plt.plot(self.sinogram[1][:-1], gaussian_r[:, 1], "o")
            # # plt.plot(self.sinogram[1][:-1], gaussian_r[:, 1], "o")
            # plt.plot(self.sinogram[1][:-1],
            #      SimulationStatusData.sin_fit(self.sinogram[1][:-1], *popt), 'r--')

            plt.xlabel("phi (º)")
            plt.ylabel("s (mm)")
            # plt.plot(self.sinogram[1][:-1], np.zeros(len(gaussian_r[:,1])), "y--")
            # plt.plot(theta,
            #          SimulationStatusData.sin_fit(theta, *popt), 'r--')
            plt.figure(3)
            plt.plot(self.sinogram[1][:-1], gaussian_r[:, 2])
            plt.title("$/sigma$")
            plt.xlabel("phi (º)")
            plt.ylabel("s (mm)")


            self.sinogram = plt.hist2d(phi, s, bins=[bins, bins_y], range=[[0, 360], [-30, 30]], cmap='jet', )
            [phi_fit, bot_t, gaussian_r, popt, amplitude] = self.return_gaussian_vector(self.sinogram, phi)

            print(np.diff(gaussian_r[:, 1]))
            print(gaussian_r[:, 1].max()+ gaussian_r[:, 1].min())

            print("Angle to correct: {}".format(
                np.degrees(np.arcsin((gaussian_r[:, 1].max() + gaussian_r[:, 1].min()) / 30)) / 2))
            print("Angle to correct: {}".format(
                gaussian_r[:, 1].max() + gaussian_r[:, 1].min()))
            result = ndimage.median_filter(self.sinogram[0], size=[1, 3])
            result[result>0] = 1
            # result[result>0] = 1
            result = self.sinogram[0] * result
            # FBP_volume = iradon(self.sinogram[0].T, theta=theta,
            #                     circle=False,
            #                     interpolation='cubic')
            FBP_volume=iradon_sart(result.T, theta=theta)
            plt.figure(4)
            extent = np.min(s), np.max(s), np.min(s), np.max(s)
            extent = -30, 30, -30, 30
            FBP_volume = ndimage.median_filter(FBP_volume, size=3)
            plt.imshow(FBP_volume, cmap="jet", extent=extent, alpha=.9)
            plt.title("FBP")
            plt.xlabel("mm")
            plt.ylabel("mm")
        plt.show()
        # if (i +1)% 2 ==0:

        # self.sinogram.cmax(self.sinogram[0].max()*0.5)

    @staticmethod
    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    @staticmethod
    def angle_between(v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = DataTest.unit_vector(v1)
        v2_u = DataTest.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def sinogram_bot_top(self):
        plt.figure(2)

        # print("Scan time: {} s".format(self.scan_time))
        # energy_window = [300,600]
        # index_ea = np.where(
        #     (self.listMode[:, 0] < energy_window[0]) | (self.listMode[:, 1] < energy_window[0]))
        # index_eb = np.where(
        #     (self.listMode[:, 0] > energy_window[1]) | (self.listMode[:, 1] > energy_window[1]))
        # union_indexes_intersection = np.union1d(index_ea, index_eb)
        # self.listMode = np.delete(self.listMode, union_indexes_intersection, axis=0)

        # listMode_part = np.copy(self.listMode)
        # n=16
        # angle = 0.9
        # # self.listMode = self.listMode[self.listMode[:, 2] == 25]
        # # self.listMode = self.listMode[self.listMode[:,3] == 25]
        # listMode_part[:, 5] = listMode_part[:, 5] -0.5*angle/n
        # # probability_side_A = np.histogram((reading_data[:, 2]+reading_data[:, 3])/2,64)[0]/len(reading_data)
        # # probability_side_B = np.histogram((reading_data[:, 3]),64)[0]/len(reading_data)
        # # self.reading_data[:,2] = np.random.choice(number_of_crystals[0]*number_of_crystals[1], len(self.reading_data), p=probability)+1
        # # self.reading_data[:,3] = np.random.choice(number_of_crystals[0]*number_of_crystals[1], len(self.reading_data), p=probability)+1
        # # copy_ = np.copy(decay_factor_cutted)
        # for i in range(n):
        #     # listMode_part[:, 5] = listMode_part[:, 5] + 0.9 / 8
        #     listMode_part[:, 5] = listMode_part[:, 5] + angle/ n
        #     # listMode_part[:, 5] = listMode_part[:, 5] + 0.1125
        #     # listMode_part[:, 2] = np.random.choice(64, len(listMode_part), p=probability_side_A)+1
        #     # listMode_part[:, 3] = np.random.choice(64, len(listMode_part), p=probability_side_A)+1
        #     self.listMode = np.append(self.listMode, listMode_part, axis=0)

        bot = np.round(self.listMode[:, 4], 6)

        top = self.listMode[:, 5]
        # top_par = top[(self.listMode[:, 2]%2==0) & (self.listMode[:, 3]%2==0)]
        # bot_par = bot[(self.listMode[:, 2]%2==0) & (self.listMode[:, 3]%2==0)]
        #
        time_indexes = self.acquisitionInfo["Turn end index"]
        listMode_turn_par = None
        listMode_turn_impar = None

        for t in range(0, len(time_indexes) - 2, 2):

            listMode_turn_impar_temp = self.listMode[time_indexes[t]:time_indexes[t + 1]]
            listMode_turn_par_temp = self.listMode[time_indexes[t + 1]:time_indexes[t + 2]]
            if listMode_turn_par is None:
                listMode_turn_par = listMode_turn_par_temp
                listMode_turn_impar = listMode_turn_impar_temp
            else:
                listMode_turn_par = np.vstack((listMode_turn_par, listMode_turn_par_temp))
                listMode_turn_impar = np.vstack((listMode_turn_impar, listMode_turn_impar_temp))

        top_par = top[(self.listMode[:, 2] % 2 == 0) & (self.listMode[:, 3] % 2 == 0)]
        bot_par = bot[(self.listMode[:, 2] % 2 == 0) & (self.listMode[:, 3] % 2 == 0)]

        top_impar = top[((self.listMode[:, 2] + 1) % 2 == 0) & ((self.listMode[:, 3] + 1) % 2 == 0)]
        bot_impar = bot[((self.listMode[:, 2] + 1) % 2 == 0) & ((self.listMode[:, 3] + 1) % 2 == 0)]

        s = np.sin(np.radians(top)) * (60 / 2)
        # s[(listmode[:,3] % 2)-(listmode[:,2] % 2) == 0] += 1.175
        # s[(listmode[:,3] % 2 + 1)-(listmode[:,2] % 2 + 1) == 0] -= 1.175
        # s = np.abs(s)

        phi = 90 + bot - top
        # phi = bot + top
        phi = phi % 360
        # top = s
        # self.sinogram = plt.hist2d(bot, top, bins=[len(np.unique(bot)),
        #                                            int(self.header[5] * self.header[4] / self.header[3])])

        # self.sinogram = plt.hist2d(phi, s, bins=[len(np.unique(self.listMode[:,4])),len(np.unique(self.listMode[:,5]))
        #                                           ])
        self.sinogram = plt.hist2d(phi, s,
                                   bins=[int(len(np.unique(self.listMode[:, 4]))),
                                         int(len(np.unique(self.listMode[:, 5])) / 32)
                                         ])

        result = ndimage.median_filter(self.sinogram[0], size=3)

        _bot_repeat = np.repeat(np.unique(bot), len(np.unique(top)))
        _top_repeat = np.tile(np.unique(top), len(np.unique(bot)))
        # new_list_mode = np.repeat(_bot_repeat, result.flatten().astype(int))
        plt.figure(3)
        # plt.imshow(result.T, aspect="auto")
        self.sinogram = plt.hist2d(self.listMode[:, 2], self.listMode[:, 3],
                                   bins=[64,
                                         64
                                         ])
        plt.figure(5)
        result = ndimage.median_filter(self.sinogram[0], size=3)
        plt.imshow(result.T, aspect="auto")
        new_index = (self.listMode[:, 2] - 1) * 64 + self.listMode[:, 3]
        new_listMode = np.zeros((len(self.listMode), 3))
        new_listMode[:, 0] = new_index
        new_listMode[:, 1:] = self.listMode[:, 4:6]
        output = np.histogramdd(self.listMode[:, 2:6], bins=[64, 64, 200, 200], range=None, normed=None, weights=None,
                                density=None)
        plt.imshow(np.sum(np.sum(output[0], axis=1), axis=0))
        plt.show()
        output[0][:, :, ndimage.median_filter(np.sum(np.sum(output[0], axis=1), axis=0), size=3) < 8] = 0
        # output = np.histogramdd(new_listMode, bins=[64*64,200,200], range=None, normed=None, weights=None, density=None)
        # hist_shap = output[0].flatten().astype(int)
        list_mode_recreate = np.zeros((int(np.sum(output[0])), 4))
        el = 0
        for i in range(len(output[1][0]) - 1):
            print(i)
            i_element = output[1][0][i]
            for j in range(len(output[1][1]) - 1):
                j_element = output[1][1][j]
                for k in range(len(output[1][2]) - 1):
                    k_element = output[1][2][k]
                    for t in range(len(output[1][3]) - 1):
                        t_element = output[1][3][t]

                        if output[0][i, j, k, t] != 0:

                            for v in range(int(output[0][i, j, k, t])):
                                list_mode_recreate[el + v, 0] = i_element
                                list_mode_recreate[el + v, 1] = j_element
                                list_mode_recreate[el + v, 2] = k_element
                                list_mode_recreate[el + v, 3] = t_element
                            el += int(output[0][i, j, k, t])
        # np.save("list_mode_recreate.npy", list_mode_recreate)
        # _bot_repeat = np.repeat(np.unique(bot), len(np.unique(top)))
        # _top_repeat = np.tile(np.unique(top), len(np.unique(bot)))
        print(output)
        # self.sinogram = plt.hist2d(bot_par, top_par, bins=[len(np.unique(bot)),
        #                                                        int(self.header[5] * self.header[4] / self.header[3])],cmap ="hot")

        # plt.figure(4)
        # self.sinogram = plt.hist2d(bot_impar, top_impar, bins=[len(np.unique(bot)),
        #                                            int(self.header[5] * self.header[4] / self.header[3])], cmap="jet")
        #
        # plt.figure(5)
        # self.sinogram_par = plt.hist2d(listMode_turn_par[:,4], listMode_turn_par[:,5], bins=[len(np.unique(bot)),
        #                                                    int(self.header[5] * self.header[4] / self.header[3])],
        #                            cmap="hot")
        # # plt.figure(6)
        # # self.sinogram_impar= plt.hist2d(listMode_turn_impar[:,4], listMode_turn_impar[:,5], bins=[len(np.unique(bot)),
        # #                                                        int(self.header[5] * self.header[4] / self.header[3])],
        # #                            cmap="jet")
        # plt.plot(bot, np.zeros(len(bot)))
        # from scipy import ndimage, misc
        # result = ndimage.median_filter(self.sinogram[0], size=2)
        # # plt.figure(3)
        # plt.matshow(result, origin="lower", aspect="auto")

    def sinogram_individual_crystals(self):

        f_sinogram, ax_list = plt.subplots(4, 4)
        j = 0
        k = 0
        for i in range(1, 17):
            listMode_temp = self.listMode[self.listMode[:, 3] == i]

            ax_list[j][k].hist2d(listMode_temp[:, 4], listMode_temp[:, 5], bins=[len(np.unique(bot)),
                                                                                 int(self.header[5] * self.header[4] /
                                                                                     self.header[3])], cmap="hot")
            j += 1
            if j >= 4:
                j = 0
                k += 1

        f_sinogram, ax_list_2 = plt.subplots(4, 4)
        j = 0
        k = 0
        for i in range(48, 64):
            listMode_temp = self.listMode[self.listMode[:, 3] == i]

            ax_list_2[j][k].hist2d(listMode_temp[:, 4], listMode_temp[:, 5], bins=[len(np.unique(bot)),
                                                                                   int(self.header[5] * self.header[4] /
                                                                                       self.header[3])], cmap="hot")
            j += 1
            if j >= 4:
                j = 0
                k += 1

    def bottom_motor_original(self):
        """"""

    def remove_adc_value(self, adc_cuts=[5, 5, 5, 5], signal="minor"):
        if signal == "minor":
            index_a1_100 = np.where(self.listMode_original[:, 0] < adc_cuts[0])
            index_a2_100 = np.where(self.listMode_original[:, 1] < adc_cuts[1])
            index_a3_100 = np.where(self.listMode_original[:, 2] < adc_cuts[2])
            index_a4_100 = np.where(self.listMode_original[:, 3] < adc_cuts[3])
        elif signal == "major":
            index_a1_100 = np.where(self.listMode_original[:, 0] >= adc_cuts[0])
            index_a2_100 = np.where(self.listMode_original[:, 1] >= adc_cuts[1])
            index_a3_100 = np.where(self.listMode_original[:, 2] >= adc_cuts[2])
            index_a4_100 = np.where(self.listMode_original[:, 3] >= adc_cuts[3])
        elif signal == "equal":
            index_a1_100 = np.where(self.listMode_original[:, 0] == adc_cuts[0])
            index_a2_100 = np.where(self.listMode_original[:, 1] == adc_cuts[1])
            index_a3_100 = np.where(self.listMode_original[:, 2] == adc_cuts[2])
            index_a4_100 = np.where(self.listMode_original[:, 3] == adc_cuts[3])

        indexes_intersection_A = np.union1d(index_a1_100, index_a2_100)
        indexes_intersection_B = np.union1d(index_a3_100, index_a4_100)

        union_indexes_intersection = np.union1d(indexes_intersection_A,
                                                indexes_intersection_B)

        # print('{}  Initial Number Counts:  {} '.format(file_folder, len(listMode)))
        self.listMode_original = np.delete(self.listMode_original, union_indexes_intersection, axis=0)

        print('After cutting threshold:  {} '.format(len(self.listMode_original)))

    def only_zeros_listMode(self, adc_cuts=[5, 5, 5, 5]):
        index_a1_100 = np.where(self.listMode_original[:, 0] > adc_cuts[0])
        index_a2_100 = np.where(self.listMode_original[:, 1] > adc_cuts[1])
        index_a3_100 = np.where(self.listMode_original[:, 2] > adc_cuts[2])
        index_a4_100 = np.where(self.listMode_original[:, 3] > adc_cuts[3])

        indexes_intersection_A = np.intersect1d(index_a1_100, index_a2_100)
        indexes_intersection_B = np.intersect1d(index_a3_100, index_a4_100)

        union_indexes_intersection = np.union1d(indexes_intersection_A,
                                                indexes_intersection_B)

        # print('{}  Initial Number Counts:  {} '.format(file_folder, len(listMode)))
        self.listMode_original = np.delete(self.listMode_original, union_indexes_intersection, axis=0)

    @staticmethod
    def x_3(x, a, b, c, d):
        return a * x ** 3 + b * x ** 2 + c * x + d

    @staticmethod
    def x_4(x, a, b, c, d, e):
        return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e

    @staticmethod
    def x_5(x, a, b, c, d, e, f):
        return a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + e * x + f

    @staticmethod
    def custom(x, a, b, c, d):
        return a + b * np.sin(x) * b * np.cos(x)

    @staticmethod
    def sin_fit(x, a, b, c, d):
        return a * np.sin(b * (x - c)) ** 2 + d


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    # file_path = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\SimulacoesGATE\\EasyPET3D64\\SpatialResolution-10reps\\attempt_FBP_1_ListMode\\attempt_FBP_1_ListMode.easypet"
    d = DataTest(file_path)
    # d.original_data_analysis()
    # d.sinogram_bot_top()
    # d.sinogram_bot_top_original(1)
    # listMode_t = d.listMode

    # d.remove_adc_value(adc_cuts=[1000, 1000, 1000, 1000])
    # d.remove_adc_value(adc_cuts=[3000, 3000, 3000, 3000], signal="major")
    # d.remove_adc_value(adc_cuts=[1023, 1023, 1023, 1023], signal="equal")
    # d.remove_adc_value(adc_cuts=[2047, 2047, 2047, 2047], signal="equal")
    # d.sinogram_bot_top_original(1)
    # d.angle_correction()
    d.sinogram_from_parametric()
    # [listMode_turn_impar_temp, listMode_turn_par_temp, top, bot] = d.cumulative_odd_even_turn()
    # d.sinogram_odd_even_original(listMode_turn_impar_temp, listMode_turn_par_temp, bot, top)
    # d.events_per_turn_hist()
    # d.only_zeros_listMode(adc_cuts=[0, 0, 0, 0])
    # d.original_data_analysis()
    # d.sinogram_individual_crystals()
    #
    #
    # d.sinogram_bot_top()
    # d.sinogram_bot_top_original(2)
    # d.original_data_analysis()
    plt.show()
