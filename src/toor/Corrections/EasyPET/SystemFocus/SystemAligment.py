# *******************************************************
# * FILE: SystemAligment.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

def angle_correction(self):
    from skimage import filters

    # listmode = listmode[listmode[:,0] > 400]
    # listmode = listmode[listmode[:,0] < 550]
    # listmode = listmode[listmode[:,1] > 400]
    # listmode = listmode[listmode[:,1] < 550]
    listmode = self.fileBodyData
    bot = listmode[:, 4]
    top = listmode[:, 5]

    s = np.sin(np.radians(top)) * (60 / 2)
    # s[(listmode[:,3] % 2)-(listmode[:,2] % 2) == 0] += 1.175
    # s[(listmode[:,3] % 2 + 1)-(listmode[:,2] % 2 + 1) == 0] -= 1.175
    # s = np.abs(s)
    phi = 90 + bot - top
    # phi = bot + top
    phi = phi % 360
    # top = s
    # bot = phi
    s = np.round(s, 6)
    phi = np.round(phi, 6)
    # self.sinogram = plt.hist2d(bot, top, bins=[len(np.unique(bot)),
    #                                            int(header[5] * header[4] / header[3])])
    # self.sinogram = plt.hist2d(phi, s, bins=[int(len(np.unique(s))), int(len(np.unique(s)))])
    self.sinogram = plt.hist2d(bot, top, bins=[int(len(np.unique(s))), int(len(np.unique(s)))])
    val = filters.threshold_otsu(self.sinogram[0])
    mask = self.sinogram[0] >= val
    print(mask)
    plt.figure(3)
    plt.imshow(mask)
    foot_print = np.array([[0, 0, 0.2, 0, 0],
                           [0, 0, 0.5, 0, 0],
                           [0, 0, 1.0, 0, 0],
                           [0, 0, 0.5, 0, 0],
                           [0, 0, 0.2, 0, 0]])
    # result = ndimage.median_filter(self.sinogram[0], size=4)
    # result = ndimage.median_filter(mask*self.sinogram[0] , size=1, footprint=foot_print)
    result = ndimage.median_filter(self.sinogram[0], size=2)
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
            popt, pcov = curve_fit(SimulationStatusData.gaussian_fit, self.sinogram[2][:-1], result[i, :], method='lm',
                                   p0=p0)
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
    plt.figure(4)
    # plt.plot(self.sinogram[2][:-1], self.sinogram[0][0,:])
    plt.bar(self.sinogram[2][:-1], result[0, :])
    plt.plot(self.sinogram[2][:-1], SimulationStatusData.gaussian_fit(self.sinogram[2][:-1], *gaussian_r[0]), "r--")

    top_fit = gaussian_r[:, 1]
    phi_fit = 90 + self.sinogram[1][:-1] - top_fit
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

    self.header = header
    # hist_shap = np.reshape(self.sinogram[0], self.sinogram[0].shape[0] * self.sinogram[0].shape[1])
    hist_shap = np.reshape(self.sinogram[0], self.sinogram[0].shape[0] * self.sinogram[0].shape[1])

    bot_t = bot_t[hist_shap != 0]
    top_t = top_t[hist_shap != 0]
    bot_t = np.repeat(bot_t, hist_shap[hist_shap != 0].astype(int))
    top_t = np.repeat(top_t, hist_shap[hist_shap != 0].astype(int))
    plt.figure(2)
    plt.scatter(bot_t, top_t)
    # popt, pcov = curve_fit(SimulationStatusData.sin_fit,self.sinogram[1][:-1], gaussian_r[:,1],
    #                        p0=np.array([np.abs((gaussian_r[:,1].max()-gaussian_r[:,1].min())/2),np.pi/360, 0]))

    popt, pcov = curve_fit(SimulationStatusData.sin_fit, phi_fit, gaussian_r[:, 1],
                           p0=np.array(
                               [np.abs((gaussian_r[:, 1].max() - gaussian_r[:, 1].min()) / 2), np.pi / 180, 0, 50]))
    print(np.sqrt(np.diag(pcov)))

    amplitude = SimulationStatusData.sin_fit(bot_t, *popt)
    amplitude_angle = np.degrees(np.arcsin(amplitude / 30))
    # np.sin(np.radians(top)) * (60 / 2)
    print("Angle to correct: {}".format((amplitude_angle.max() + amplitude_angle.min())))
    # popt, pcov = curve_fit(DataTest.sin_fit, bot_t, top_t)
    #
    plt.plot(phi_fit, gaussian_r[:, 1], "go")
    # plt.plot(self.sinogram[1][:-1], np.zeros(len(gaussian_r[:,1])), "y--")
    plt.plot(bot_t,
             SimulationStatusData.sin_fit(bot_t, *popt), 'r--')
    print("Angle to correct: {}".format(np.degrees(np.arcsin((gaussian_r[:, 1].max() + gaussian_r[:, 1].min()) / 30))))
    plt.show()
