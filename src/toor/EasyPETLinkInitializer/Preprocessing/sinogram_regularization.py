import numpy as np
import matplotlib.pyplot as plt


class RegularizationMichelogram:
    def __init__(self, listMode, shape_histogram_bot_top, number_of_crystals):
        self.listMode = listMode
        self.shape_histogram_top_bot = shape_histogram_bot_top
        if number_of_crystals is None:
            number_of_crystals = int(np.max(self.listMode[:,2]))

        self.number_of_crystals = number_of_crystals

    def calculate_statistics(self):
        """"""

    def generate_michelogram(self):
        """"""


if __name__ == "__main__":
    from EasyPETLinkInitializer.EasyPETDataReader import binary_data

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    [listmode, Version_binary, header, dates, otherinfo, acquisitionInfo, stringdata,
     systemConfigurations_info,
     energyfactor_info, peakMatrix_info] = binary_data().open(file_path)
    print("Smallest BOT Angle: {}".format(header[1]/header[2]))
    print("Smallest TOP Angle: {}".format(header[3]/header[4]))
    number_of_crystals = [systemConfigurations_info["array_crystal_x"], systemConfigurations_info["array_crystal_y"]]
    n_z = 1
    n_t = 64
    binx = int(number_of_crystals[0]*number_of_crystals[1])
    binz = int(360/(header[1]/header[2])/n_z)
    binz = len(np.unique(listmode[:,4]))
    bint = int(header[5]/(header[3]/header[4])/n_t)
    print("Smallest TOP Angle: {}".format(header[3] / header[4]*n_t))
    bins = [binx, binx,binz, bint]
    print(bins)
    michelogram = np.histogramdd(listmode[:,2:6], [binx, binx,binz, bint])
    from scipy import ndimage

    out = ndimage.median_filter(np.sum(np.sum(michelogram[0], axis=1), axis=0), size=3)
    michelogram[0][:, :, out < 0.1*np.max(out)] = 0
    output= michelogram
    list_mode_recreate = np.ones((int(np.sum(output[0])), 7))
    decay_factor_cutted = np.ones(int(np.sum(output[0])))
    el = 0
    for i in range(len(output[1][0]) - 1):
        print(i)
        i_element = np.ceil(output[1][0][i])
        for j in range(len(output[1][1]) - 1):
            j_element = np.ceil(output[1][1][j])
            for k in range(len(output[1][2]) - 1):
                k_element = output[1][2][k]
                for t in range(len(output[1][3]) - 1):
                    t_element = output[1][3][t]

                    if output[0][i, j, k, t] != 0:

                        for v in range(int(output[0][i, j, k, t])):
                            list_mode_recreate[el + v, 2] = i_element
                            list_mode_recreate[el + v, 3] = j_element
                            list_mode_recreate[el + v, 4] = k_element
                            list_mode_recreate[el + v, 5] = t_element
                        el += int(output[0][i, j, k, t])
    np.save("list_mode_recreate.npy", list_mode_recreate)
    plt.imshow(np.sum(np.sum(michelogram[0], axis=1), axis=0))
    plt.show()
