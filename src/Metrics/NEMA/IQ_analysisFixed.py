import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import median_filter
import numpy as np
from src.ImageReader.DICOM.filesreader import DicomReader
from src.ImageReader.RawData import RawDataSetter
from src.Phantoms import NEMAIQ2008NU


class IQAnalysis:
    def __init__(self, filepath=None, phantom_center=None, dicom_files=False, voxelSize=None, size_file_m=None, frame_time=0):
        self.limits_uniform = None
        if phantom_center is None:
            phantom_center = np.array([0, 0, -5])
        self._filePath = filepath
        self._phantomCenter = phantom_center

        self.colorMap = "hot"
        self.v_lim = np.array([0.12, 0.9])
        if dicom_files:
            d = DicomReader(_file_init=self._filePath)
            d.readDirectory()
            if frame_time is None:
                self.voxeldata = np.flip(np.rot90(d.volumes(), 1), axis=0)[:, :, :, frame_time]
            else:
                self.voxeldata = np.flip(np.rot90(d.volumes(),1),axis=0)[:,:,:,frame_time]
            dicomHeader = d.dicomHeaders
            voxelSize = d.dicomHeaders[0].PixelSpacing
            voxelSize.append(d.dicomHeaders[0].SliceThickness)
        else:
            r = RawDataSetter(filepath, size_file_m=size_file_m)
            r.read_files()
            self.voxeldata = np.flip(np.rot90(r.volume,1),axis=0)
            if voxelSize is None:
                voxelSize = [1, 1, 72.68/ size_file_m[2]]
            else:
                size_file_m = r.size_file_m
                voxelSize[2] = 72.68/ size_file_m[2]
        # self.voxeldata = median_filter(self.voxeldata, 3)
        # filter_2 = median_filter(self.voxeldata, 5)
        # self.voxeldata[filter_2!=0] = self.voxeldata[filter_2!=0]* (median_filter(self.voxeldata[filter_2!=0], 3) / filter_2[filter_2!=0])
        self.scale_x = float(voxelSize[0])
        self.scale_y = float(voxelSize[1])
        self.scale_z = float(voxelSize[2])
        self._phantom = NEMAIQ2008NU(centerPhantom=[0, 0, self.scale_z*self.voxeldata.shape[2] / 2])
        self.extent_x_y = [-self.scale_x * self.voxeldata.shape[0] / 2, self.scale_x * self.voxeldata.shape[0] / 2,
                      - self.scale_y * self.voxeldata.shape[1] / 2, self.scale_y * self.voxeldata.shape[1] / 2]

        # generate coordinates

        self.x = np.arange(-self.scale_x*self.voxeldata.shape[0]/2,self.scale_x* self.voxeldata.shape[0]/2,self.scale_x)
        self.y = np.arange(-self.scale_y*self.voxeldata.shape[1]/2, self.scale_y* self.voxeldata.shape[1]/2, self.scale_y)
        self.xx, self.yy = np.meshgrid(self.x, self.y)

        self.uni_voi = None
        self.uni_mean = None
        self.uni_std = None
        self.uni_std_percent = None
        self.uni_max = None
        self.uni_min = None

    def MakeMask(self, location, diameter=10.0, lenght=20.0, fixed_z=None):
        # Show average of image, and click in center roi
        image = self.voxeldata

        upper_z_cut = int(np.ceil(location[2] + lenght * 0.5) / self.scale_z)
        lower_z_cut = int(np.floor(location[2] - lenght * 0.5) / self.scale_z)

        limits = [lower_z_cut, upper_z_cut]
        image_cutted = image[:, :, lower_z_cut:upper_z_cut]

        radius = diameter * 0.5
        x = location[0]
        y = location[1]

        mask = (self.xx - x) ** 2 + (self.yy - y) ** 2 <= (radius ** 2)
        mask_z = np.copy(image)
        mask_z[:, :, upper_z_cut:] = 0
        mask_z[:, :, :lower_z_cut] = 0
        mask_z[:, :, lower_z_cut:upper_z_cut] = 1
        mask_z = mask_z * mask[..., None]
        image_mean = np.mean(image * mask_z, axis=2)

        return mask, image_mean, mask_z, limits

    def generateUniformData(self):
        self.f_selection_uniform, (axAxial, axCoronal, axSagittal) = plt.subplots(1, 3)

        axial_image = np.mean(self.voxeldata[:,:, 45:65], axis=2)
        coronal_image = np.mean(self.voxeldata, axis=1)
        sagittal_image = np.mean(self.voxeldata, axis=0)

        axAxial.imshow(axial_image, self.colorMap, interpolation="gaussian", clim=self.v_lim * axial_image.max())
        # axAxial.axis('off')
        # axAxial.set_xlabel("mm")
        # axAxial.set_ylabel("mm")
        axCoronal.imshow(coronal_image.T, self.colorMap, interpolation="gaussian", clim=self.v_lim * coronal_image.max())
        # axCoronal.axis('off')
        axSagittal.imshow(sagittal_image.T, self.colorMap, interpolation="gaussian", clim=self.v_lim * sagittal_image.max())
        # axSagittal.axis('off')
        center = self._phantom._bodyHollow.center
        # center = self._phantom._bodyHollow.center
        center[2] -= 11.5
        uni_mask, uni_image_mean, uni_mask_z, self.limits_uniform = self.MakeMask(center, 22.5, lenght=10)
        axAxial.imshow(uni_mask, self.colorMap, alpha=0.5)
        axCoronal.imshow(np.max(uni_mask_z, axis=0).T, self.colorMap, alpha=0.5)
        axSagittal.imshow(np.max(uni_mask_z, axis=1).T, self.colorMap, alpha=0.5)
        draw()
        plt.tight_layout()

        self.fig_uniform_profile, main_ax = plt.subplots()
        divider = make_axes_locatable(main_ax)
        top_ax = divider.append_axes("top", 1.05, pad=0.1, sharex=main_ax)
        side_ax = divider.append_axes("right", 1.05, pad=0.1, sharey=main_ax)
        top_ax.xaxis.set_tick_params(labelbottom=False)
        side_ax.yaxis.set_tick_params(labelbottom=False)

        main_ax.imshow(uni_image_mean, "hot", origin='lower', extent=self.extent_x_y)
        main_ax.set_xlabel("$mm$")
        main_ax.set_ylabel("$mm$")
        main_ax.autoscale(enable=False)
        top_ax.autoscale(enable=False)

        top_ax.set_ylim(top=1.3 * uni_image_mean[int(uni_image_mean.shape[0] / 2)].max())
        top_ax.grid(True)
        top_ax.minorticks_on()

        side_ax.set_xlim(right=1.3 * uni_image_mean[:, int(uni_image_mean.shape[1] / 2)].max())
        side_ax.autoscale(enable=False)
        side_ax.grid(True)
        side_ax.minorticks_on()

        v_line = main_ax.axvline(0, color='#176BAA')
        h_line = main_ax.axhline(0, color='#DE542C')
        x_hl = np.arange(uni_image_mean.shape[0]) * self.scale_x - self.scale_x * self.voxeldata.shape[0] / 2
        y_hl = uni_image_mean[int(uni_image_mean.shape[0] / 2)]
        x_vl = np.arange(uni_image_mean.shape[1]) * self.scale_y - self.scale_y * self.voxeldata.shape[1] / 2
        y_vl = uni_image_mean[:, int(uni_image_mean.shape[1] / 2)]
        h_prof, = top_ax.plot(x_hl,
                              y_hl, "-",
                              color='#DE542C')
        v_prof, = side_ax.plot(y_vl, x_vl, "-", color='#176BAA')

        top_ax.fill(x_hl, y_hl, color='#DE542C', alpha=0.5)
        top_ax.ticklabel_format(useMathText=True, axis='y', style="sci", scilimits=(0, 0))
        top_ax.set_ylabel("$Bq/ml$")

        side_ax.fill(y_vl, x_vl, color='#176BAA', alpha=0.5)
        side_ax.ticklabel_format(useMathText=True, style="sci", axis='x', scilimits=(0, 0))
        side_ax.set_xlabel("$Bq/ml$")

        # mean, min, max activity concentration and %std is output
        # self.mask_uniform = uni_mask_z
        self.uni_voi = (self.voxeldata * uni_mask_z)
        self.uni_mean = np.mean((self.voxeldata * uni_mask_z)[self.voxeldata * uni_mask_z != 0])
        self.uni_std = np.std((self.voxeldata * uni_mask_z)[self.voxeldata * uni_mask_z != 0])
        self.uni_std_percent = self.uni_std / self.uni_mean * 100.
        self.uni_max = np.max(self.uni_voi)
        self.uni_min = np.min(self.uni_voi[self.voxeldata * uni_mask_z != 0])


        print("----UNIFORMITY----")
        print("Uniform mean value voi: {} Bq/ml".format(self.uni_mean))
        print("Uniform standard deviation voi: {} Bq/ml".format(self.uni_std))
        print("uni_std_percent: {} %".format(self.uni_std_percent))
        print("uni_max: {}".format(self.uni_max))
        print("uni_min: {}".format(self.uni_min))

    def fiveRods(self):
        # Cyl_diam is diameter of 5-rods in mm,
        voxeldata = self.voxeldata
        rods = [self._phantom._rod1mm, self._phantom._rod2mm, self._phantom._rod3mm,self._phantom._rod4mm,self._phantom._rod5mm]
        cyl_diam = np.array([1., 2., 3., 4., 5.])
        self.cyl_mean = np.zeros(np.shape(cyl_diam))
        self.cyl_std = np.zeros(np.shape(cyl_diam))
        self.RC_STD = np.zeros(np.shape(cyl_diam))
        cyl_voi_t = [None] * len(cyl_diam)
        self.f_rods_selection, (axAxial_rods, axCoronal_rods, axSagittal_rods) = plt.subplots(1, 3)
        axial_image = np.mean(voxeldata[:, :, :self.limits_uniform[0]], axis=2)
        coronal_image = np.mean(voxeldata[:, :, :self.limits_uniform[0]], axis=1)
        sagittal_image = np.mean(voxeldata[:, :, :self.limits_uniform[0]], axis=0)
        axAxial_rods.imshow(axial_image, self.colorMap, interpolation="gaussian")
        axCoronal_rods.imshow(coronal_image.T, self.colorMap, interpolation="gaussian")
        axSagittal_rods.imshow(sagittal_image.T, self.colorMap, interpolation="gaussian")

        # Analyse each rod seperately
        image_rods = np.zeros(voxeldata.shape)
        fixed_z_coord = None

        for i in range(len(cyl_diam)):
            # uni_mask, uni_image_mean, uni_mask_z, self.limits_uniform = self.MakeMask(
            #     self._phantom._bodyHollowWater.center, 22.5, lenght=10)
            center = rods[i].center
            center[2] -= 5
            cyl_mask, cyl_image_mean, rods_mask_z, limits_rods = self.MakeMask(center, 0.5 * cyl_diam[i],
                                                                                       lenght=10)


            cyl_xy = np.unravel_index(np.argmax(cyl_image_mean), [cyl_image_mean.shape[0], cyl_image_mean.shape[1]])
            image_cutted_ = rods_mask_z * voxeldata
            image_rods += image_cutted_

            cyl_voi = (image_cutted_[cyl_xy[0], cyl_xy[1], limits_rods[0]:limits_rods[1]]).astype(float)
            cyl_voi_t[i] = np.copy(cyl_voi)

            # The mean, cyl_mean, and %std, RC_STD,
            # of the RC of the rod are found
            # cyl_voi = cyl_voi[cyl_voi != 0]
            cyl_voi /= self.uni_mean
            self.cyl_mean[i] = np.mean(cyl_voi)

            self.cyl_mean[i] = np.mean(image_cutted_[image_cutted_!=0])/self.uni_mean
            self.cyl_std[i] = np.std(cyl_voi)
            self.RC_STD[i] = 100. * np.sqrt((self.cyl_std[i] / self.cyl_mean[i]) ** 2 + (self.uni_std / self.uni_mean) ** 2)

        print("-----HOT ROD ------")
        print("Rod:   {} %".format(["1 mm", "2 mm", "3 mm", "4 mm", "5 mm"]))
        print("Mean:   {} %".format(self.cyl_mean * 100))
        print("Std:    {} ".format(self.cyl_std))
        print("RC STD: {} ".format(self.RC_STD))
        # cyl_voi_t = np.array(cyl_voi_t)
        self.f_rods_segmentation, (axAxial_seg_rods, axCoronal_seg_rods, axSagittal_seg_rods) = plt.subplots(1, 3)
        axial_image = np.mean(image_rods, axis=2)
        coronal_image = np.mean(image_rods, axis=1)
        sagittal_image = np.mean(image_rods, axis=0)
        axAxial_seg_rods.imshow(axial_image, self.colorMap, interpolation="gaussian", )
        axCoronal_seg_rods.imshow(coronal_image.T, self.colorMap, interpolation="gaussian")
        axSagittal_seg_rods.imshow(sagittal_image.T, self.colorMap, interpolation="gaussian")

        self.fig_recovery_coef = figure()
        # plt.errorbar(cyl_diam, self.cyl_mean * 100, fmt="--o", color="black", yerr=self.RC_STD / 2)
        plt.plot(cyl_diam, self.cyl_mean, "--o", color="black")
        plt.fill_between(cyl_diam, self.cyl_mean - self.cyl_mean * self.RC_STD / 100, self.cyl_mean + self.cyl_mean * self.RC_STD /100, alpha=0.2)
        plt.minorticks_on()
        # And a corresponding grid
        plt.grid(which='both')

        # Or if you want different settings for the grids:
        plt.grid(which='minor', alpha=0.2, linestyle='--')
        plt.grid(which='major', alpha=0.5, linestyle=':')
        plt.xlabel("$Rod \: diameter (mm)$")
        plt.ylabel("$Recovery \: coefficient$")
        plt.ylim(0, 1)
        plt.grid(True)

        self.fig_axial_profile = figure()
        # length = 10
        # cyl_zz = np.linspace(-length / 2, length / 2, len(cyl_voi_t[0]))
        # markers = ["-s", "-p", "-P", "-*", "-o"]
        # labels = ["$1 mm$", "$2 mm$", "$3 mm$", "$4 mm$", "$5 mm$"]
        # for i in range(len(cyl_diam)):
        #     plt.plot(cyl_zz, cyl_voi_t[i], markers[i], label=labels[i])
        # plt.xlabel("$Axial \: direction \:(mm)$")
        # plt.ylabel("$Bq/ml$")
        # plt.legend()
        # plt.grid(True)

    def calculateSOR(self):
        ###WATER- AND AIR-FILLED CHAMBERS###
        # Plot average dicom image
        self.f_water_air, (axAxial_rods, axCoronal_rods, axSagittal_rods) = plt.subplots(1, 3)
        axial_image = np.mean(self.voxeldata[:, :, self.limits_uniform[1]:], axis=2)
        coronal_image = np.mean(self.voxeldata[:, :, self.limits_uniform[1]:], axis=1)
        sagittal_image = np.mean(self.voxeldata[:, :, self.limits_uniform[1]:], axis=0)
        axAxial_rods.imshow(axial_image, self.colorMap, interpolation="gaussian", clim=self.v_lim * axial_image.max())
        axAxial_rods.axis('off')
        axCoronal_rods.imshow(coronal_image.T, self.colorMap, interpolation="gaussian", clim=self.v_lim * axial_image.max())
        axCoronal_rods.axis('off')
        axSagittal_rods.imshow(sagittal_image.T, self.colorMap, interpolation="gaussian", clim=self.v_lim * axial_image.max())
        axSagittal_rods.axis('off')
        plt.tight_layout()
        # Analyse one chamber at the time
        # for i in range(2):
        center = self._phantom._waterChamberFilling.center
        center[2] -= 5

        cyl_mask, cyl_image_mean, rods_mask_z, limits_rods = self.MakeMask(center, 4, lenght=7.5,
                                                                                   fixed_z=None)
        mask_total = rods_mask_z


        # The spill-over ratio and %std are now found
        w_a_voi = (self.voxeldata * rods_mask_z)
        self.w_a_voi = w_a_voi
        # w_a_voi = w_a_image[:, w_a_mask]
        self.w_a_mean = np.mean(w_a_voi[w_a_voi != 0])
        self.w_a_std = np.std(w_a_voi)
        self.w_a_STD = 100. * np.sqrt((self.w_a_std / self.w_a_mean) ** 2 + (self.uni_std / self.uni_mean) ** 2)
        self.SOR_water = self.w_a_mean / self.uni_mean

        print("----Water  ------")
        print("Water mean {} Bq/ml".format(self.w_a_mean))
        print("Water std {} Bq/ml".format(self.w_a_std))
        print("Water STD {} %".format(self.w_a_STD))
        print("Water SOR {}".format(self.SOR_water))

        center = self._phantom._airChamberFilling.center
        center[2] -= 5 # fantoma esta deslocado 5mm em z alterar no futuro
        cyl_mask, cyl_image_mean, rods_mask_z, limits_rods = self.MakeMask(center, 4,
                                                                           lenght=7.5,
                                                                           fixed_z=None)
        mask_total += rods_mask_z
        # mask_total =np.abs(mask_total - 1)
        image_cups = mask_total

        air_voi = (self.voxeldata * rods_mask_z)
        self.air_voi = air_voi
        # air_voi = air_image[:, air_mask]
        self.air_mean = np.mean(air_voi[air_voi != 0])
        self.air_std = np.std(air_voi)
        self.air_STD = 100. * np.sqrt((self.air_std / self.air_mean) ** 2 + (self.uni_std / self.uni_mean) ** 2)
        self.SOR_air = self.air_mean / self.uni_mean

        print("----Air-Filled ------")
        print("Air mean {} Bq/ml".format(self.air_mean))
        print("Air std {} Bq/ml".format(self.air_std))
        print("Air STD {} %".format(self.air_STD))
        print("Air SOR {}".format(self.SOR_air))

        self.f_cups_segmentation, (axAxial_seg_cups, axCoronal_seg_cups, axSagittal_seg_cups) = plt.subplots(1, 3)
        center_x = int(image_cups.shape[0]/2)
        center_y = int(image_cups.shape[1]/2)
        center_z = int(image_cups.shape[2]/2)
        axial_image = np.max(image_cups[:,:,self.limits_uniform[1]:], axis=2)
        coronal_image = np.max(image_cups[:,:, :], axis=1)
        sagittal_image = np.max(image_cups[center_x-5: center_x+5,:, :], axis=0)

        axAxial_seg_cups.imshow(np.mean(self.voxeldata[:,:,self.limits_uniform[1]:], axis=2), self.colorMap, interpolation="gaussian")
        axCoronal_seg_cups.imshow(np.mean(self.voxeldata, axis=1).T, self.colorMap, interpolation="gaussian")
        axSagittal_seg_cups.imshow(np.mean(self.voxeldata[center_x-5: center_x+5,:, :], axis=0).T, self.colorMap, interpolation="gaussian")

        axAxial_seg_cups.imshow(axial_image, self.colorMap, interpolation="gaussian", alpha=0.5 )
        axCoronal_seg_cups.imshow(coronal_image.T, self.colorMap, interpolation="gaussian",alpha=0.5)
        axSagittal_seg_cups.imshow(sagittal_image.T, self.colorMap, interpolation="gaussian", alpha=0.5)




    def saveResults(self, path_data_validation=None, iteration=None):
            """ """
            if path_data_validation is None:
                path_data_validation = os.path.dirname(self._filePath)

            if iteration is not None:
                #make a folder for each iteration
                path_data_validation = os.path.join(path_data_validation, "iteration_{}".format(iteration))
                if not os.path.isdir(path_data_validation):
                    os.makedirs(path_data_validation)

            self.fig_uniform_profile.savefig(os.path.join(path_data_validation, "Uniform_profile.png"), dpi=300, pad_inches=.1,
                                        bbox_inches='tight')
            self.f_selection_uniform.savefig(os.path.join(path_data_validation, "f_selection_uniform.png"), dpi=300,
                                        pad_inches=.1,
                                        bbox_inches='tight')
            self.fig_recovery_coef.savefig(os.path.join(path_data_validation, "fig_recovery_coef.png"), dpi=300, pad_inches=.1,
                                      bbox_inches='tight')
            self.fig_axial_profile.savefig(os.path.join(path_data_validation, "fig_axial_profile.png"), dpi=300, pad_inches=.1,
                                      bbox_inches='tight')
            self.f_water_air.savefig(os.path.join(path_data_validation, "f_water_air.png"), dpi=300, pad_inches=.1,
                                bbox_inches='tight')
            self.f_rods_selection.savefig(os.path.join(path_data_validation, "f_rods_selection.png"), dpi=300, pad_inches=.1,
                                     bbox_inches='tight')
            self.f_rods_segmentation.savefig(os.path.join(path_data_validation, "f_rods_segmentation.png"), dpi=300,
                                        pad_inches=.1,
                                        bbox_inches='tight')
            self.f_cups_segmentation.savefig(os.path.join(path_data_validation, "f_cups_segmentation.png"), dpi=300,
                                        pad_inches=.1,
                                        bbox_inches='tight')
            # Printing results
            with open(os.path.join(path_data_validation, "Results.txt"), "a") as f:
                print("----UNIFORMITY----", file=f)
                print("Uniform mean value voi: {} Bq/ml".format(self.uni_mean), file=f)
                print("Uniform standard deviation voi: {} Bq/ml".format(self.uni_std), file=f)
                print("uni_std_percent: {} %".format(self.uni_std_percent), file=f)
                print("uni_max: {}".format(self.uni_max), file=f)
                print("uni_min: {}".format(self.uni_min), file=f)
                print("-----HOT ROD ------", file=f)
                print("Rod:   {} %".format(["1 mm", "2 mm", "3 mm", "4 mm", "5 mm"]), file=f)
                print("Mean:   {} %".format(self.cyl_mean * 100), file=f)
                print("Std:    {} ".format(self.cyl_std), file=f)
                print("RC STD: {} ".format(self.RC_STD), file=f)
                print("----Water ------", file=f)
                print("Water mean {} Bq/ml".format(self.w_a_mean), file=f)
                print("Water std {} Bq/ml".format(self.w_a_std), file=f)
                print("Water STD {} %".format(self.w_a_STD), file=f)
                print("Water SOR {}".format(self.SOR_water), file=f)
                print("----Air ------", file=f)
                print("Air mean {} Bq/ml".format(self.air_mean), file=f)
                print("Air std {} Bq/ml".format(self.air_std), file=f)
                print("Air STD {} %".format(self.air_STD), file=f)
                print("Air SOR {}".format(self.SOR_air), file=f)

            print("Results saved in: {}".format(path_data_validation))
            # print("----Air ------", file=f)
            # print("Air mean {} Bq/ml".format(self.air_mean), file=f)
            # print("Air std {} Bq/ml".format(self.air_std), file=f)
            # print("Air STD {} %".format(self.air_STD), file=f)
            # print("Air SOR {}".format(self.SOR_air), file=f)

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
    import tkinter as tk
    from tkinter import filedialog
    plt_configure()
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename()
    #

    # filepath = "/home/Documentos/Simulations/Easypet/ID_26 Jan 2022 - 00h 16m 02s_1p80bot_ IMAGE (52, 52, 68).T"

    # filepath = "E:\\OneDrive - Universidade de Aveiro\\SimulacoesGATE\\EasyPET3D64\\NEMA-NU-4-2008-IQ" \
    #            "\\15-December-2022_11h49_54turn_0p005s_1p80bot_0p23top_range108_ListMode\\iterations\\" \
    #            "iq MLEM 0_5 101 101 124\\EasyPETScan_it30_sb0"
    path_folder = os.path.dirname(filepath)
    path_data_validation = os.path.join(path_folder, "IQ_analyisis_man")
    if not os.path.isdir(path_data_validation):
        os.makedirs(path_data_validation)
    # iq_an = IQAnalysis(filepath, dicom_files=False, voxelSize=[0.5, 0.5, 0.5*(0.44/0.4)])
    iq_an = IQAnalysis(filepath, dicom_files=False, voxelSize=[0.5, 0.5, 0.5*(0.44/0.4)], size_file_m=[71, 71, 129])
    voxel_data_copy = np.copy(iq_an.voxeldata)
    # for i in range(voxel_data_copy.shape[3]):
    #     iq_an.voxeldata = voxel_data_copy[:,:,:,i]
    #     iq_an.generateUniformData()
    #     iq_an.fiveRods()
    #     iq_an.saveResults()


    # iq_an.voxeldata = voxel_data_copy[:, :, :, i]
    iq_an.generateUniformData()
    iq_an.fiveRods()
    iq_an.calculateSOR()
    iq_an.saveResults(path_data_validation)




# UNIFORMITY###
# Plot average dicom image


# 5-ROD REGION (recovery coefficients)###
# Plot average dicom image

# side_image = np.mean(voxeldata, axis=1)


#





