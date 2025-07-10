# *******************************************************
# * FILE: IQ_analyisis.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import numpy as np
# import dicom
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ImageReader import RawDataSetter
import tkinter as tk
from tkinter import filedialog


def plt_configure():
    fsize = 15
    tsize = 18

    tdir = 'in'

    major = 5.0
    minor = 3.0

    style = 'default'
    plt.style.use(style)
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = fsize
    plt.rcParams['legend.fontsize'] = tsize
    plt.rcParams['xtick.direction'] = tdir
    plt.rcParams['ytick.direction'] = tdir
    plt.rcParams['xtick.major.size'] = major
    plt.rcParams['xtick.minor.size'] = minor
    plt.rcParams['ytick.major.size'] = major
    plt.rcParams['ytick.minor.size'] = minor
    plt.rcParams["grid.linestyle"] = (5, 9)
    plt.rcParams["figure.figsize"] = (16, 12)


def MakeMask(clicks, image, diameter=10.0, lenght=20.0, fixed_z=None):
    # Show average of image, and click in center roi
    if fixed_z is not None:
        clicks = 1
    clicks = ginput(clicks)
    if fixed_z is not None:
        upper_z_cut = int(np.ceil(fixed_z + lenght * 0.5 / scale_z))
        lower_z_cut = int(np.floor(fixed_z - lenght * 0.5 / scale_z))

    else:
        upper_z_cut = int(np.ceil(clicks[1][1] + lenght * 0.5 / scale_z))
        lower_z_cut = int(np.floor(clicks[1][1] - lenght * 0.5 / scale_z))

    limits = [lower_z_cut, upper_z_cut]
    image_cutted = image[:, :, lower_z_cut:upper_z_cut]

    radius = diameter * 0.5
    x = clicks[0][0]
    y = clicks[0][1]

    mask = (xx - x) ** 2 + (yy - y) ** 2 <= (radius / scale_z) ** 2
    mask_z = np.copy(image)
    mask_z[:, :, upper_z_cut:] = 0
    mask_z[:, :, :lower_z_cut] = 0
    mask_z[:, :, lower_z_cut:upper_z_cut] = 1
    mask_z = mask_z * mask[..., None]
    image_mean = np.mean(image * mask_z, axis=2)

    return clicks, mask, image_mean, mask_z, limits


plt_configure()
root = tk.Tk()
root.withdraw()
filepath = filedialog.askopenfilename()
path_folder = os.path.dirname(os.path.dirname(filepath))
path_data_validation = os.path.join(path_folder, "IQ_analyisis")
if not os.path.isdir(path_data_validation):
    os.makedirs(path_data_validation)
size_file_m = [103,103,124]
size_file_m = [71,71,129]
r = RawDataSetter(filepath, size_file_m=size_file_m)
# r = RawDataSetter(filepath)
r.read_files()
voxeldata = r.volume

# voxeldata[:, :, -10:] = 0
# voxeldata[:, :, :10] = 0
v_lim = np.array([0.05, 0.8])
scale_x = float(0.4)
scale_y = float(0.4)
scale_z = float(0.4 / (0.4 / 0.44))

extent_x_y = [-scale_x * voxeldata.shape[0] / 2, scale_x * voxeldata.shape[0] / 2,
              -scale_y * voxeldata.shape[1] / 2, scale_y * voxeldata.shape[1] / 2]

# generate coordinates
height = r.size_file_m[0]
width = r.size_file_m[1]
x = np.arange(0, width)
y = np.arange(0, width)
xx, yy = np.meshgrid(x, y)
from Geometry import GeometryDesigner
gd = GeometryDesigner(volume=voxeldata)
gd._draw_image_reconstructed()
# UNIFORMITY###
# Plot average dicom image
f_selection_uniform, (axAxial, axCoronal, axSagittal) = plt.subplots(1, 3)

axial_image = np.mean(voxeldata, axis=2)
coronal_image = np.mean(voxeldata, axis=1)
sagittal_image = np.mean(voxeldata, axis=0)

axAxial.imshow(axial_image, "jet", interpolation="gaussian", clim=v_lim * axial_image.max())
axAxial.set_xlabel("mm")
axAxial.set_ylabel("mm")
axCoronal.imshow(coronal_image.T, "jet", interpolation="gaussian", clim=v_lim * coronal_image.max())
axSagittal.imshow(sagittal_image.T, "jet", interpolation="gaussian", clim=v_lim * sagittal_image.max())

clicks_uni, uni_mask, uni_image_mean, uni_mask_z, limits_uniform = MakeMask(2, voxeldata, 22.5, lenght=10)

axAxial.imshow(uni_mask, "jet", alpha=0.5)
axCoronal.imshow(np.max(uni_mask_z, axis=0).T, "jet", alpha=0.5)
axSagittal.imshow(np.max(uni_mask_z, axis=0).T, "jet", alpha=0.5)
draw()

fig_uniform_profile, main_ax = plt.subplots()
divider = make_axes_locatable(main_ax)
top_ax = divider.append_axes("top", 1.05, pad=0.1, sharex=main_ax)
side_ax = divider.append_axes("right", 1.05, pad=0.1, sharey=main_ax)
top_ax.xaxis.set_tick_params(labelbottom=False)
side_ax.yaxis.set_tick_params(labelbottom=False)

main_ax.imshow(uni_image_mean, "gray", origin='lower', extent=extent_x_y)
main_ax.set_xlabel("$mm$")
main_ax.set_ylabel("$mm$")
main_ax.autoscale(enable=False)
top_ax.autoscale(enable=False)

top_ax.set_ylim(top=1.3 * uni_image_mean[int(uni_image_mean.shape[0] / 2)].max())
top_ax.grid(True)

side_ax.set_xlim(right=1.3 * uni_image_mean[:, int(uni_image_mean.shape[1] / 2)].max())
side_ax.autoscale(enable=False)
side_ax.grid(True)

v_line = main_ax.axvline(0, color='#176BAA')
h_line = main_ax.axhline(0, color='#DE542C')
x_hl = np.arange(uni_image_mean.shape[0]) * scale_x - scale_x * voxeldata.shape[0] / 2
y_hl = uni_image_mean[int(uni_image_mean.shape[0] / 2)]
x_vl = np.arange(uni_image_mean.shape[1]) * scale_y - scale_y * voxeldata.shape[1] / 2
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
uni_voi = (voxeldata * uni_mask_z)
uni_mean = np.mean((voxeldata * uni_mask_z)[voxeldata * uni_mask_z != 0])
uni_std = np.std((voxeldata * uni_mask_z)[voxeldata * uni_mask_z != 0])
uni_std_percent = uni_std / uni_mean * 100.
uni_max = np.max(uni_voi)
uni_min = np.min(uni_voi[voxeldata * uni_mask_z != 0])

print("----UNIFORMITY----")
print("Uniform mean value voi: {} Bq/ml".format(uni_mean))
print("Uniform standard deviation voi: {} Bq/ml".format(uni_std))
print("uni_std_percent: {} %".format(uni_std_percent))
print("uni_max: {}".format(uni_max))
print("uni_min: {}".format(uni_min))

# 5-ROD REGION (recovery coefficients)###
# Plot average dicom image

# side_image = np.mean(voxeldata, axis=1)

# Cyl_diam is diameter of 5-rods in mm,
cyl_diam = np.array([1., 2., 3., 4., 5.])
cyl_mean = np.zeros(np.shape(cyl_diam))
cyl_std = np.zeros(np.shape(cyl_diam))
RC_STD = np.zeros(np.shape(cyl_diam))
cyl_voi_t = [None] * len(cyl_diam)
f_rods_selection, (axAxial_rods, axCoronal_rods, axSagittal_rods) = plt.subplots(1, 3)
axial_image = np.mean(voxeldata[:, :, :limits_uniform[0]], axis=2)
coronal_image = np.mean(voxeldata[:, :, :limits_uniform[0]], axis=1)
sagittal_image = np.mean(voxeldata[:, :, :limits_uniform[0]], axis=0)
axAxial_rods.imshow(axial_image, "jet", interpolation="gaussian")
axCoronal_rods.imshow(coronal_image.T, "jet", interpolation="gaussian")
axSagittal_rods.imshow(sagittal_image.T, "jet", interpolation="gaussian")

# Analyse each rod seperately
image_rods = np.zeros(voxeldata.shape)
fixed_z_coord = None

for i in range(len(cyl_diam)):
    clicks_rods, cyl_mask, cyl_image_mean, rods_mask_z, limits_rods = MakeMask(2, voxeldata, 2 * cyl_diam[i], lenght=10,
                                                                               fixed_z=fixed_z_coord)
    if i == 0:
        fixed_z_coord = clicks_rods[1][1]

    cyl_xy = np.unravel_index(np.argmax(cyl_image_mean), [cyl_image_mean.shape[0], cyl_image_mean.shape[1]])
    image_cutted_ = rods_mask_z * voxeldata
    image_rods += image_cutted_

    cyl_voi = (image_cutted_[cyl_xy[0], cyl_xy[1], limits_rods[0]:limits_rods[1]]).astype(float)
    cyl_voi_t[i] = np.copy(cyl_voi)

    # The mean, cyl_mean, and %std, RC_STD,
    # of the RC of the rod are found
    # cyl_voi = cyl_voi[cyl_voi != 0]
    cyl_voi /= uni_mean
    cyl_mean[i] = np.mean(cyl_voi)
    cyl_std[i] = np.std(cyl_voi)
    RC_STD[i] = 100. * np.sqrt((cyl_std[i] / cyl_mean[i]) ** 2 + (uni_std / uni_mean) ** 2)

print("-----HOT ROD ------")
print("Rod:   {} %".format(["1 mm", "2 mm", "3 mm", "4 mm", "5 mm"]))
print("Mean:   {} %".format(cyl_mean * 100))
print("Std:    {} ".format(cyl_std))
print("RC STD: {} ".format(RC_STD))
cyl_voi_t = np.array(cyl_voi_t)
f_rods_segmentation, (axAxial_seg_rods, axCoronal_seg_rods, axSagittal_seg_rods) = plt.subplots(1, 3)
axial_image = np.mean(image_rods, axis=2)
coronal_image = np.mean(image_rods, axis=1)
sagittal_image = np.mean(image_rods, axis=0)
axAxial_seg_rods.imshow(axial_image, "jet", interpolation="gaussian", )
axCoronal_seg_rods.imshow(coronal_image.T, "jet", interpolation="gaussian")
axSagittal_seg_rods.imshow(sagittal_image.T, "jet", interpolation="gaussian")

fig_recovery_coef = figure()
plt.errorbar(cyl_diam, cyl_mean * 100, fmt="--o", color="black", yerr=RC_STD / 2)
plt.xlabel("$Rod \: diameter (mm)$")
plt.ylabel("$Recovery \: coefficient \: (\%$)")
plt.ylim(0, 100)
plt.grid(True)

fig_axial_profile = figure()
length = 10
cyl_zz = np.linspace(-length / 2, length / 2, len(cyl_voi_t[0]))
markers = ["-s", "-p", "-P", "-*", "-o"]
labels = ["$1 mm$", "$2 mm$", "$3 mm$", "$4 mm$", "$5 mm$"]
for i in range(len(cyl_diam)):
    plt.plot(cyl_zz, cyl_voi_t[i], markers[i], label=labels[i])
plt.xlabel("$Axial \: direction \:(mm)$")
plt.ylabel("$Bq/ml$")
plt.legend()
plt.grid(True)

#
###WATER- AND AIR-FILLED CHAMBERS###
# Plot average dicom image
f_water_air, (axAxial_rods, axCoronal_rods, axSagittal_rods) = plt.subplots(1, 3)
axial_image = np.mean(voxeldata[:, :, limits_uniform[1]:], axis=2)
coronal_image = np.mean(voxeldata[:, :, limits_uniform[1]:], axis=1)
sagittal_image = np.mean(voxeldata[:, :, limits_uniform[1]:], axis=0)
axAxial_rods.imshow(axial_image, "jet", interpolation="gaussian")
axCoronal_rods.imshow(coronal_image.T, "jet", interpolation="gaussian")
axSagittal_rods.imshow(sagittal_image.T, "jet", interpolation="gaussian")





# Printing results
with open(os.path.join(path_data_validation, "Results.txt"), "a") as f:
    print("----UNIFORMITY----", file=f)
    print("Uniform mean value voi: {} Bq/ml".format(uni_mean),file=f)
    print("Uniform standard deviation voi: {} Bq/ml".format(uni_std), file=f)
    print("uni_std_percent: {} %".format(uni_std_percent), file=f)
    print("uni_max: {}".format(uni_max), file=f)
    print("uni_min: {}".format(uni_min), file=f)
    print("-----HOT ROD ------", file=f)
    print("Rod:   {} %".format(["1 mm", "2 mm", "3 mm", "4 mm", "5 mm"]), file=f)
    print("Mean:   {} %".format(cyl_mean * 100), file=f)
    print("Std:    {} ".format(cyl_std), file=f)
    print("RC STD: {} ".format(RC_STD), file=f)
    # Analyse one chamber at the time
    for i in range(2):
        clicks_rods, cyl_mask, cyl_image_mean, rods_mask_z, limits_rods = MakeMask(2, voxeldata, 4, lenght=7.5,
                                                                                   fixed_z=None)

        # The spill-over ratio and %std are now found
        w_a_voi = (voxeldata * rods_mask_z)
        # w_a_voi = w_a_image[:, w_a_mask]
        w_a_mean = np.mean(w_a_voi[w_a_voi != 0])
        w_a_std = np.std(w_a_voi)
        w_a_STD = 100. * np.sqrt((w_a_std / w_a_mean) ** 2 + (uni_std / uni_mean) ** 2)
        SOR = w_a_mean / uni_mean
        print("----Water or Air-Filled ------", file=f)
        print("Water mean {} Bq/ml".format(w_a_mean),file=f)
        print("Water std {} Bq/ml".format(w_a_std), file=f)
        print("Water STD {} %".format(w_a_STD), file=f)
        print("Water SOR {}".format(SOR), file=f)


fig_uniform_profile.savefig(os.path.join(path_data_validation, "Uniform_profile.png"), dpi=300, pad_inches=.1,
                            bbox_inches='tight')
f_selection_uniform.savefig(os.path.join(path_data_validation, "f_selection_uniform.png"), dpi=300, pad_inches=.1,
                            bbox_inches='tight')
fig_recovery_coef.savefig(os.path.join(path_data_validation, "fig_recovery_coef.png"), dpi=300, pad_inches=.1,
                          bbox_inches='tight')
fig_axial_profile.savefig(os.path.join(path_data_validation, "fig_axial_profile.png"), dpi=300, pad_inches=.1,
                          bbox_inches='tight')
f_water_air.savefig(os.path.join(path_data_validation, "f_water_air.png"), dpi=300, pad_inches=.1,
                    bbox_inches='tight')
f_rods_selection.savefig(os.path.join(path_data_validation, "f_rods_selection.png"), dpi=300, pad_inches=.1,
                         bbox_inches='tight')
f_rods_segmentation.savefig(os.path.join(path_data_validation, "f_rods_segmentation.png"), dpi=300, pad_inches=.1,
                            bbox_inches='tight')
