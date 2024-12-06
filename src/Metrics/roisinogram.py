import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from src.EasyPETLinkInitializer.Preprocessing import Sinogram, PrepareEasyPETdata
from src.StandaloneInitializer import ReconstructOpenFileTest
from src.EasyPETLinkInitializer import ReconstructionInitializer
from skimage.transform import iradon, iradon_sart
from scipy import ndimage, misc

# from openpyxl.styles.alignment import vertical_aligments


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

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

data_original = np.load(os.path.join(dirname, "listmode.npy"))
data_cut = np.load(os.path.join(dirname, "listmode_cut.npy"))
list_open_studies = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\PhD\\Simulations\\SimulationIPET\\IPET_DerenzoSmall_500uCi_Na22_ListMode\\IPET_DerenzoSmall_500uCi_Na22_ListMode.easypet"
list_open_studies = "/media/crispim/Storage/Simulations/nec_acquisitions/nec_listmode/nec_listmode.easypet"

# list_open_studies = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\ICNAS\Fantomas ICNAS\\NEC\\Easypet Scan 20 Apr 2022 - 12h 21m 45s\\Easypet Scan 20 Apr 2022 - 12h 21m 45s\\Easypet Scan 20 Apr 2022 - 12h 21m 45s.easypet"
# list_open_studies = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\PhD\\Simulations\\SimulationIPET\\\Moby\\ListModeParts_ListMode\\ListModeParts_ListMode.easypet"
simulation_file = True

if simulation_file:
    remove_incomplete_turn = False
    save_validation_data = False
    swap_sideAtoB = False
else:
    remove_incomplete_turn = False
    save_validation_data = False
    swap_sideAtoB = True
# prepareEasypetdata = PrepareEasyPETdata(study_file=list_open_studies, simulation_file=simulation_file,
#                                         swap_sideAtoB=swap_sideAtoB, remove_incomplete_turn=remove_incomplete_turn,)
#                                         # reconstruction_data_type=self.reconstruction_data_type,
#                                         # top_correction_angle=self.correction_angle,
#                                         # parameters_2D_cut=self.parameters_2D_cut,
#                                         # energy_window=self.energy_window,
#                                         # threshold_ratio=self.threshold_ratio,
#                                         # save_spectrum_file=False, save_validation_data=save_validation_data,
#
#                                         # simulation_file=simulation_file,
#                                         # only_left_side_crystals=self.selectable_crystals_bool[0],
#                                         # only_right_side_crystals=self.selectable_crystals_bool[1],
#                                         # right_left_crystals=self.selectable_crystals_bool[2],
#                                         # left_right_crystals=self.selectable_crystals_bool[3],
#
# prepareEasypetdata.dataRemove()
# # prepareEasypetdata.listMode[:,4] *= -1
#
# # data_original = prepareEasypetdata.listMode
# prepareEasypetdata.listMode = data_cut
# # detector_normalization = "Simulation"
# detector_normalization = "On demand parametric calculation"
# algorithm_function = ReconstructionInitializer(Easypetdata=prepareEasypetdata,
#                                                study_path=os.path.dirname(list_open_studies),
#                                                transform_into_positive=False, simulation_file=simulation_file)
# plt.figure()
# sinogramClass = Sinogram(prepareEasypetdata, algorithm_function.parametric_coordinates)
# sinogramClass.calculate_s_phi()
# # sinogram_cutted = sinogramClass.projected_sinogram()
# sinogramClass.calculateMichelogram(bins_x=100,bins_y=100)
# sinogram_cutted = sinogramClass.michelogram
# result = np.mean(sinogram_cutted[0], axis=2)
#
# theta = sinogram_cutted[1][:-1]
# # result = ndimage.median_filter(sinogram_cutted[0], size=[2, 3])
#
# FBP_volume = iradon_sart(result.T, theta=theta)
#
# extent = np.min(sinogramClass.s), np.max(sinogramClass.s), np.min(sinogramClass.s), np.max(sinogramClass.s)
# # extent = -15, 15, -15, 15
# # FBP_volume = ndimage.median_filter(FBP_volume, size=3)
# plt.figure()
# plt.imshow(FBP_volume, cmap="hot", extent=extent, alpha=.9, interpolation="bilinear")

plt_configure()
plt.figure()


# fsize = 18
# tsize = 30
#
# tdir = 'in'
#
# major = 6.0
# minor = 3.0
#
# style = 'seaborn'
#
# plt.style.use(style)
# # plt.rcParams['text.usetex'] = True
# plt.rcParams['font.size'] = fsize
# plt.rcParams['axes.labelsize'] = 14
# plt.rcParams['xtick.labelsize'] = 14
# plt.rcParams['ytick.labelsize'] = 14
# plt.rcParams['legend.fontsize'] = tsize
# plt.rcParams['xtick.direction'] = tdir
# plt.rcParams['ytick.direction'] = tdir
# plt.rcParams['xtick.major.size'] = major
# plt.rcParams['xtick.minor.size'] = minor
# plt.rcParams['ytick.major.size'] = major
# plt.rcParams['ytick.minor.size'] = minor

arra_y = np.linspace(data_original[:, 5].min(), data_original[:, 5].max(), len(np.unique(data_original[:, 5])))
plt.figure()
data_cut[:,4] = data_cut[:,4] %360
sinogram_cutted = plt.hist2d(data_cut[:, 4], data_cut[:, 5],
                             range=[[data_cut[:, 4].min(), data_cut[:, 4].max()],
                                    [data_cut[:, 5].min(), data_cut[:, 5].max()]],
                             bins=[np.unique(data_cut[:, 4]), np.unique(data_cut[:, 5])])
# sinogram_cutted = plt.hist2d(sinogramClass.phi, sinogramClass.s,
#                              range=[[sinogramClass.phi.min(), sinogramClass.phi.max()],
#                                     [sinogramClass.s.min(), sinogramClass.s.max()]],
#                              bins=[int(len(np.unique(sinogramClass.phi))/100), int(len(np.unique(sinogramClass.s))/1000)])


plt.figure()
# plt.rcParams['text.usetex'] = True
# matplotlib.rc('text.latex', preamble=r'\usepackage{cmbright}')


sinogram = plt.hist2d(data_original[:, 4], data_original[:, 5],
                      bins=[len(np.unique(data_original[:, 4])), len(np.unique(data_original[:, 5]))])
(fig, (ax1, ax2)) = plt.subplots(2, 1)
ax1.imshow(sinogram[0].T, interpolation="gaussian", cmap="hot",
           extent=[0, 360, data_original[:, 5].min(), data_original[:, 5].max()])
ax1.text(350, data_original[:, 5].max() - 5, r'$whole \,body$', color="white", horizontalalignment='right',
         verticalalignment="top")
ax1.grid(False)
ax1.set_xticks([])
ax1.set_ylabel("$Front \, motor$ \n $angular \, position$")
y_ticks = np.linspace(data_original[:, 5].min(), data_original[:, 5].max(), 5)
y_ticks_str = ["${}$".format(int(v))+"$^{\circ}$" for v in y_ticks]
ax1.set_yticks(np.linspace(data_original[:, 5].min(), data_original[:, 5].max(), 5))
ax1.set_yticklabels(y_ticks_str)

# ax2.imshow(sinogram_cutted[0].T, interpolation="gaussian", cmap="hot",
#            extent=[0, 360, data_original[:, 5].min(), data_original[:, 5].max()])
# ax2.imshow(sinogram_cutted[0].T, interpolation="gaussian", cmap="hot",
#            extent=[0, 360, -30,30])
#set sinogram with the same range as the original data
sinogram_cutted_new= np.pad(sinogram_cutted[0].T, ((int(np.abs(sinogram[2][0]-sinogram_cutted[2][0])),int(np.abs(sinogram[2][-1]-sinogram_cutted[2][-1]))),((0,0))), 'constant', constant_values=((0,0),(0,0)))
# sinogram_cutted_new= np.pad(sinogram_cutted[0].T, ((400,400),((0,0))), 'constant', constant_values=((100,100),(0,0)))


# ax2.imshow(sinogram_cutted_new, interpolation="gaussian", cmap="hot",
#            extent = [sinogram_cutted[1][0],sinogram_cutted[1][-1],sinogram_cutted[2][0],sinogram_cutted[2][-1]])
ax2.imshow(sinogram_cutted_new, interpolation="gaussian", cmap="hot", extent = [0,360,data_original[:,5].min(),data_original[:,5].max()])
ax2.set_xlabel("$Rear \, motor$ \n $angular \,position$")
ax2.set_ylabel("$Front \, motor$ \n $angular \, position$")
ax2.set_ylim(data_original[:, 5].min(), data_original[:, 5].max())
ax2.text(350, data_original[:, 5].max() - 5, r'$heart$', color="white", horizontalalignment='right',
         verticalalignment="top")
ax2.grid(False)
x_ticks = ["${}$".format(int(v))+"$^{\circ}$" for v in np.linspace(0, 360, 6)]
ax2.set_xticks(np.linspace(0, 360, 6))
ax2.set_xticklabels(x_ticks)
ax2.set_yticks(np.linspace(data_original[:, 5].min(), data_original[:, 5].max(), 5))
ax2.set_yticklabels(y_ticks_str)

fig.tight_layout()
plt.savefig("../../outputs/sinogram_intelligent_scan.pdf", dpi=300, bbox_inches='tight')
plt.savefig("../../outputs/sinogram_intelligent_scan.png", dpi=300, bbox_inches='tight')
plt.show()
