import matplotlib.pyplot as plt
import numpy as np
from src.ImageReader import RawDataSetter
from src.Segmentation import FindISOSurface

size_file_m = [101, 101, 153]
file_name = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\PhD\\Simulations\\SimulationIPET\\\Moby\\ListModeParts_ListMode\\map_heart"


s = RawDataSetter(file_name, size_file_m=size_file_m, pixel_size=1, pixel_size_axial=1, offset=0)
s._readFiles()
volume = s.volume

isosurface = FindISOSurface(volume=volume)

isosurface.get_active_pixels()
surface_volume = isosurface.surface_volume
# isosurface.save_calculated_surface()

# plt.figure()

fsize = 18
tsize = 30

tdir = 'in'

major = 6.0
minor = 3.0

style = 'seaborn-dark'

plt.style.use(style)
# plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = fsize
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = tsize
plt.rcParams['xtick.direction'] = tdir
plt.rcParams['ytick.direction'] = tdir
plt.rcParams['xtick.major.size'] = major
plt.rcParams['xtick.minor.size'] = minor
plt.rcParams['ytick.major.size'] = major
plt.rcParams['ytick.minor.size'] = minor
pixel_size = 0.5
(fig, ax) = plt.subplots(1, 1)
projection = np.max(volume, axis=1)
ax.imshow(projection,interpolation="Bilinear", cmap="afmhot",
           extent=[0, pixel_size*projection.shape[0], 0,pixel_size*projection.shape[1]])
ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
ax.grid(False)
ax.set_yticklabels([])
ax.set_xticklabels([])
# plt.xlabel("mm")
# plt.ylabel("mm")
plt.show()