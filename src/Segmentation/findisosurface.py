import os
from array import array
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure


class FindISOSurface:
    def __init__(self, volume=None,directory=None):
        self.volume = volume
        self.surface_volume = None
        self.active_pixels = None
        self.contours_list = None
        if directory is not None:
            self.file_name = os.path.join(directory, "surface")

    def threshold2DISOSurface(self):
        surface_volume = self.volume
        self.contours_list = [None] * self.volume.shape[2]
        # 2D isosurface
        for k in range(self.volume.shape[2]):
            self.contours_list[k] = measure.find_contours(self.volume[:,:,k], 0.2)

    def plot2DISOSurface(self):
        fig, ax_list = plt.subplots(4, 8,figsize=(25.6,10.8))
        # fig.figsize=(200,640)

        k=0
        plt.subplots_adjust(wspace=-0.5, hspace=0.5)
        for m in range(ax_list.shape[0]):
            for n in range(ax_list.shape[1]):
                ax_list[m][n].imshow(self.volume[:,:,k], cmap=plt.cm.gray)
                contours = self.contours_list[k]
                for c, contour in enumerate(contours):
                    ax_list[m][n].plot(contour[:, 1], contour[:, 0], linewidth=2)

                ax_list[m][n].axis('image')
                ax_list[m][n].set_xticks([])
                ax_list[m][n].set_yticks([])
                ax_list[m][n].set_aspect('equal')
                k +=1

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()

    def threshold3DISOSurface(self):
        # 3D isosurface
        #
        self.volume[self.volume==np.min(self.volume)] = 0
        surfaces = measure.marching_cubes_lewiner(self.volume, level=0, spacing=(1.0, 1.0, 1.0),
                                                  gradient_direction='descent', step_size=1,
                                                  allow_degenerate=True)

        surface_volume = np.zeros((self.volume.shape))
        coord_array = np.round(surfaces[0], 0).astype(np.int16)
        z_coord_unique = np.unique(coord_array[2])
        for coor in coord_array:
            surface_volume[coor[0], coor[1], coor[2]] += 1

        surface_volume = (1 - surface_volume) * self.volume
        surface_volume[:, :, 0:int(surface_volume.shape[2] / 2)] = 0
        surface_volume[:, 0:int(surface_volume.shape[1] / 2), :] = 0
        surface_volume[:, :, :] = 0

        self.surface_volume = surface_volume
        # xx = (np.tile(np.arange(0, im_final.shape[0]), (im_final.shape[0], 1)) - (
        #             im_final.shape[0] - 1) / 2) ** 2
        # yy = (np.tile(np.arange(0, im_final.shape[1]), (im_final.shape[1], 1)) - (
        #             im_final.shape[1] - 1) / 2) ** 2
        # yy = yy.T
        # circle_cut = xx + yy - (im_final.shape[1] * 0.46) ** 2
        # circle_cut[circle_cut >= 0] = 0
        # circle_cut[circle_cut < 0] = 1
        # im_final = np.tile(circle_cut[:, :, None], (1, 1, im_final.shape[2]))

    def apply_fov_cut(self):
        # Find contours at a constant value of 0.8
        volume = self.volume
        xx = (np.tile(np.arange(0, volume.shape[0]), (volume.shape[0], 1)) - (
                volume.shape[0] - 1) / 2) ** 2
        yy = (np.tile(np.arange(0, volume.shape[1]), (volume.shape[1], 1)) - (
                volume.shape[1] - 1) / 2) ** 2
        yy = yy.T
        circle_cut = xx + yy - (volume.shape[1] * 0.4) ** 2
        circle_cut[circle_cut >= 0] = 0
        circle_cut[circle_cut < 0] = 1
        circle_cut = np.tile(circle_cut[:, :, None], (1, 1, volume.shape[2]))
        self.volume = volume * circle_cut

    def save_calculated_surface(self):
        volume = self.surface_volume.astype(np.float32)
        length = volume.shape[0] * volume.shape[2] * volume.shape[1]
        data = np.reshape(volume, [1, length], order='F')
        output_file = open(self.file_name, 'wb')
        arr = array('f', data[0])
        arr.tofile(output_file)
        output_file.close()

    def get_active_pixels(self):
        self.volume[self.volume == np.min(self.volume)] = 0
        self.surface_volume = np.copy(self.volume)
        self.active_pixels = np.where(self.volume > 0)
        self.surface_volume[self.surface_volume > 0] = 1


if __name__ == "__main__":
    FindISOSurface()
