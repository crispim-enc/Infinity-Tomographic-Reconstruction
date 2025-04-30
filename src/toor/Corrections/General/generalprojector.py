#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: generalprojector
# * AUTHOR: Pedro Encarnação
# * DATE: 13/03/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************


import numpy as np


class GeneralProjector:
    def __init__(self):
        self._imIndexX = None
        self._imIndexY = None
        self._imIndexZ = None
        self._xRangeLim = None
        self._yRangeLim = None
        self._zRangeLim = None

    def transformIntoPositivePoints(self):
        """
        Transform the points into positive points
        """
        for coor in range(3):
            abs_min_min = np.abs(np.min(
                [np.min(self.pointCenterList[:, coor]), np.min(self.pointCorner1List[:, coor]),
                 np.min(self.pointCorner2List[:, coor]),
                 np.min(self.pointCorner3List[:, coor]), np.min(self.pointCorner4List[:, coor])]))

            # abs_min_min = np.abs(np.min(self.pointCenterList[:, coor]))

            self.pointCorner1List[:, coor] += abs_min_min
            self.pointCorner2List[:, coor] += abs_min_min
            self.pointCorner3List[:, coor] += abs_min_min
            self.pointCorner4List[:, coor] += abs_min_min
            self.pointCenterList[:, coor] += abs_min_min

    def amplifyPointsToGPUCoordinateSystem(self):
        """
        Amplify the points to the GPU coordinate system acorddingly to the voxel size
        :return:
        """
        for coor in range(3):
            self.pointCenterList[:, coor] = (self.pointCenterList[:, coor] / self.voxelSize[coor]).astype(
                np.float32)
            self.pointCorner1List[:, coor] = (self.pointCorner1List[:, coor] / self.voxelSize[coor]).astype(
                np.float32)
            self.pointCorner2List[:, coor] = (self.pointCorner2List[:, coor] / self.voxelSize[coor]).astype(
                np.float32)
            self.pointCorner3List[:, coor] = (self.pointCorner3List[:, coor] / self.voxelSize[coor]).astype(
                np.float32)
            self.pointCorner4List[:, coor] = (self.pointCorner4List[:, coor] / self.voxelSize[coor]).astype(
                np.float32)

    def createVectorialSpace(self):
        """
        Create the vectorial space
        Needs pointCenterList, pointCorner1List, pointCorner2List, pointCorner3List, pointCorner4List to be
        defined first
        """

        self.transformIntoPositivePoints()
        self.amplifyPointsToGPUCoordinateSystem()

        # Create a int range array from 0 to number of pixels)
        # self.x_range_lim = [np.floor((np.max(x) - FoV) / 2), np.ceil((np.max(x) - FoV) / 2 + np.ceil(FoV))]
        # self.y_range_lim = [np.floor((np.max(y) - FoV) / 2), np.ceil((np.max(y) - FoV) / 2 + np.ceil(FoV))]
        # self.z_range_lim = [np.floor(np.min(z) - 1.5 * crystal_height), np.ceil(np.max(z) + 1.5 * crystal_height)]
        # self.x_range_lim = [-self.FoVRadial/2, self.FoVRadial/2]
        # self.y_range_lim = [-self.FoVTangencial/2, self.FoVTangencial/2]
        # self.z_range_lim = [-self.FoVAxial/2, self.FoVAxial/2]
        self.max_x = self.pointCorner1List[:, 0].max()
        self.max_y = self.pointCorner1List[:, 1].max()

        extra_pixel_x = np.abs(self.pointCorner1List[:, 0].min() - self.pointCorner2List[:, 0].min())
        extra_pixel_y = np.abs(self.pointCorner1List[:, 1].min() - self.pointCorner4List[:, 1].min())
        extra_pixel_z = np.abs(self.pointCorner1List[:, 2].min() - self.pointCorner3List[:, 2].min())
        if self._only_fov:
            self.x_range_lim = [np.floor((self.max_x - self.fov) / 2),
                                np.ceil((self.max_x - self.fov) / 2 + np.ceil(self.fov))]
            self.y_range_lim = [np.floor((self.max_y - self.fov) / 2),
                                np.ceil((self.max_y - self.fov) / 2 + np.ceil(self.fov))]
        else:
            self.x_range_lim = [np.ceil(self.pointCorner1List[:, 0].min()),
                                np.ceil(self.pointCorner1List[:, 0].max())]
            self.y_range_lim = [np.ceil(self.pointCorner1List[:, 1].min()),
                                np.ceil(self.pointCorner1List[:, 1].max())]
        self.z_range_lim = [np.ceil(self.pointCorner3List[:, 2].min()),
                            np.ceil(self.pointCorner1List[:, 2].max() + extra_pixel_z)]

        self.number_of_pixels_x = int(np.ceil(self.x_range_lim[1] - self.x_range_lim[0]))
        self.number_of_pixels_y = int(np.ceil(self.y_range_lim[1] - self.y_range_lim[0]))
        self.number_of_pixels_z = int(np.ceil(self.z_range_lim[1] - self.z_range_lim[0]))

        self.im_index_x = np.ascontiguousarray(
            np.empty((self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z), dtype=np.int32))
        self.im_index_y = np.ascontiguousarray(
            np.empty((self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z), dtype=np.int32))
        self.im_index_z = np.ascontiguousarray(
            np.empty((self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z), dtype=np.int32))
        print(f"Image Shape{self.im_index_z.shape}")
        x_range = np.arange(self.x_range_lim[0], self.x_range_lim[1],
                            dtype=np.int32)
        y_range = np.arange(self.y_range_lim[0], self.y_range_lim[1],
                            dtype=np.int32)

        z_range = np.arange(self.z_range_lim[0], self.z_range_lim[1],
                            dtype=np.int32)

        # Create 3 EMPTY arrays with images size.

        # Repeat values in one direction. Like x_axis only grows in x_axis (0, 1, 2 ... number of pixels)
        # but repeat these values on y an z axis
        self.im_index_x[:] = x_range[..., None, None]
        self.im_index_y[:] = y_range[None, ..., None]
        self.im_index_z[:] = z_range[None, None, ...]