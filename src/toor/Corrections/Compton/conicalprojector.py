#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: conicalprojector
# * AUTHOR: Pedro Encarnação
# * DATE: 07/03/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

"""
Brief description of the file.
"""
import numpy as np


class ConicalProjector:
    """
    the general quadratic equation of the conic surface in the form:
    Ax^2+By^2+Cz^2+Dxy+Eyz+Fzx+Gx+Hy+Iz+J=0

    The conical projector is a conical surface that is used to project the photons into the detector.


    """

    def __init__(self, voxelSize=(1, 1, 1)):
        self._geometryType = "ConicalProjector"
        self._angles = []
        self._energies = []
        self._comptonPointsOfInteraction = None
        self._scatterPointsOfInteraction = None
        self._vectorComptonScatter = None
        self._errorConicalSurface = 0.02  # error of the conical surface
        self._conicalEquationOut = None
        self._conicalEquationIn = None
        self._A = None
        self._B = None
        self._C = None
        self._D = None
        self._E = None
        self._F = None
        self._H = None
        self._G = None
        self._I = None
        self._J = None
        self._imIndexX = None
        self._imIndexY = None
        self._imIndexZ = None
        self._xRangeLim = None
        self._yRangeLim = None
        self._zRangeLim = None
        self._numberOfPixelsX = None
        self._numberOfPixelsY = None
        self._numberOfPixelsZ = None
        self._voxelSize = np.array(voxelSize, dtype=np.float32)

    def __str__(self):
        return f"ConicalProjector: {self._geometryType}"

    @property
    def angles(self):
        """
        Get the angles of the conical projector.
        """
        return self._angles

    @property
    def energies(self):
        """
        Get the energies of the conical projector.
        """
        return self._energies

    @property
    def scatterPointsOfInteraction(self):
        """
        Get the scatter points of interaction of the conical projector.
        """
        return self._scatterPointsOfInteraction

    @property
    def comptonPointsOfInteraction(self):
        """
        Get the compton points of interaction of the conical projector.
        """
        return self._comptonPointsOfInteraction

    def setConeEquation(self):
        """
        Set the conical equation of the outer surface.
        """
        # x^2 + y^2−k2z2−2   x0x−2    y0y + 2    k2z0z + (x02+y02−k2z0) = 0
        c = np.cos(self._angles)
        self._A = self._vectorComptonScatter[:, 0] ** 2 - c ** 2
        self._B = self._vectorComptonScatter[:, 1] ** 2 - c ** 2
        self._C = self._vectorComptonScatter[:, 2] ** 2 - c ** 2
        self._D = 2 * self._vectorComptonScatter[:, 0] * self._vectorComptonScatter[:, 1]
        self._E = 2 * self._vectorComptonScatter[:, 0] * self._vectorComptonScatter[:, 2]
        self._F = 2 * self._vectorComptonScatter[:, 1] * self._vectorComptonScatter[:, 2]
        self._G = (-2 * self._A * self._scatterPointsOfInteraction[:, 0] - self._D *
                   self._scatterPointsOfInteraction[:, 1] - self._E * self._scatterPointsOfInteraction[:, 2])
        self._H = (-2 * self._B * self._scatterPointsOfInteraction[:, 1] -
                   self._D * self._scatterPointsOfInteraction[:, 0] - self._F * self._scatterPointsOfInteraction[:, 2])
        self._I = (-2 * self._C * self._scatterPointsOfInteraction[:, 2] -
                   self._E * self._scatterPointsOfInteraction[:, 0] - self._F * self._scatterPointsOfInteraction[:, 1])
        self._J = (self._A * self._scatterPointsOfInteraction[:, 0] ** 2 +
                   self._B * self._scatterPointsOfInteraction[:, 1] ** 2 +
                   self._C * self._scatterPointsOfInteraction[:, 2] ** 2 +
                   self._D * self._scatterPointsOfInteraction[:, 0] * self._scatterPointsOfInteraction[:, 1] +
                   self._E * self._scatterPointsOfInteraction[:, 0] * self._scatterPointsOfInteraction[:, 2] +
                   self._F * self._scatterPointsOfInteraction[:, 1] * self._scatterPointsOfInteraction[:, 2])

        return np.array([self._A, self._B, self._C, self._D, self._E, self._F, self._G,
                         self._H, self._I, self._J], dtype=np.float32)

    def setVectorComptonScatter(self):
        """
        Vector of the Compton scatter
        """
        self._vectorComptonScatter = self._comptonPointsOfInteraction - self._scatterPointsOfInteraction # confirmar
        norm_vector = ConicalProjector.normVector(self._vectorComptonScatter)
        for coordinate in range(3):
            self._vectorComptonScatter[:, coordinate] = self._vectorComptonScatter[:, coordinate] / norm_vector

        return self._vectorComptonScatter

    @staticmethod
    def normVector(vector):
        """
        Norm of the vector
        """
        return np.sqrt(vector[:, 0] ** 2 + vector[:, 1] ** 2 + vector[:, 2] ** 2)

    def setAngles(self, angles, angles_unit='degrees'):
        """
        Set the angles of the conical projector.
        """
        if angles_unit == 'degrees':
            angles = np.array(np.radians(angles), dtype=np.float32)

        elif angles_unit == 'radians':
            angles = np.array(angles, dtype=np.float32)

        self._angles = angles

    def setEnergies(self, energies):
        """
        Set the energies of the conical projector.
        """
        self._energies = energies
        # as float32
        self._energies = np.array(self._energies, dtype=np.float32)

    def setScatterPointsOfInteraction(self, point):
        """
        Set the point of interaction of the conical projector.
        """

        self._scatterPointsOfInteraction = np.array(point, dtype=np.float32)

    def setComptonPointsOfInteraction(self, point):
        """
        Set the point of interaction of the conical projector.
        """

        self._comptonPointsOfInteraction = np.array(point, dtype=np.float32)

    def setConicalSurfaceError(self, error):
        """
        Set the error of the conical surface.
        """

    def setConicalEquationOut(self):
        """
        Set the conical equation of the outer surface.
        """
        # conical equation of the outer surface

    def setRangeLim(self, x_range_lim=None, y_range_lim=None, z_range_lim=None):
        """
        Set the range limit of the conical projector.
        """
        min_x = np.min(self._scatterPointsOfInteraction[:, 0])
        min_y = np.min(self._scatterPointsOfInteraction[:, 1])
        min_z = np.min(self._scatterPointsOfInteraction[:, 2])
        max_x = np.max(self._scatterPointsOfInteraction[:, 0])
        max_y = np.max(self._scatterPointsOfInteraction[:, 1])
        max_z = np.max(self._scatterPointsOfInteraction[:, 2])
        if x_range_lim is not None:
            self._xRangeLim = x_range_lim/self._voxelSize[0]
        else:
            self._xRangeLim = [np.floor(min_x), np.ceil(max_x)]

        if y_range_lim is not None:
            self._yRangeLim = y_range_lim/self._voxelSize[1]
        else:
            self._yRangeLim = [np.floor(min_y), np.ceil(max_y)]

        if z_range_lim is not None:
            self._zRangeLim = z_range_lim / self._voxelSize[2]
        else:
            self._zRangeLim = [np.floor(min_z), np.ceil(max_z)]

    def transformIntoPositivePoints(self):
        """
        Transform the points into positive points
        """
        for coor in range(3):
            abs_min_min = np.abs(np.min([np.min(self._scatterPointsOfInteraction[:, coor]),
                                            np.min(self._comptonPointsOfInteraction[:, coor])]))

            # abs_min_min = np.abs(np.min(self.pointCenterList[:, coor]))

            self._scatterPointsOfInteraction[:, coor] += abs_min_min
            self._comptonPointsOfInteraction[:, coor] += abs_min_min

    def amplifyPointsToGPUCoordinateSystem(self):
        """
        Amplify the points to the GPU coordinate system acorddingly to the voxel size
        :return:
        """
        for coor in range(3):
            self._scatterPointsOfInteraction[:, coor] = self._scatterPointsOfInteraction[:, coor] / self._voxelSize[coor]
            self._comptonPointsOfInteraction[:, coor] = self._comptonPointsOfInteraction[:, coor] / self._voxelSize[coor]

    def createVectorialSpace(self):
        """
        Create the vectorial space
        Needs pointCenterList, pointCorner1List, pointCorner2List, pointCorner3List, pointCorner4List to be
        defined first
        """

        # self.transformIntoPositivePoints()
        # self.amplifyPointsToGPUCoordinateSystem()

        # Create a int range array from 0 to number of pixels)
        # self.x_range_lim = [np.floor((np.max(x) - FoV) / 2), np.ceil((np.max(x) - FoV) / 2 + np.ceil(FoV))]
        # self.y_range_lim = [np.floor((np.max(y) - FoV) / 2), np.ceil((np.max(y) - FoV) / 2 + np.ceil(FoV))]
        # self.z_range_lim = [np.floor(np.min(z) - 1.5 * crystal_height), np.ceil(np.max(z) + 1.5 * crystal_height)]

        self._numberOfPixelsX = int(np.ceil(self._xRangeLim[1] - self._xRangeLim[0]))
        self._numberOfPixelsY = int(np.ceil(self._yRangeLim[1] - self._yRangeLim[0]))
        self._numberOfPixelsZ = int(np.ceil(self._zRangeLim[1] - self._zRangeLim[0]))

        self._imIndexX = np.ascontiguousarray(
            np.empty((self._numberOfPixelsX, self._numberOfPixelsY, self._numberOfPixelsZ), dtype=np.int32))
        self._imIndexY = np.ascontiguousarray(
            np.empty((self._numberOfPixelsX, self._numberOfPixelsY, self._numberOfPixelsZ), dtype=np.int32))
        self._imIndexZ = np.ascontiguousarray(
            np.empty((self._numberOfPixelsX, self._numberOfPixelsY, self._numberOfPixelsZ), dtype=np.int32))
        print(f"Image Shape{self._imIndexZ.shape}")
        x_range = np.arange(self._xRangeLim[0], self._xRangeLim[1],
                            dtype=np.int32)
        y_range = np.arange(self._yRangeLim[0], self._yRangeLim[1],
                            dtype=np.int32)

        z_range = np.arange(self._zRangeLim[0], self._zRangeLim[1],
                            dtype=np.int32)

        # Create 3 EMPTY arrays with images size.

        # Repeat values in one direction. Like x_axis only grows in x_axis (0, 1, 2 ... number of pixels)
        # but repeat these values on y an z axis
        self._imIndexX[:] = x_range[..., None, None]
        self._imIndexY[:] = y_range[None, ..., None]
        self._imIndexZ[:] = z_range[None, None, ...]


if __name__ == "__main__":

    conicalProjector = ConicalProjector(voxelSize=(2, 2, 2))
    # conicalProjector = ConicalProjector(voxelSize=(1, 1, 1))
    # x^2 + y^2−k2z2−2   x0x−2    y0y + 2    k2z0z + (x02+y02−k2z0) = 0

    point_scatter = np.array([[0, 0, 0], [20, 3, 10]], dtype=np.float32)
    point_compton = np.array([[5, 2, 2], [10, 10, 10]], dtype=np.float32)
    angles_compton = np.array([90, 10], dtype=np.float32)
    energies_compton = np.array([511, 511], dtype=np.float32)

    file_path = r"C:\Users\pedro\OneDrive\Documentos\2Sources_openAngle_allEvents.txt"
    data = np.loadtxt(file_path, delimiter=' ', skiprows=1)
    point_scatter = data[:, 3:6]
    point_compton = data[:, 0:3]
    angles_compton = data[:, 6]
    # remove larger than 100 degrees angles
    point_scatter = point_scatter[angles_compton < 90]
    point_compton = point_compton[angles_compton < 90]
    angles_compton = angles_compton[angles_compton < 90]

    conicalProjector.setScatterPointsOfInteraction(point_scatter)
    conicalProjector.setComptonPointsOfInteraction(point_compton)
    # conicalProjector.transformIntoPositivePoints()
    conicalProjector.amplifyPointsToGPUCoordinateSystem()

    conicalProjector.setAngles(angles_compton)
    conicalProjector.setEnergies(energies_compton)
    conicalProjector.setVectorComptonScatter()
    # conicalProjector.setRangeLim(np.array([0, 200], dtype=np.float32), np.array([0, 200], dtype=np.float32),
    #                              np.array([-220, -100], dtype=np.float32))

    conicalProjector.setRangeLim(np.array([-100, 100], dtype=np.float32), np.array([-100, 100], dtype=np.float32),
                                 np.array([-250, -180], dtype=np.float32))
    # conicalProjector.setRangeLim()
    conicalProjector.createVectorialSpace()
    equation = conicalProjector.setConeEquation()

    import matplotlib.pyplot as plt

    # draw the conical surface
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    XX = conicalProjector._imIndexX
    YY = conicalProjector._imIndexY
    ZZ = conicalProjector._imIndexZ
    x_flat = XX.flatten()
    y_flat = YY.flatten()
    z_flat = ZZ.flatten()
    x_flat = XX
    y_flat = YY
    z_flat = ZZ
    # for i in range(point_scatter.shape[0]):
    # for i in range(point_scatter.shape[0]):
    update_im = np.ones((XX.shape[0], YY.shape[1], ZZ.shape[2]), dtype=np.float32)/point_scatter.shape[0]
    im = np.ones((XX.shape[0], YY.shape[1], ZZ.shape[2]), dtype=np.float32)/point_scatter.shape[0]
    for it in range(4):

        for i in range(point_scatter.shape[0]):
        # for i in range(500):

        # for i in range(500):
            # Ax2+By2+Cz2+Dxy+Eyz+Fxz=0
            value = equation[0, i] * x_flat ** 2 + \
                    equation[1, i] * y_flat ** 2 + \
                    equation[2, i] * z_flat ** 2 + \
                    equation[3, i] * x_flat * y_flat + \
                    equation[4, i] * x_flat * z_flat + \
                    equation[5, i] * y_flat * z_flat + \
                    equation[6, i] * x_flat + \
                    equation[7, i] * y_flat + \
                    equation[8, i] * z_flat + \
                    equation[9, i]

            cut = conicalProjector._errorConicalSurface
            cut = 2
            # plot the points that are inside the conical surface
            mask = np.abs(value) < cut
            # number_of_points = np.sum(mask)
            sum_im = np.sum(update_im[mask])
            if sum_im!= 0:
                # update_im[mask] = 1/sum_im
                im[mask] += 1/sum_im

            if i % 100 == 0:
                print(f"Processing {i} of {point_scatter.shape[0]}")
        print(f"Sum of image: {np.sum(im)}")
        update_im *= im
        im = np.ones((XX.shape[0], YY.shape[1], ZZ.shape[2]), dtype=np.float32) / point_scatter.shape[0]

        print(f"Sum of update image: {np.sum(update_im)}")
        plt.imshow(np.sum(update_im, axis=2), cmap='jet', extent=[-100, 100, -100, 100])
        plt.colorbar()
        plt.show()


        # plt.plot(x_flat[np.abs(value) < cut],
        #              y_flat[np.abs(value) < cut],
        #              z_flat[np.abs(value) < cut],
        #              marker='o', linestyle='None', markersize=2, label='Conical Surface Error')

    plt.figure()
    plt.imshow(np.sum(update_im, axis=2), cmap='jet', extent=[-100, 100, -100, 100])
    plt.colorbar()

    plt.figure()
    plt.imshow(np.sum(update_im, axis=1), cmap='gray')

    plt.figure()
    plt.imshow(np.sum(update_im, axis=0), cmap='gray')

    print(conicalProjector)
    plt.show()

    # save im data
    file_save = r"C:\Users\pedro\OneDrive\Documentos\conical_projector.npy"
    np.save(file_save, im)
