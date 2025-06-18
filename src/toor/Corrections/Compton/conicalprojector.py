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
    def __init__(self):
        self._geometryType = "ConicalProjector"
        self._angles = []
        self._energies = []
        self._comptonPointsOfInteraction = None
        self._scatterPointsOfInteraction = None
        self._vectorComptonScatter = None
        self._errorConicalSurface = 1  # error of the conical surface
        self._conicalEquationOut = None
        self._conicalEquationIn = None
        self._A = None
        self._B = None
        self._C = None
        self._D = None
        self._E = None
        self._F = None
        self._imIndexX = None
        self._imIndexY = None
        self._imIndexZ = None
        self._xRangeLim = None
        self._yRangeLim = None
        self._zRangeLim = None

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
        c = np.cos(self._angles/2)
        self._A = self._vectorComptonScatter[:, 0]**2 - c**2
        self._B = self._vectorComptonScatter[:, 1]**2 - c**2
        self._C = self._vectorComptonScatter[:, 2]**2 - c**2
        self._D = 2 * self._vectorComptonScatter[:, 0] * self._vectorComptonScatter[:, 1]
        self._E = 2 * self._vectorComptonScatter[:, 0] * self._vectorComptonScatter[:, 2]
        self._F = 2 * self._vectorComptonScatter[:, 1] * self._vectorComptonScatter[:, 2]
        self._G = (-2 * self._A * self._scatterPointsOfInteraction[:, 0] - self._D *
                   self._scatterPointsOfInteraction[:, 1] - self._E * self._scatterPointsOfInteraction[:, 2])
        self._H = (-2 * self._B * self._scatterPointsOfInteraction[:, 1] -
                   self._D * self._scatterPointsOfInteraction[:, 0] - self._F * self._scatterPointsOfInteraction[:, 2])
        self._I = (-2 * self._C * self._scatterPointsOfInteraction[:, 2] -
                   self._E * self._scatterPointsOfInteraction[:, 0] - self._F * self._scatterPointsOfInteraction[:, 1])
        self._J = (self._A * self._scatterPointsOfInteraction[:, 0]**2 +
                      self._B * self._scatterPointsOfInteraction[:, 1]**2 +
                        self._C * self._scatterPointsOfInteraction[:, 2]**2 +
                        self._D * self._scatterPointsOfInteraction[:, 0] * self._scatterPointsOfInteraction[:, 1] +
                        self._E * self._scatterPointsOfInteraction[:, 0] * self._scatterPointsOfInteraction[:, 2] +
                        self._F * self._scatterPointsOfInteraction[:, 1] * self._scatterPointsOfInteraction[:, 2])


        return np.array([self._A, self._B, self._C, self._D, self._E, self._F,self._G,
                         self._H, self._I, self._J], dtype=np.float32)


    def setVectorComptonScatter(self):
        """
        Vector of the Compton scatter
        """
        self._vectorComptonScatter = self._comptonPointsOfInteraction - self._scatterPointsOfInteraction
        norm_vector = ConicalProjector.normVector(self._vectorComptonScatter)
        for coordinate in range(3):
            self._vectorComptonScatter[:, coordinate] = self._vectorComptonScatter[:, coordinate] / norm_vector

        return self._vectorComptonScatter

    @staticmethod
    def normVector(vector):
        """
        Norm of the vector
        """
        return np.sqrt(vector[:, 0]**2 + vector[:, 1]**2 + vector[: ,2]**2)

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
        self._scatterPointsOfInteraction = point
        self._scatterPointsOfInteraction = np.array(self._scatterPointsOfInteraction, dtype=np.float32)

    def setComptonPointsOfInteraction(self, point):
        """
        Set the point of interaction of the conical projector.
        """
        self._comptonPointsOfInteraction = point
        self._comptonPointsOfInteraction = np.array(self._comptonPointsOfInteraction, dtype=np.float32)

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

        self._xRangeLim = x_range_lim
        self._yRangeLim = y_range_lim
        self._zRangeLim = z_range_lim
        min_x = np.min(self._scatterPointsOfInteraction[:, 0])
        min_y = np.min(self._scatterPointsOfInteraction[:, 1])
        min_z = np.min(self._scatterPointsOfInteraction[:, 2])
        max_x = np.max(self._scatterPointsOfInteraction[:, 0])
        max_y = np.max(self._scatterPointsOfInteraction[:, 1])
        max_z = np.max(self._scatterPointsOfInteraction[:, 2])
        if x_range_lim is None:
            self._xRangeLim = [np.floor(min_x), np.ceil(max_x)]
        if y_range_lim is None:
            self._yRangeLim = [np.floor(min_y), np.ceil(max_y)]
        if z_range_lim is None:
            self._zRangeLim = [np.floor(min_z), np.ceil(max_z)]


    def transformIntoPositivePoints(self):
        """
        Transform the points into positive points
        """
        for coor in range(3):
            abs_min_min = np.abs(np.min([np.min(self.pointCenterList[:, coor]),np.min(self.pointCorner1List[:, coor]), np.min(self.pointCorner2List[:, coor]),
                                np.min(self.pointCorner3List[:, coor]), np.min(self.pointCorner4List[:, coor])]))

            # abs_min_min = np.abs(np.min(self.pointCenterList[:, coor]))

            self.pointCorner1List[:, coor] += abs_min_min

    def amplifyPointsToGPUCoordinateSystem(self):
        """
        Amplify the points to the GPU coordinate system acorddingly to the voxel size
        :return:
        """
        for coor in range(3):
            self.pointCenterList[:, coor] = (self.pointCenterList[:, coor]/self.voxelSize[coor]).astype(np.float32)

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



        self._numberOfPixelsX = int(np.ceil(self._xRangeLim[1]-self._xRangeLim[0]))
        self._numberOfPixelsY = int(np.ceil(self._yRangeLim[1]-self._yRangeLim[0]))
        self._numberOfPixelsZ = int(np.ceil(self._zRangeLim[1]-self._zRangeLim[0]))

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

    conicalProjector = ConicalProjector()
    # x^2 + y^2−k2z2−2   x0x−2    y0y + 2    k2z0z + (x02+y02−k2z0) = 0

    point_scatter = np.array([[0, 0, 0], [20,3,10]], dtype=np.float32)
    point_compton = np.array([[5, 2, 2], [10,10,10]], dtype=np.float32)
    angles_compton = np.array([90, 10], dtype=np.float32)
    energies_compton = np.array([511, 511], dtype=np.float32)

    file_path = r"C:\Users\pedro\OneDrive\Documentos\2Sources_openAngle_allEvents.txt"
    data = np.loadtxt(file_path, delimiter=' ', skiprows=1)
    point_scatter = data[:, 3:6]
    point_compton = data[:, 0:3]
    angles_compton = data[:, 6]
    # remove larger than 100 degrees angles
    point_scatter = point_scatter[angles_compton < 100]
    point_compton = point_compton[angles_compton < 100]
    angles_compton = angles_compton[angles_compton < 100]

    conicalProjector.setScatterPointsOfInteraction(point_scatter)
    conicalProjector.setComptonPointsOfInteraction(point_compton)
    conicalProjector.setAngles(angles_compton)
    conicalProjector.setEnergies(energies_compton)
    conicalProjector.setVectorComptonScatter()
    # conicalProjector.setRangeLim(np.array([-20, 20], dtype=np.float32),np.array([-20, 20], dtype=np.float32), np.array([-20, 20], dtype=np.float32))
    conicalProjector.setRangeLim()
    conicalProjector.createVectorialSpace()
    equation = conicalProjector.setConeEquation()

    import matplotlib.pyplot as plt
    #draw the conical surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    XX = conicalProjector._imIndexX
    YY = conicalProjector._imIndexY
    ZZ = conicalProjector._imIndexZ
    x_flat = XX.flatten()
    y_flat = YY.flatten()
    z_flat = ZZ.flatten()
    x_flat =XX
    y_flat = YY
    z_flat = ZZ
    # for i in range(point_scatter.shape[0]):
    # for i in range(point_scatter.shape[0]):
    im = np.zeros((XX.shape[0], YY.shape[1], ZZ.shape[2]), dtype=np.float32)
    for i in range(point_scatter.shape[0]):
    # for i in range(100):
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

        # plot the points that are inside the conical surface
        im += np.abs(value) < conicalProjector._errorConicalSurface

        # plt.plot(x_flat[np.abs(value) < conicalProjector._errorConicalSurface],
        #              y_flat[np.abs(value) < conicalProjector._errorConicalSurface],
        #              z_flat[np.abs(value) < conicalProjector._errorConicalSurface],
        #              marker='o', linestyle='None', markersize=2, label='Conical Surface Error')

    plt.figure()
    plt.imshow(np.mean(im, axis=2), cmap='gray', extent=(conicalProjector._xRangeLim[0], conicalProjector._xRangeLim[1],
                                                   conicalProjector._yRangeLim[0], conicalProjector._yRangeLim[1]))
    print(conicalProjector)
    plt.show()

