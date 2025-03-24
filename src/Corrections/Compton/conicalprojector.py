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
        self._errorConicalSurface = 0.1
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
        self._A = 1 - np.tan(self._angles) ** 2 *(self._vectorComptonScatter[0] ** 2)
        self._B = 1 - np.tan(self._angles) ** 2 *(self._vectorComptonScatter[1] ** 2)
        self._C = -np.tan(self._angles) ** 2 *(self._vectorComptonScatter[2] ** 2)
        self._D = -2 * self._scatterPointsOfInteraction[0] * self._vectorComptonScatter[0]
        self._E = -2 * self._scatterPointsOfInteraction[1] * self._vectorComptonScatter[1]
        self._F = 2 * self._scatterPointsOfInteraction[2] * self._vectorComptonScatter[2]

        return self._A, self._B, self._C, self._D, self._E, self._F

    def setVectorComptonScatter(self):
        """
        Vector of the Compton scatter
        """
        self._vectorComptonScatter = self._comptonPointsOfInteraction - self._scatterPointsOfInteraction
        self._vectorComptonScatter /= ConicalProjector.normVector(self._vectorComptonScatter)
        return self._vectorComptonScatter

    @staticmethod
    def normVector(vector):
        """
        Norm of the vector
        """
        return np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)

    def setAngles(self, angles):
        """
        Set the angles of the conical projector.
        """
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
        if x_range_lim is None:
            self.x_range_lim = [np.ceil(self._scatterPointsOfInteraction[0]), np.ceil(self._scatterPointsOfInteraction[0])]
        if y_range_lim is None:
            self.y_range_lim = [np.ceil(self._scatterPointsOfInteraction[1]), np.ceil(self._scatterPointsOfInteraction[1])]
        if z_range_lim is None:
            z_range_lim = [np.ceil(self._scatterPointsOfInteraction[2]), np.ceil(self._scatterPointsOfInteraction[2])]

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



        self.numberOfPixelsX = int(np.ceil(self.xRangeLim[1]-self.xRangeLim[0]))
        self.numberOfPixelsY = int(np.ceil(self.yRangeLim[1]-self.yRangeLim[0]))
        self.numberOfPixelsZ = int(np.ceil(self.zRangeLim[1]-self.zRangeLim[0]))

        self._imIndexX = np.ascontiguousarray(
            np.empty((self.numberOfPixelsX, self.numberOfPixelsY, self.numberOfPixelsZ), dtype=np.int32))
        self._imIndexY = np.ascontiguousarray(
            np.empty((self.numberOfPixelsX, self.numberOfPixelsY, self.numberOfPixelsZ), dtype=np.int32))
        self._imIndexZ = np.ascontiguousarray(
            np.empty((self.numberOfPixelsX, self.numberOfPixelsY, self.numberOfPixelsZ), dtype=np.int32))
        print(f"Image Shape{self._imIndexZ.shape}")
        x_range = np.arange(self.x_range_lim[0], self.x_range_lim[1],
                            dtype=np.int32)
        y_range = np.arange(self.y_range_lim[0], self.y_range_lim[1],
                            dtype=np.int32)

        z_range = np.arange(self.z_range_lim[0], self.z_range_lim[1],
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

    point_scatter = np.array([[0, 0, 0], [5,5,0]], dtype=np.float32)
    point_compton = np.array([[0, 0, 0], [5,5,0]], dtype=np.float32)
    angles_compton = np.array([45, 45], dtype=np.float32)
    energies_compton = np.array([511, 511], dtype=np.float32)

    conicalProjector.setScatterPointsOfInteraction(point_scatter)
    conicalProjector.setComptonPointsOfInteraction(point_compton)
    conicalProjector.setAngles(angles_compton)
    conicalProjector.setEnergies(energies_compton)
    conicalProjector.setVectorComptonScatter()
    conicalProjector.setConeEquation()

    print(conicalProjector)

    import matplotlib.pyplot as plt
