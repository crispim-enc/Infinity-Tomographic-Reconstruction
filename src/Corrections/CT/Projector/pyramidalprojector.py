import numpy as np
# from Projector import GeneralProjector


class PyramidalProjector:
    def __init__(self, voxelSize=None, FoVAxial=45, FoVRadial=25, FoVTangencial=30, FovRadialStart=None,
                 FovRadialEnd=None, fov=45, only_fov=False):
        """
        Pyramidal Projector

        :param voxelSize: list with the voxel dimensions in mm
        :param FoVAxial: Field of view in axial direction in mm
        :param FoVRadial: Field of view in radial direction in mm
        :param FoVTangencial: Field of view in tangencial direction in mm
        """

        if voxelSize is None:
            voxelSize = [0.25, 0.25, 0.25]
        # super().__init__(voxelSize=voxelSize,FoVAxial=45, FoVRadial=25, FoVTangencial=30, FovRadialStart=FovRadialStart,
        #          FovRadialEnd=FovRadialEnd)
        self.type = "Pyramidal"
        self.voxelSize = voxelSize
        self.FoVAxial = FoVAxial
        self.FoVRadial = FoVRadial/voxelSize[0]
        self.FoVRadial_min = FovRadialStart/voxelSize[0]
        self.FoVRadial_max = FovRadialEnd/voxelSize[0]
        self.FoVTangencial = FoVTangencial
        self._only_fov = only_fov
        self.fov = fov / voxelSize[0]

        self.pointCenterList = None
        self.pointCorner1List = None
        self.pointCorner2List = None
        self.pointCorner3List = None
        self.pointCorner4List = None
        self.planes = None
        # Plane Left
        self.aLeft = None
        self.bLeft = None
        self.cLeft = None
        self.dLeft = None
        # Plane Right
        self.aRight = None
        self.bRight = None
        self.cRight = None
        self.dRight = None
        # Plane Front
        self.aFront = None
        self.bFront = None
        self.cFront = None
        self.dFront = None
        # Plane Back
        self.aBack = None
        self.bBack = None
        self.cBack = None
        self.dBack = None

        self.number_of_pixels_x = int(np.ceil(self.FoVRadial))
        self.number_of_pixels_y = int(np.ceil(self.FoVTangencial))
        self.number_of_pixels_z = int(np.ceil(self.FoVAxial))

        self.x_range_lim = None
        self.y_range_lim = None
        self.z_range_lim = None

        self._countsPerPosition = None

    @property
    def countsPerPosition(self):
        """
        Return the number of counts per position
        Valid for non ListMode data
        :return:
        """
        return self._countsPerPosition

    def setCountsPerPosition(self, countsPerPosition):
        """
        Set the number of counts per position
        Valid for non ListMode data
        :param countsPerPosition:
        :return:
        """
        self._countsPerPosition = countsPerPosition
        self._countsPerPosition = np.ascontiguousarray(self._countsPerPosition, dtype=np.int32)

    def transformIntoPositivePoints(self):
        """
        Transform the points into positive points
        """
        for coor in range(3):
            abs_min_min = np.abs(np.min([np.min(self.pointCenterList[:, coor]),np.min(self.pointCorner1List[:, coor]), np.min(self.pointCorner2List[:, coor]),
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
            self.pointCenterList[:, coor] = (self.pointCenterList[:, coor]/self.voxelSize[coor]).astype(np.float32)
            self.pointCorner1List[:, coor] = (self.pointCorner1List[:, coor]/self.voxelSize[coor]).astype(np.float32)
            self.pointCorner2List[:, coor] = (self.pointCorner2List[:, coor]/self.voxelSize[coor]).astype(np.float32)
            self.pointCorner3List[:, coor] = (self.pointCorner3List[:, coor]/self.voxelSize[coor]).astype(np.float32)
            self.pointCorner4List[:, coor] = (self.pointCorner4List[:, coor]/self.voxelSize[coor]).astype(np.float32)

    def createVectorialSpace(self):
        """
        Create the vectorial space
        Needs pointCenterList, pointCorner1List, pointCorner2List, pointCorner3List, pointCorner4List to be
        defined first
        """
        print("Creating vectorial space")
        print("Transforming points to positive points")
        self.transformIntoPositivePoints()
        print("Amplifying points to GPU coordinate system")
        self.amplifyPointsToGPUCoordinateSystem()

        # Create a int range array from 0 to number of pixels)
        # self.x_range_lim = [np.floor((np.max(x) - FoV) / 2), np.ceil((np.max(x) - FoV) / 2 + np.ceil(FoV))]
        # self.y_range_lim = [np.floor((np.max(y) - FoV) / 2), np.ceil((np.max(y) - FoV) / 2 + np.ceil(FoV))]
        # self.z_range_lim = [np.floor(np.min(z) - 1.5 * crystal_height), np.ceil(np.max(z) + 1.5 * crystal_height)]
        # self.x_range_lim = [-self.FoVRadial/2, self.FoVRadial/2]
        # self.y_range_lim = [-self.FoVTangencial/2, self.FoVTangencial/2]
        # self.z_range_lim = [-self.FoVAxial/2, self.FoVAxial/2]
        self.max_x = self.pointCorner1List[:,0].max()
        self.max_y = self.pointCorner1List[:,1].max()


        extra_pixel_x = np.abs(self.pointCorner1List[:,0].min() - self.pointCorner2List[:,0].min())
        extra_pixel_y = np.abs(self.pointCorner1List[:,1].min() - self.pointCorner4List[:,1].min())
        extra_pixel_z = np.abs(self.pointCorner1List[:,2].min() - self.pointCorner3List[:,2].min())
        if self._only_fov:
            self.x_range_lim = [np.floor((self.max_x - self.fov) / 2), np.ceil((self.max_x - self.fov) / 2 + np.ceil(self.fov))]
            self.y_range_lim = [np.floor((self.max_y - self.fov) / 2), np.ceil((self.max_y - self.fov) / 2 + np.ceil(self.fov))]
            # self.x_range_lim = [0, 60]
            # self.y_range_lim = [0, 60]
        else:
            self.x_range_lim = [np.ceil(self.pointCorner1List[:,0].min()), np.ceil(self.pointCorner1List[:,0].max())]
            self.y_range_lim = [np.ceil(self.pointCorner1List[:,1].min()), np.ceil(self.pointCorner1List[:,1].max())]
        self.z_range_lim = [np.floor(self.pointCorner3List[:,2].min()), np.ceil(self.pointCorner1List[:,2].max()+extra_pixel_z)]
        # self.z_range_lim = [0, 73/self.voxelSize[2]]

        self.number_of_pixels_x = int(np.ceil(self.x_range_lim[1]-self.x_range_lim[0]))
        self.number_of_pixels_y = int(np.ceil(self.y_range_lim[1]-self.y_range_lim[0]))
        self.number_of_pixels_z = int(np.ceil(self.z_range_lim[1]-self.z_range_lim[0]))

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

    def createPlanes(self):
        """
        Create the planes of the projector
        Defines the planes equations
        ax + by + cz + d = 0
        for each plane.
        The planes are defined by the 4 corners of the collimator and the focal point
        :return:
        """
        v1 = PyramidalProjector.calcVector(self.pointCenterList, self.pointCorner1List)
        v2 = PyramidalProjector.calcVector(self.pointCenterList, self.pointCorner2List)
        v3 = PyramidalProjector.calcVector(self.pointCenterList, self.pointCorner3List)
        v4 = PyramidalProjector.calcVector(self.pointCenterList, self.pointCorner4List)

        n1 = PyramidalProjector.norm(v1)
        n2 = PyramidalProjector.norm(v2)
        n3 = PyramidalProjector.norm(v3)
        n4 = PyramidalProjector.norm(v4)

        for coor in range(3):
            v1[:, coor] = v1[:, coor] / n1
            v2[:, coor] = v2[:, coor] / n2
            v3[:, coor] = v3[:, coor] / n3
            v4[:, coor] = v4[:, coor] / n4

        self.aLeft, self.bLeft, self.cLeft, cp = PyramidalProjector.crossProduct(v1, v2)
        self.dLeft = PyramidalProjector.calcD(self.pointCenterList, cp)

        self.aFront, self.bFront, self.cFront, cp = PyramidalProjector.crossProduct(v2, v3)
        self.dFront = PyramidalProjector.calcD(self.pointCenterList, cp)

        self.aRight, self.bRight, self.cRight, cp = PyramidalProjector.crossProduct(v3, v4)
        self.dRight = PyramidalProjector.calcD(self.pointCenterList, cp)


        self.aBack, self.bBack, self.cBack, cp = PyramidalProjector.crossProduct(v4, v1)
        self.dBack = PyramidalProjector.calcD(self.pointCenterList, cp)

        self.planes = np.array([[self.aLeft, self.bLeft, self.cLeft, self.dLeft],
                       [self.aRight, self.bRight, self.cRight, self.dRight],
                       [self.aFront, self.bFront, self.cFront, self.dFront],
                        [self.aBack, self.bBack, self.cBack, self.dBack]])

    @staticmethod
    def norm(vector):
        """
        Calculate the norm of a vector
        :param vector:
        :return:
        """
        return np.sqrt(vector[:, 0] ** 2 + vector[:, 1] ** 2 + vector[:, 2] ** 2)

    @staticmethod
    def calcVector(p1, p2):
        """
        Calculate the vector between two points
        :param p1:
        :param p2:
        :return:
        """
        return p1-p2

    @staticmethod
    def calcD(point, crossproduct):
        """
        Calculate the d value of the plane equation
        :param point:
        :param crossproduct:
        :return:
        """
        cp = crossproduct[:, 0] * point[:, 0] + crossproduct[:, 1] * point[:, 1]+crossproduct[:, 2] * point[:, 2]
        cp = np.ascontiguousarray(np.array(cp, dtype=np.float32))
        return cp

    @staticmethod
    def crossProduct(v1,v2):
        """
        Calculate the cross product of two vectors
        :param v1:
        :param v2:
        :return:
        """
        cp = np.cross(v1, v2).astype(np.float32)
        a = np.ascontiguousarray(cp[:, 0], dtype=np.float32)
        b = np.ascontiguousarray(cp[:, 1], dtype=np.float32)
        c = np.ascontiguousarray(cp[:, 2], dtype=np.float32)
        return a, b, c, cp


if __name__ == "__main__":
    from src.Geometry import HeadGeometry
    from src.Designer import GeometryDesignerObject
    spectGeometry = HeadGeometry()
    spectGeometry.calculateInitialGeometry()
    pointCenter = np.copy(spectGeometry.CZTModules[0].initialMatrix).T
    p1 = spectGeometry.collimators[0].vertex1.T
    p2 = spectGeometry.collimators[0].vertex2.T
    p3 = spectGeometry.collimators[0].vertex3.T
    p4 = spectGeometry.collimators[0].vertex4.T

    p = PyramidalProjector(FovRadialStart=0, FovRadialEnd=20, FoVRadial=20)
    p.pointCenterList = pointCenter
    p.pointCorner1List = p1
    p.pointCorner2List = p2
    p.pointCorner3List = p3
    p.pointCorner4List = p4
    p.createVectorialSpace()
    p.createPlanes()
    # gd = GeometryDesignerObject()
    # for i in range(spectGeometry.numberOfModules):
    #     Czt1 = spectGeometry.CZTModules[i]
    #
    #     gd.drawDetectors(geometryVector=[Czt1.initialMatrix[0], Czt1.initialMatrix[1], Czt1.initialMatrix[2]],
    #                      moduleNumber=0)
    # p = [
    #     pointCenter,
    #     p1,
    #     p2,
    #     p3,
    #     p4
    # ]
    # gd.makePyramid(p)
    # gd.makeAxes()
    #
    # # gd.drawPlanes(plane_values=[p.aLeft, p.bLeft, p.cLeft])
    # # gd.drawPlanes(plane_values=[p.aRight, p.bRight, p.cRight])
    # gd.renderWin.Render()
    # gd.renderInteractor.Start()

