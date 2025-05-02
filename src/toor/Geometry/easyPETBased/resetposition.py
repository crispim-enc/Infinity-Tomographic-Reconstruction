class ResetParametricPointsToOrigin:
    def __init__(self, listMode=None, geometry_file=None, simulation_files=False, point_location="crystal_center",
                 crystal_height=2, crystal_width=2, crystal_depth=20, shuffle=False, FoV=45, distance_between_motors=30,
                 distance_crystals=60):
        """

        """
        self.listMode = listMode
        self.simulation_files = simulation_files
        self.point_location = point_location
        self.crystal_height = crystal_height
        self.crystal_width = crystal_width
        self.fov = FoV
        self.distance_between_motors = distance_between_motors
        self.distance_crystals = distance_crystals

        # self.geometryCorrection = MatrixGeometryCorrection()

    def farSideCoordinates(self, points_to_rotate=None):
        """

        """
        nrCrystals_per_side = 16
        distance_between_motors = 60
        half_crystal_depth = 30
        points_x = points_to_rotate[0]
        points_y = points_to_rotate[1]
        points_z = points_to_rotate[2]
        # crystal_distance_to_center_fov_sideA = [geometry_file[(self.listMode[:, 2] - 1).astype(np.int), i] for i in
        #                                         range(3)]  # em mm
        # crystal_distance_to_center_fov_sideB = [
        #     geometry_file[(self.listMode[:, 3] - 1 + nrCrystals_per_side).astype(np.int), i] for i in
        #     range(3)]  # em mm

        top = -np.deg2rad(self.listMode[5], dtype=np.float32)  # sim0ulation

        bot = np.deg2rad(self.listMode[4], dtype=np.float32)

        top = np.pi + top

        # End Points - Crystal on the other side of top motor positions
        # zav = np.float32(
        #     np.arctan(crystal_distance_to_center_fov_sideB[1] / (distance_crystals + half_crystal_depth)))
        # vtr = np.float32(
        #     ((distance_crystals + half_crystal_depth) ** 2 + crystal_distance_to_center_fov_sideB[1] ** 2) ** 0.5)
        # zav = np.zeros(crystal_distance_to_center_fov_sideA[1].shape)
        # vtr = np.zeros(crystal_distance_to_center_fov_sideA[1].shape)

        # A = np.array([[np.cos(bot), -np.sin(bot), distance_between_motors * np.cos(bot)],
        #               [np.sin(bot), np.cos(bot), distance_between_motors * np.sin(bot)],
        #               [np.zeros(len(bot)), np.zeros(len(bot)), np.ones(len(bot))]], dtype=np.float32)

        A = np.array([[np.cos(bot), -np.sin(bot), np.zeros(len(bot))],
                      [np.sin(bot), np.cos(bot), np.zeros(len(bot))],
                      [np.zeros(len(bot)), np.zeros(len(bot)), np.ones(len(bot))]], dtype=np.float32)
        # B = np.array([points_x,
        #               points_y,
        #               points_z], dtype=np.float32)

        B = points_to_rotate
        dotA_B = np.array([A[0, 0] * B[0] + A[0, 1] * B[1] + A[0, 2] * B[2],
                           A[1, 0] * B[0] + A[1, 1] * B[1] + A[1, 2] * B[2],
                           A[2, 0] * B[0] + A[2, 1] * B[1] + A[2, 2] * B[2]], dtype=np.float32)

        return dotA_B

    def angleFromPointToTopRotationCenter(self, point, midpoint, centers):
        """
        """
        v1 = ResetParametricPointsToOrigin.calcVector(point, midpoint)
        v2 = ResetParametricPointsToOrigin.calcVector(centers, midpoint)
        v1_norm = ResetParametricPointsToOrigin.norm(v1)
        v2_norm = ResetParametricPointsToOrigin.norm(v2)

        dot_product = np.sum(v1 * v2, axis=1)
        angle = np.arcos(dot_product / (v1_norm * v2_norm))
        return angle

    @staticmethod
    def norm(vector):
        """
        Calculate the norm of a vector
        :param vector:
        :return:
        """
        return np.sqrt(vector[:, 0] ** 2 + vector[:, 1] ** 2 + vector[:, 2] ** 2, dtype=np.float32)


    @staticmethod
    def calcVector(p1, p2):
        """
        Calculate the vector between two points
        :param p1:
        :param p2:
        :return:
        """
        return p1 - p2

    @staticmethod
    def calcD(point, crossproduct):
        """
        Calculate the d value of the plane equation
        :param point:
        :param crossproduct:
        :return:
        """
        cp = crossproduct[:, 0] * point[:, 0] + crossproduct[:, 1] * point[:, 1] + crossproduct[:, 2] * point[:, 2]
        cp = np.ascontiguousarray(np.array(cp, dtype=np.float32))
        return cp

    @staticmethod
    def crossProduct(v1, v2):
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
    import numpy as np

    from Geometry.easyPETBased import ResetParametricPointsToOrigin

    file_path = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\SimulacoesGATE\\EasyPET3D64\\" \
                "FOV-UniformSource\\20-December-2022_17h23_8turn_0p005s_1p80bot_0p23top_range108\\" \
                "easyPET_part0_filtered.root"
    file_path = '/home/crispim/Documentos/Simulations/easyPET_part0_copy.root'