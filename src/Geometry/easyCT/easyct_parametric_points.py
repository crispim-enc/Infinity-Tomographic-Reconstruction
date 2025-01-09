#  Copyright (c) 2025. # *******************************************************
#  * FILE: $FILENAME$
#  * AUTHOR: Pedro Encarnação
#  * DATE: $CURRENT_DATE$
#  * LICENSE: Your License Name
#  *******************************************************

import numpy as np


class SetParametricsPoints:

    def __init__(self, listMode=None, geometry_file=None, simulation_files=False, point_location="crystal_center",
                 crystal_height=2, crystal_width=2, crystal_depth=30, shuffle=False, FoV=45,
                 distance_between_motors=30,
                 distance_crystals=60, source_position=None, normalization=False, centers_for_doi=False):

        """
           v3___________________________v7
          /                           / |
         /                           /  |
        /v1_______________________v5/   |v8
        |   v4                     |   /
        |                          |  /
        |v2_______________________v6|/
        """
        if listMode is None or geometry_file is None:
            return

        if shuffle:
            print('Shuflled')
            np.random.shuffle(listMode)
        # crystal_depth =6
        self.point_location = point_location
        self.crystal_height = crystal_height
        self.crystal_width = crystal_width
        self.crystal_depth = crystal_depth
        self.source_position = source_position
        if source_position is None:
            self.source_position = [12.55, 0, 0]
        self.shuffle = shuffle
        self.FoV = FoV
        self.simulation_files = simulation_files
        self.centers_for_doi = centers_for_doi

        print('Número de eventos: {}'.format(len(listMode)))
        geometry_file = geometry_file.astype(np.float32)
        nrCrystals_per_side = int(len(geometry_file) / 2)

        if simulation_files is True:
            if normalization:
                crystal_distance_to_center_fov_sideA = [geometry_file[(listMode[:, 2] - 1).astype(np.int32), i] for
                                                        i in
                                                        range(3)]  # em mm
                crystal_distance_to_center_fov_sideB = [
                    geometry_file[(listMode[:, 3] - 1 + nrCrystals_per_side).astype(np.int32), i] for i in
                    range(3)]  # em mm

                top = -np.deg2rad(listMode[:, 1], dtype=np.float32)  # sim0ulation

                bot = np.deg2rad(listMode[:, 0], dtype=np.float32)


            else:
                crystal_distance_to_center_fov_sideA = [geometry_file[(listMode[:, 2] - 1).astype(np.int32), i] for
                                                        i in
                                                        range(3)]  # em mm
                crystal_distance_to_center_fov_sideB = [
                    geometry_file[(listMode[:, 3] - 1 + nrCrystals_per_side).astype(np.int32), i] for i in
                    range(3)]  # em mm

                top = -np.deg2rad(listMode[:, 5], dtype=np.float32)  # sim0ulation

                bot = np.deg2rad(listMode[:, 4], dtype=np.float32)

        else:
            if normalization:
                crystal_distance_to_center_fov_sideA = [geometry_file[(listMode[:, 2] - 1).astype(np.int32), i] for
                                                        i in
                                                        range(3)]  # em mm
                crystal_distance_to_center_fov_sideB = [
                    geometry_file[(listMode[:, 3] - 1 + nrCrystals_per_side).astype(np.int32), i] for i in
                    range(3)]  # em mm

                top = np.deg2rad(listMode[:, 1], dtype=np.float32)  # sim0ulation

                bot = np.deg2rad(listMode[:, 0], dtype=np.float32)
            else:
                crystal_distance_to_center_fov_sideA = [geometry_file[(listMode[:, 2] - 1).astype(np.int32), i] for
                                                        i in
                                                        range(3)]  # em mm
                crystal_distance_to_center_fov_sideB = [
                    geometry_file[(listMode[:, 3] - 1 + nrCrystals_per_side).astype(np.int32), i] for i in
                    range(3)]  # em mm

                top = np.deg2rad(listMode[:, 5], dtype=np.float32)  # real
                # top = np.deg2rad(listMode[:, 5],dtype=np.float32) # simulation

                bot = np.deg2rad(listMode[:, 4], dtype=np.float32)

        self.beginPoints(top, bot, distance_between_motors, crystal_depth, crystal_width, crystal_height,
                         crystal_distance_to_center_fov_sideA)

        self.endPoints(top, bot, crystal_distance_to_center_fov_sideB, crystal_depth, crystal_width,crystal_height,
                       distance_crystals, distance_between_motors)

    # if transform_into_positive:
    #     self._transform_into_positive_values(FoV, crystal_width)
    # if not number_of_neighbours == "Auto":
    #     self.filter_neighbours(crystal_width, number_of_neighbours)
    #
    # if recon2D:
    #     self.filter_neighbours(crystal_width, neighbours=0)

    def beginPoints(self, top, bot, distance_between_motors, crystal_depth, crystal_width, crystal_height,
                    crystal_distance_to_center_fov_sideA):
        r_a = np.float32(distance_between_motors)

        source_depth = self.source_position[0]
        source_width = self.source_position[1] # crystal_distance_to_center_fov_sideA[1] == source_width

        # ang_to_crystal_center = np.arctan(crystal_distance_to_center_fov_sideA[1] / source_depth,
        #                                   dtype=np.float32)
        #
        # point_rotation_to_center_crystal = np.sqrt(
        #     crystal_distance_to_center_fov_sideA[1] ** 2 + source_depth ** 2, dtype=np.float32)
        #
        # initial_point = np.array([point_rotation_to_center_crystal * np.cos(top + ang_to_crystal_center),
        #                           point_rotation_to_center_crystal * np.sin(top + ang_to_crystal_center)],
        #                          dtype=np.float32)
        distance_to_correct=0
        distance_to_crystal_point = np.sqrt(
            np.abs((crystal_distance_to_center_fov_sideA[1]) - source_width) ** 2
            + source_depth ** 2)
        ang_to_crystal_center = np.arctan((crystal_distance_to_center_fov_sideA[
                                                  1]-distance_to_correct - source_width * np.sign(
            crystal_distance_to_center_fov_sideA[1])) / source_depth,
                                             dtype=np.float32)

        initial_point = np.array(
            [distance_to_crystal_point * np.cos(top + ang_to_crystal_center),
             distance_to_crystal_point * np.sin(top + ang_to_crystal_center)],
            dtype=np.float32)

        central_crystal_point = self._rotate_point(bot, initial_point)

        RP = np.array([r_a * np.cos(bot), r_a * np.sin(bot), np.zeros(bot.shape[0])],
                      dtype=np.float32)  # rotation point
        self.origin_system_wz = RP


        sourceCenter = np.copy(RP)
        sourceCenter[0] += central_crystal_point[0]
        sourceCenter[1] += central_crystal_point[1]
        sourceCenter[2] += crystal_distance_to_center_fov_sideA[2]
        self.sourceCenter = np.array([sourceCenter[0], sourceCenter[1], sourceCenter[2]], dtype=np.float32).T


    def endPoints(self, top, bot, crystal_distance_to_center_fov_sideB, crystal_depth, crystal_width, crystal_height,
                  distance_crystals, distance_between_motors):
        # -------------END POINTS--------------
        top = np.pi + top
        half_crystal_depth = crystal_depth / 2
        half_crystal_height = crystal_height / 2
        half_crystal_width = crystal_width / 2
        # End Points - Crystal on the other side of top motor positions
        zav = np.float32(
            np.arctan(crystal_distance_to_center_fov_sideB[1] / (distance_crystals + half_crystal_depth)))
        vtr = np.float32(
            ((distance_crystals + half_crystal_depth) ** 2 + crystal_distance_to_center_fov_sideB[1] ** 2) ** 0.5)

        A00, A01, B = SetParametricsPoints.dotAB(top + zav, vtr, bot)
        x_corner = A00 * B[0] - A01 * B[1] + distance_between_motors * A00 * B[2]
        y_corner = A01 * B[0] + A00 * B[1] + distance_between_motors * A01 * B[2]
        z_corner = crystal_distance_to_center_fov_sideB[2] + half_crystal_height

        self.center_face = np.array([x_corner, y_corner, z_corner], dtype=np.float32).T

        angle_to_vertice =  np.float32(
            np.arctan(half_crystal_width / (distance_crystals)))

        distance_to_vertice_pos = np.float32(
            ((distance_crystals) ** 2 + (crystal_distance_to_center_fov_sideB[1] + half_crystal_width) ** 2) ** 0.5)

        distance_to_vertice_neg = np.float32(
            ((distance_crystals) ** 2 + (crystal_distance_to_center_fov_sideB[1] - half_crystal_width) ** 2) ** 0.5)


        A00, A01, B = SetParametricsPoints.dotAB(top + angle_to_vertice, distance_to_vertice_pos, bot)
        x_corner = A00 * B[0] - A01 * B[1] + distance_between_motors * A00 * B[2]
        y_corner = A01 * B[0] + A00 * B[1] + distance_between_motors * A01 * B[2]
        z_corner = crystal_distance_to_center_fov_sideB[2] + half_crystal_height

        self.corner1list = np.array([x_corner, y_corner , z_corner], dtype=np.float32).T

        A00, A01, B = SetParametricsPoints.dotAB(top - angle_to_vertice, distance_to_vertice_neg, bot)
        x_corner = A00 * B[0] - A01 * B[1] + distance_between_motors * A00 * B[2]
        y_corner = A01 * B[0] + A00 * B[1] + distance_between_motors * A01 * B[2]
        z_corner = crystal_distance_to_center_fov_sideB[2] + half_crystal_height

        self.corner2list = np.array([x_corner, y_corner, z_corner], dtype=np.float32).T

        A00, A01, B = SetParametricsPoints.dotAB(top - angle_to_vertice, distance_to_vertice_neg, bot)
        x_corner = A00 * B[0] - A01 * B[1] + distance_between_motors * A00 * B[2]
        y_corner = A01 * B[0] + A00 * B[1] + distance_between_motors * A01 * B[2]
        z_corner = crystal_distance_to_center_fov_sideB[2] - half_crystal_height

        self.corner3list = np.array([x_corner, y_corner, z_corner], dtype=np.float32).T

        A00, A01, B = SetParametricsPoints.dotAB(top + angle_to_vertice, distance_to_vertice_pos, bot)
        x_corner = A00 * B[0] - A01 * B[1] + distance_between_motors * A00 * B[2]
        y_corner = A01 * B[0] + A00 * B[1] + distance_between_motors * A01 * B[2]
        z_corner = crystal_distance_to_center_fov_sideB[2] - half_crystal_height

        self.corner4list = np.array([x_corner, y_corner, z_corner], dtype=np.float32).T

        # A = np.array([[np.cos(bot), -np.sin(bot), distance_between_motors * np.cos(bot)],
        #               [np.sin(bot), np.cos(bot), distance_between_motors * np.sin(bot)],
        #               [np.zeros(len(bot)), np.zeros(len(bot)), np.ones(len(bot))]], dtype=np.float32)
        #
        # B = np.array([np.cos(top - zav) * vtr,
        #               np.sin(top - zav) * vtr,
        #               np.ones(len(bot))], dtype=np.float32)
        #
        # dotA_B = np.array([A[0, 0] * B[0] + A[0, 1] * B[1] + A[0, 2] * B[2],
        #                    A[1, 0] * B[0] + A[1, 1] * B[1] + A[1, 2] * B[2],
        #                    A[2, 0] * B[0] + A[2, 1] * B[1] + A[2, 2] * B[2]], dtype=np.float32)

    @staticmethod
    def dotAB(angle1, distance1, angle2):
        B = np.array([np.cos(angle1) * distance1,
                      np.sin(angle1) * distance1,
                      np.ones(len(angle2))], dtype=np.float32)

        A00 = np.cos(angle2).astype(np.float32)
        A01 = np.sin(angle2).astype(np.float32)

        return A00, A01, B

    @staticmethod
    def _rotate_point(angle, initial_point):
        rotation_matrix = np.array([[np.cos(angle, dtype=np.float32), -np.sin(angle, dtype=np.float32)],
                                    [np.sin(angle, dtype=np.float32), np.cos(angle, dtype=np.float32)]],
                                   dtype=np.float32)

        return np.array([rotation_matrix[0, 0] * initial_point[0] + rotation_matrix[0, 1] * initial_point[1],
                         rotation_matrix[1, 0] * initial_point[0] + rotation_matrix[1, 1] * initial_point[1]],
                        dtype=np.float32)

    def _transform_into_positive_values(self, FoV, crystal_width):
        x = np.zeros((len(self.xi), 2), dtype=np.float32)
        x[:, 0] = self.xi
        x[:, 1] = self.xf
        y = np.zeros((len(self.xi), 2), dtype=np.float32)
        y[:, 0] = self.yi
        y[:, 1] = self.yf

        z = np.zeros((len(self.zi), 2), dtype=np.float32)
        z[:, 0] = self.zi
        z[:, 1] = self.zf

        self.midpoint[0] = self.midpoint[0] + np.abs(np.min(x))
        self.midpoint[1] = self.midpoint[1] + np.abs(np.min(y))
        self.farest_vertex[0] = self.farest_vertex[0] + np.abs(np.min(x))
        self.farest_vertex[1] = self.farest_vertex[1] + np.abs(np.min(y))
        if self.centers_for_doi:
            self.center_frontal_face_sideA[0] += np.abs(np.min(x))
            self.center_frontal_face_sideA[1] += np.abs(np.min(y))
            self.center_left_face_sideA[0] += np.abs(np.min(x))
            self.center_left_face_sideA[1] += np.abs(np.min(y))
            self.center_bottom_face_sideA[0] += np.abs(np.min(x))
            self.center_bottom_face_sideA[1] += np.abs(np.min(y))
            self.crystal_centerA[0] += np.abs(np.min(x))
            self.crystal_centerA[1] += np.abs(np.min(y))
        self.xi = self.xi + np.abs(np.min(x))
        self.xf = self.xf + np.abs(np.min(x))
        self.yi = self.yi + np.abs(np.min(y))
        self.yf = self.yf + np.abs(np.min(y))
        if np.min(z) < 0:
            self.zi = self.zi + np.abs(np.min(z) + crystal_width / 2)
            self.zf = self.zf + np.abs(np.min(z) + crystal_width / 2)
