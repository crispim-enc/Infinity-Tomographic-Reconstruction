#  Copyright (c) 2025. # *******************************************************
#  * FILE: $FILENAME$
#  * AUTHOR: Pedro Encarnação
#  * DATE: $CURRENT_DATE$
#  * LICENSE: Your License Name
#  *******************************************************

import numpy as np
from toor.Geometry.easyPETBased import DualRotationSystem


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
        distance_to_correct = 0
        distance_to_crystal_point = np.sqrt(np.abs((crystal_distance_to_center_fov_sideA[1]) - source_width) ** 2
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

        A00, A01, B = DualRotationSystem.applyDualRotation(top + zav, vtr, bot, self._distanceBetweenMotors)
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


class EasyCTGeometry(DualRotationSystem):
    def __init__(self, detector_moduleA=None, detector_moduleB=None, x_ray_producer=None, model="Pyramidal"):
        super().__init__(detector_moduleA=detector_moduleA, detector_moduleB=detector_moduleB)
        if detector_moduleA is None:
            raise ValueError("Detector module is not defined. Please provice a detectorModule")

        if x_ray_producer is None:
            raise ValueError("X-ray producer is not defined. Please provide a xRayProducer")

        if detector_moduleB is None:
            raise Warning("Detector module B is not defined. You are choosing a only a CT based solution."
                      "\nIf a PET/CT solution is needit define a module type for side B ")

        # self._detectorModuleA = detector_moduleA
        self._xRayProducer = x_ray_producer
        self._sourceCenter = None
        self._centerFace = None
        self._verticesA = None
        self._verticesB = None
        self._model = model

        # if model == "Pyramidal":
        #     self._corner1list = None
        #     self._corner2list = None
        #     self._corner3list = None
        #     self._corner4list = None

    # @property
    # def corner1list(self):
    #     return self._corner1list
    #
    # @property
    # def corner2list(self):
    #     return self._corner2list
    #
    # @property
    # def corner3list(self):
    #     return self._corner3list
    #
    # @property
    # def corner4list(self):
    #     return self._corner4list

    @property
    def centerFace(self):
        return self._centerFace

    @property
    def xRayProducer(self):
        return self._xRayProducer

    def evaluateInitialSourcePosition(self):
        """
        If it runned after sourcePositionAfterMovement, it will rewrite vector of possitions
        """
        self.sourcePositionAfterMovement(np.zeros(1), np.zeros(1))
        self._xRayProducer.setFocalSpotInitialPositionXYSystem(self._sourceCenter)
        # print warning if the command is not runned after sourcePositionAfterMovement

    def generateInitialCoordinatesXYSystem(self):
        """
        Generate the initial coordinates in the XY system
        """
        axial_motor_angles = np.zeros(1)
        fan_motor_angles = np.zeros(1)
        self.sourcePositionAfterMovement(axial_motor_angles, fan_motor_angles)

    def detectorSideBCoordinatesAfterMovement(self, axialMotorAngle, fanMotorAngle, uniqueIdDetectorheader=None):
        """
        Load the list mode data np.array
        """
        print("Calculating parametric positions of the center and vertices of the detector for all events...")
        crystalCenters = [self.detectorModulesSideB[0].modelHighEnergyLightDetectors[i].centroid for i in uniqueIdDetectorheader]
        crystalCenters = np.array(crystalCenters, dtype=np.float32)

        fanMotorAngle = np.deg2rad(fanMotorAngle) + np.pi
        axialMotorAngle = np.deg2rad(axialMotorAngle)
        # axialDetectorCoordinate = [geometry_file[(uniqueIdDetectorheader).astype(np.int32), i] for i in  range(3)]

        half_crystal_depth = self.detectorModulesSideB[0].modelHighEnergyLightDetectors[0].crystalSizeZ / 2
        half_crystal_height = self.detectorModulesSideB[0].modelHighEnergyLightDetectors[0].crystalSizeY / 2
        half_crystal_width = self.detectorModulesSideB[0].modelHighEnergyLightDetectors[0].crystalSizeX / 2
        # End Points - Crystal on the other side of top motor positions
        # zav = np.float32(np.arctan(crystalCenters[:,1] / (self.distanceFanMotorToDetectorModulesOnSideB +
        #                                                   half_crystal_depth)))
        zav = np.float32(np.arctan(crystalCenters[:,1] / crystalCenters[:,0]))
        vtr = np.float32((crystalCenters[:,0] ** 2 + crystalCenters[:,1]  ** 2) ** 0.5)

        self._centerFace = DualRotationSystem.applyDualRotation(fanMotorAngle + zav, vtr, axialMotorAngle,
                                                               self._distanceBetweenMotors, crystalCenters[:,2])
        print("Centroid calculated for all events...")

        vertices = [self.detectorModulesSideB[0].modelHighEnergyLightDetectors[i].vertices for i in uniqueIdDetectorheader]
        vertices = np.array(vertices, dtype=np.float32)
        self._verticesB = np.zeros((vertices.shape), dtype=np.float32)
        for i in range(vertices.shape[1]):
            angleToAdd = np.float32(np.tan(vertices[:, i, 1] / vertices[:, i, 0]))
            distanceToPointOfRotation = np.sqrt(vertices[:, i, 0] ** 2 + vertices[:, i, 1] ** 2).astype(np.float32)

            self._verticesB[:, i, :] = DualRotationSystem.applyDualRotation(fanMotorAngle+angleToAdd,
                                                                          distanceToPointOfRotation, axialMotorAngle,
                                                                          self._distanceBetweenMotors,
                                                                          vertices[:, i, 2])
            print("Vertice {} calculated for all events...".format(i))
            # self._verticesB[:, i, :] = DualRotationSystem.applyDualRotation(fanMotorAngle + angleToAdd,
            #                                                                 vtr, axialMotorAngle,
            #                                                                 self._distanceBetweenMotors,
            #                                                                 vertices[:, i, 2])

            # self._verticesB[:,i,1] += vertices[:, i, 1]
            # self._verticesB[:,i,0] += vertices[:, i, 0]
            # self._verticesB[:,i,2] += vertices[:, i, 2]
        print(self._verticesB)

        # for evaluation of the center of the face
        # angle_to_vertice = np.float32(
        #     np.arctan(half_crystal_width / (distance_crystals)))
        #
        # distance_to_vertice_pos = np.float32(
        #     ((distance_crystals) ** 2 + (crystal_distance_to_center_fov_sideB[1] + half_crystal_width) ** 2) ** 0.5)
        #
        # distance_to_vertice_neg = np.float32(
        #     ((distance_crystals) ** 2 + (crystal_distance_to_center_fov_sideB[1] - half_crystal_width) ** 2) ** 0.5)
        #
        # A00, A01, B = DualRotationSystem.dotAB(fanMotorAngle + angle_to_vertice, distance_to_vertice_pos, bot)
        # x_corner = A00 * B[0] - A01 * B[1] + self._distanceBetweenMotors * A00 * B[2]
        # y_corner = A01 * B[0] + A00 * B[1] + self._distanceBetweenMotors * A01 * B[2]
        # z_corner = crystal_distance_to_center_fov_sideB[2] + half_crystal_height
        #
        # self._corner1list = np.array([x_corner, y_corner, z_corner], dtype=np.float32).T
        #
        # A00, A01, B = DualRotationSystem.dotAB(fanMotorAngle - angle_to_vertice, distance_to_vertice_neg, bot)
        # x_corner = A00 * B[0] - A01 * B[1] + self._distanceBetweenMotors  * A00 * B[2]
        # y_corner = A01 * B[0] + A00 * B[1] + self._distanceBetweenMotors  * A01 * B[2]
        # z_corner = crystal_distance_to_center_fov_sideB[2] + half_crystal_height
        #
        # self.corner2list = np.array([x_corner, y_corner, z_corner], dtype=np.float32).T
        #
        # A00, A01, B = DualRotationSystem.dotAB(fanMotorAngle - angle_to_vertice, distance_to_vertice_neg, bot)
        # x_corner = A00 * B[0] - A01 * B[1] + distance_between_motors * A00 * B[2]
        # y_corner = A01 * B[0] + A00 * B[1] + distance_between_motors * A01 * B[2]
        # z_corner = crystal_distance_to_center_fov_sideB[2] - half_crystal_height
        #
        # self.corner3list = np.array([x_corner, y_corner, z_corner], dtype=np.float32).T
        #
        # A00, A01, B = DualRotationSystem.dotAB(fanMotorAngle + angle_to_vertice, distance_to_vertice_pos, bot)
        # x_corner = A00 * B[0] - A01 * B[1] + distance_between_motors * A00 * B[2]
        # y_corner = A01 * B[0] + A00 * B[1] + distance_between_motors * A01 * B[2]
        # z_corner = crystal_distance_to_center_fov_sideB[2] - half_crystal_height
        #
        # self.corner4list = np.array([x_corner, y_corner, z_corner], dtype=np.float32).T

    def sourcePositionAfterMovement(self, axialMotorAngle, fanMotorAngle):
        print("Calculating source position for all events detected...")
        axialMotorAngle = np.deg2rad(axialMotorAngle)
        fanMotorAngle = np.deg2rad(fanMotorAngle)
        r_a = np.float32(self._distanceBetweenMotors)

        sourceDistanceToWZOrigin = np.sign(self._xRayProducer.focalSpotInitialPositionWKSystem[0]) * np.sqrt(self._xRayProducer.focalSpotInitialPositionWKSystem[0] ** 2 + self._xRayProducer.focalSpotInitialPositionWKSystem[1] ** 2)
        angToFanPointOfRotationWZ = np.arctan(self._xRayProducer.focalSpotInitialPositionWKSystem[1]
                                              / self._xRayProducer.focalSpotInitialPositionWKSystem[0],
                                              dtype=np.float32)
        RP = np.array(
            [r_a * np.cos(axialMotorAngle), r_a * np.sin(axialMotorAngle), np.zeros(axialMotorAngle.shape[0])],
            dtype=np.float32)  # rotation point
        self._originSystemWK = RP

        if sourceDistanceToWZOrigin == 0:
            sourceCenter = np.copy(RP)
            sourceCenter[2] += self._xRayProducer.focalSpotInitialPositionWKSystem[2]
        else:
            initial_point = np.array([sourceDistanceToWZOrigin * np.cos(fanMotorAngle + angToFanPointOfRotationWZ),
                 sourceDistanceToWZOrigin * np.sin(fanMotorAngle + angToFanPointOfRotationWZ)],
                dtype=np.float32)

            sourceCenterCorrectionDualRotation = DualRotationSystem._rotatePoint(axialMotorAngle, initial_point)
            sourceCenter = np.copy(RP)
            sourceCenter[0] += sourceCenterCorrectionDualRotation[0]
            sourceCenter[1] += sourceCenterCorrectionDualRotation[1]
            sourceCenter[2] += self._xRayProducer.focalSpotInitialPositionWKSystem[2]
        self._sourceCenter = np.array([sourceCenter[0], sourceCenter[1], sourceCenter[2]], dtype=np.float32).T

    @property
    def sourceCenter(self):
        return self._sourceCenter

    @property
    def verticesA(self):
        return self._verticesA

    @property
    def verticesB(self):
        return self._verticesB

def testSourceDistance(focal_point, source_position, point_of_rotation):
    distanceToWZOrigin = np.sqrt(focal_point[0] ** 2 + focal_point[1] ** 2)
    distanceSourceToWZOrigin = np.sqrt((source_position[:,0]-point_of_rotation[:,0]) ** 2 + (source_position[:,1]-point_of_rotation[:,1]) ** 2)
    for i in range(len(distanceSourceToWZOrigin)):
        print(distanceToWZOrigin, distanceSourceToWZOrigin[i])
        # assert distanceToWZOrigin == distanceSourceToWZOrigin[i]


if __name__ == "__main__":
    from DetectionLayout.Modules import easyPETModule
    from DetectionLayout.RadiationProducer import GenericRadiativeSource
    from Designer import DeviceDesignerStandalone
    import matplotlib.pyplot as plt
    _module = easyPETModule

    xrayproducer = GenericRadiativeSource()

    newDevice = EasyCTGeometry(detector_moduleA=_module, detector_moduleB=_module, x_ray_producer=xrayproducer)

    #Set source
    newDevice.xRayProducer.setFocalSpotInitialPositionWKSystem([-2, 0, 36.2/2])
    newDevice.evaluateInitialSourcePosition()

    #Set modules Side A
    newDevice.setNumberOfDetectorModulesSideA(2)
    moduleSideA_X_translation = np.array([-15, -15], dtype=np.float32)
    moduleSideA_Y_translation = np.array([-2.175, 2.175], dtype=np.float32)
    moduleSideA_Z_translation = np.array([36.2/2, 36.2/2], dtype=np.float32)
    moduleSideA_alpha_rotation = np.array([0, 0], dtype=np.float32)
    moduleSideA_beta_rotation = np.array([0, 0], dtype=np.float32)
    moduleSideA_sigma_rotation = np.array([-90, -90], dtype=np.float32)

    for i in range(newDevice.numberOfDetectorModulesSideA):
        newDevice.detectorModulesSideA[i].setXTranslation(moduleSideA_X_translation[i])
        newDevice.detectorModulesSideA[i].setYTranslation(moduleSideA_Y_translation[i])
        newDevice.detectorModulesSideA[i].setZTranslation(moduleSideA_Z_translation[i])
        newDevice.detectorModulesSideA[i].setAlphaRotation(moduleSideA_alpha_rotation[i])
        newDevice.detectorModulesSideA[i].setBetaRotation(moduleSideA_beta_rotation[i])
        newDevice.detectorModulesSideA[i].setSigmaRotation(moduleSideA_sigma_rotation[i])

    newDevice.setNumberOfDetectorModulesSideB(2)
    moduleSideB_X_translation = np.array([75, 75], dtype=np.float32)
    moduleSideB_Y_translation = np.array([-2.175, 2.175], dtype=np.float32)
    moduleSideB_Z_translation = np.array([36.2/2, 36.2/2], dtype=np.float32)
    moduleSideB_alpha_rotation = np.array([0, 0], dtype=np.float32)
    moduleSideB_beta_rotation = np.array([0, 0], dtype=np.float32)
    moduleSideB_sigma_rotation = np.array([90, 90], dtype=np.float32)

    for i in range(newDevice.numberOfDetectorModulesSideB):
        newDevice.detectorModulesSideB[i].setXTranslation(moduleSideB_X_translation[i])
        newDevice.detectorModulesSideB[i].setYTranslation(moduleSideB_Y_translation[i])
        newDevice.detectorModulesSideB[i].setZTranslation(moduleSideB_Z_translation[i])
        newDevice.detectorModulesSideB[i].setAlphaRotation(moduleSideB_alpha_rotation[i])
        newDevice.detectorModulesSideB[i].setBetaRotation(moduleSideB_beta_rotation[i])
        newDevice.detectorModulesSideB[i].setSigmaRotation(moduleSideB_sigma_rotation[i])

    # S
    # newDevice
    newDevice.setDeviceName("EasyCT")
    newDevice.setDeviceType("CT")
    newDevice.generateInitialCoordinatesWKSystem()

    # newDevice.generateDeviceUUID()
    # newDevice.createDirectory()
    print(newDevice.deviceUUID)
    print(newDevice.deviceName)
    #plot center of rotation axial
    axial_motor_angles = np.deg2rad(np.arange(0, 360, 45))

    fan_motor_angles = np.deg2rad(np.arange(-45, 60, 15))
    # repeat the fan motor angles for each axial motor angle
    fan_motor_angles = np.repeat(fan_motor_angles, len(axial_motor_angles))
    axial_motor_angles = np.tile(axial_motor_angles, len(fan_motor_angles) // len(axial_motor_angles))
    newDevice.sourcePositionAfterMovement(axial_motor_angles, fan_motor_angles)

    plt.plot(newDevice.originSystemWZ[0], newDevice.originSystemWZ[1], 'ro', label='Origin Fan Motor')
    #plot source center
    plt.plot(newDevice.sourceCenter[:,0], newDevice.sourceCenter[:,1], 'bo', label='Source Center')
    #plot a line from the origin to the source center at fan motor angle 0
    testSourceDistance(newDevice.xRayProducer.focalSpotInitialPositionWKSystem, newDevice.sourceCenter, newDevice.originSystemWZ.T)
    index_fan_motor_angle_0 = np.where(fan_motor_angles == 0)
    source_center_fan_motor_angle_0 = newDevice.sourceCenter[index_fan_motor_angle_0]
    origin_fan_motor_angle_0 = newDevice.originSystemWZ.T[index_fan_motor_angle_0]

    # plt.plot(origin_fan_motor_angle_0[0], origin_fan_motor_angle_0[1], 'x')
    plt.plot(source_center_fan_motor_angle_0[:,0], source_center_fan_motor_angle_0[:,1], 'gx')

    plt.plot([origin_fan_motor_angle_0[:,0], source_center_fan_motor_angle_0[:,0]], [origin_fan_motor_angle_0[:,1], source_center_fan_motor_angle_0[:,1]], '-')
    plt.legend()
    plt.title("Configuration Source side of detector module A")
    plt.title("Configuration Source in front module")
    plt.show()

    designer = DeviceDesignerStandalone(device=newDevice)
    designer.addDevice()
    designer.addxRayProducerSource()
    designer.startRender()

    #