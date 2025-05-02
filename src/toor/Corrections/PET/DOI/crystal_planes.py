import numpy as np


class CrystalPlanes:
    def __init__(self, parametric_coordinates_object=None, pixelSizeXY=None, pixelSizeXYZ=None):
        self.cent_p1 = parametric_coordinates_object.crystal_centerA

        self.arestA1 = parametric_coordinates_object.center_frontal_face_sideA
        self.arestA2 = parametric_coordinates_object.center_left_face_sideA
        self.arestA3 = parametric_coordinates_object.center_bottom_face_sideA
        self.cent_p2 = None
        self.arestB1 = None
        self.arestB2 = None
        self.arestB3 = None
        self.plane_centerA1 = None
        self.plane_centerB1 = None
        self.plane_centerC1 = None
        self.plane_centerA2 = None
        self.plane_centerB2 = None
        self.plane_centerC2 = None
        self.pixelXY = pixelSizeXY
        self.pixelXYZ = pixelSizeXYZ
        self.pixel_array = np.array([[pixelSizeXY], [pixelSizeXY], [pixelSizeXYZ]], dtype=np.float32)
        self.distance_planeA1crystral = None
        self.distance_planeB1crystral = None
        self.distance_planeC1crystral = None
        self.distance_planeA2crystral = None
        self.distance_planeB2crystral = None
        self.distance_planeC2crystral = None

    def multiply_by_pixel_size(self):
        # for i in range(self.pixel_array):
        self.arestA1 = self.arestA1/self.pixel_array
        self.arestA2 = self.arestA2/self.pixel_array
        self.arestA3 = self.arestA3/self.pixel_array
        # self.arestB1 = self.arestB1/self.pixel_array
        # self.arestB2 = self.arestB2/self.pixel_array
        # self.arestB3 = self.arestB3/self.pixel_array

    def crystals_central_planes(self):
        [self.plane_centerA1,self.plane_centerB1, self.plane_centerC1] = \
            self._calculation_central_crystal_planes(self.cent_p1, self.arestA1, self.arestA2, self.arestA3)

        # [self.plane_centerA2,self.plane_centerB2, self.plane_centerC2] = \
        #     self._calculation_central_crystal_planes(self.cent_p2, self.arestA2, self.arestB2, self.arestC2)

    def _calculation_central_crystal_planes(self, cent_p1, point_1, point_2, point_3):
        cent_p1 = cent_p1.T
        point_1 = point_1.T
        point_2 = point_2.T
        point_3 = point_3.T

        v_c_1 = cent_p1 - point_1
        v_c_2 = cent_p1 - point_2
        v_c_3 = cent_p1 - point_3

        v_c_1 = self.norm_vector(v_c_1)
        v_c_2 = self.norm_vector(v_c_2)
        v_c_3 = self.norm_vector(v_c_3)

        plane_centerA = self.plane_values(v_c_1, v_c_2, cent_p1)
        plane_centerB = self.plane_values(v_c_1, v_c_3, cent_p1)
        plane_centerC = self.plane_values(v_c_2, v_c_3, cent_p1)

        return plane_centerA, plane_centerB, plane_centerC

    @staticmethod
    def norm_vector(v):
        nf = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2)
        for coor in range(3):
            v[:, coor] = v[:, coor] / nf
        return v

    @staticmethod
    def plane_values(vector_a, vector_b, p1):
        cp = np.cross(vector_a, vector_b).astype(np.float32)
        d = cp[:, 0] * p1[:, 0] \
            + cp[:, 1] * p1[:, 1] \
            + cp[:, 2] * p1[:, 2]

        return np.array([cp[:, 0], cp[:, 1], cp[:, 2], d])
