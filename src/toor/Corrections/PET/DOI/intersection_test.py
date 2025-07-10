# *******************************************************
# * FILE: intersection_test.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

import os.path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
import csv


class PlanesCalc:
    def __init__(self, p1, p2, crystal_shape=None, number_of_detectores=None):
        """
        """
        self.crystal_shape = crystal_shape
        self.number_of_detectores = number_of_detectores

        # Points for coincidence
        self.p1_list = p1
        self.p2_list = p2
        self.p3_list = np.copy(self.p1_list)
        self.p3_list[:, 1] = self.p3_list[:, 1] + self.crystal_shape[1] / 2
        self.p4_list = self.p1_list.copy()
        self.p4_list[:, 2] = self.p4_list[:, 2] + self.crystal_shape[2] / 2
        self.p5_list = (self.p1_list + self.p2_list) / 2
        self.p7_list = self.p5_list.copy()
        self.p6_list = self.p5_list + self.p3_list - self.p1_list
        self.p7_list[:, 2] = self.p7_list[:, 2] + crystal_shape[2] / 2

        self.min_distances_to_center = None
        self.max_distances_to_center = None

    @staticmethod
    def norm_vector(v):
        nf = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2)
        for coor in range(3):
            v[:, coor] = v[:, coor] / nf
        return v

    def planes_coincidence(self):

        v1 = self.p2_list - self.p1_list
        v2 = self.p3_list - self.p1_list
        v3 = self.p4_list - self.p1_list
        v4 = self.p5_list - self.p6_list
        v5 = self.p7_list - self.p5_list

        v1 = self.norm_vector(v1)
        v2 = self.norm_vector(v2)
        v3 = self.norm_vector(v3)
        v4 = self.norm_vector(v4)
        v5 = self.norm_vector(v5)

        plane = self.plane_values(v1, v2, self.p1_list)
        plane_normal = self.plane_values(v1, v3, self.p1_list)
        plane_cf = self.plane_values(v4, v5, self.p5_list)
        return plane, plane_normal, plane_cf

    def crystal_planes(self, cent_p1):

        vertice_1 = np.copy(cent_p1)
        vertice_1[:, 0] = cent_p1[:, 0] + self.crystal_shape[0] / 2
        vertice_1[:, 1] = cent_p1[:, 1] + self.crystal_shape[1] / 2
        vertice_1[:, 2] = cent_p1[:, 2] + self.crystal_shape[2] / 2

        vertice_2 = np.copy(cent_p1)
        vertice_2[:, 0] = cent_p1[:, 0] + self.crystal_shape[0] / 2
        vertice_2[:, 1] = cent_p1[:, 1] + self.crystal_shape[1] / 2
        vertice_2[:, 2] = cent_p1[:, 2] - self.crystal_shape[2] / 2

        vertice_3 = np.copy(vertice_2)
        vertice_3[:, 1] = cent_p1[:, 1] - self.crystal_shape[1] / 2

        vertice_4 = np.copy(vertice_2)
        vertice_4[:, 0] = cent_p1[:, 0] - self.crystal_shape[0] / 2

        vertice_5 = np.copy(vertice_2)
        vertice_5[:, 1] = cent_p1[:, 1] - self.crystal_shape[1] / 2
        vertice_5[:, 0] = cent_p1[:, 0] - self.crystal_shape[0] / 2

        vertice_6 = np.copy(vertice_1)
        vertice_6[:, 0] = cent_p1[:, 0] - self.crystal_shape[0] / 2

        vertice_7 = np.copy(vertice_1)
        vertice_7[:, 1] = cent_p1[:, 1] - self.crystal_shape[1] / 2

        vertice_8 = np.copy(vertice_1)
        vertice_8[:, 0] = cent_p1[:, 0] - self.crystal_shape[0] / 2
        vertice_8[:, 1] = cent_p1[:, 1] - self.crystal_shape[1] / 2
        # vertice_6[:, 1] = cent_p1[:, 1] + crystal_shape[1] / 2

        # vertices = np.array([ vertice_4, vertice_2,   vertice_3, vertice_5, vertice_4, vertice_6, vertice_1, vertice_7,vertice_8])
        vertices = np.array([vertice_2, vertice_4, vertice_5, vertice_3,
                             vertice_2, vertice_1, vertice_6, vertice_4,
                             vertice_2, vertice_1, vertice_7, vertice_8,
                             vertice_6, vertice_4, vertice_5, vertice_8,
                             vertice_8, vertice_5, vertice_3, vertice_7
                             ])

        v1_2 = vertice_1 - vertice_2
        v1_7 = vertice_1 - vertice_7
        v2_3 = vertice_2 - vertice_3
        v2_4 = vertice_2 - vertice_4
        v6_4 = vertice_6 - vertice_4
        v6_8 = vertice_6 - vertice_8
        v7_8 = vertice_7 - vertice_8
        v7_3 = vertice_7 - vertice_3

        v1_2 = self.norm_vector(v1_2)
        v1_7 = self.norm_vector(v1_7)
        v2_3 = self.norm_vector(v2_3)
        v2_4 = self.norm_vector(v2_4)
        v6_4 = self.norm_vector(v6_4)
        v6_8 = self.norm_vector(v6_8)
        v7_8 = self.norm_vector(v7_8)
        v7_3 = self.norm_vector(v7_3)

        planeA = self.plane_values(v1_2, v2_3, vertice_1)
        planeB = self.plane_values(v1_2, v2_4, vertice_1)
        planeC = self.plane_values(v2_4, v2_3, vertice_2)
        planeD = self.plane_values(v7_8, v1_7, vertice_1)
        planeE = self.plane_values(v6_4, v6_8, vertice_6)
        planeF = self.plane_values(v7_8, v7_3, vertice_7)
        # planeG = self.plane_values(v2_4, v2_3, vertice_2)

        return planeA, planeB, planeC, planeD, planeE, planeF, vertices

    def central_crystal_planes(self, cent_p1):
        point_1 = np.copy(cent_p1)
        point_1[:, 0] = cent_p1[:, 0] + self.crystal_shape[0] / 2

        point_2 = np.copy(cent_p1)
        point_2[:, 1] = cent_p1[:, 1] + self.crystal_shape[1] / 2

        point_3 = np.copy(cent_p1)
        point_3[:, 2] = cent_p1[:, 2] + self.crystal_shape[2] / 2

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
    def plane_values(vector_a, vector_b, p1):
        """"""
        cp = np.cross(vector_a, vector_b).astype(np.float64)
        d = cp[:, 0] * p1[:, 0] \
            + cp[:, 1] * p1[:, 1] \
            + cp[:, 2] * p1[:, 2]

        return np.array([cp[:, 0], cp[:, 1], cp[:, 2], d])

    @staticmethod
    def three_plane_intersection(plane1, plane2, plane3):
        m_det = np.array([[plane1[0], plane1[1], plane1[2]],
                          [plane2[0], plane2[1], plane2[2]],
                          [plane3[0], plane3[1], plane3[2]]])

        m_x = np.array([[plane1[3], plane1[1], plane1[2]],
                        [plane2[3], plane2[1], plane2[2]],
                        [plane3[3], plane3[1], plane3[2]]])

        m_y = np.array([[plane1[0], plane1[3], plane1[2]],
                        [plane2[0], plane2[3], plane2[2]],
                        [plane3[0], plane3[3], plane3[2]]])

        m_z = np.array([[plane1[0], plane1[1], plane1[3]],
                        [plane2[0], plane2[1], plane2[3]],
                        [plane3[0], plane3[1], plane3[3]]])
        det = PlanesCalc.determinant(m_det)
        det_x = PlanesCalc.determinant(m_x)
        det_y = PlanesCalc.determinant(m_y)
        det_z = PlanesCalc.determinant(m_z)
        x = det_x
        y = det_y
        z = det_z
        x[det[:] != 0] = x[det[:] != 0] / det[det[:] != 0]
        y[det[:] != 0] = y[det[:] != 0] / det[det[:] != 0]
        z[det[:] != 0] = z[det[:] != 0] / det[det[:] != 0]
        x[det[:] == 0] = np.nan
        y[det[:] == 0] = np.nan
        z[det[:] == 0] = np.nan

        return x, y, z

    @staticmethod
    def two_plane_intersection(plane1, plane2):
        mx = -(plane2[1] * plane1[0] / (plane1[0] * plane2[1] - plane1[1] * plane2[0]) * (
                    plane1[1] * plane2[2] + plane1[2] * plane2[1]) / (plane1[0] * plane2[1]))
        bx = (plane2[1] * plane1[0] / (plane1[0] * plane2[1] - plane1[1] * plane2[0]) * (
                    plane1[3] * plane2[1] - plane1[1] * plane2[3]) / (plane1[0] * plane2[1]))
        my = mx * (-plane2[0] / plane2[1]) - plane2[2] * (plane1[0] * plane2[1] - plane1[1] * plane2[0]) * (
                    plane1[0] * plane2[1])
        by = bx * (-plane2[0] / plane2[1]) + plane2[3] / plane2[1]
        return mx, bx, my, by

    @staticmethod
    def determinant(matrix):
        a = matrix[0, 0]
        b = matrix[0, 1]
        c = matrix[0, 2]
        d = matrix[1, 0]
        e = matrix[1, 1]
        f = matrix[1, 2]
        g = matrix[2, 0]
        h = matrix[2, 1]
        i = matrix[2, 2]

        det = (a * e * i + b * f * g + c * d * h) - (a * f * h + b * d * i + c * e * g)
        return det

    @staticmethod
    def point_distance_to_plane(plane, point):
        # try:
        distance = np.abs(plane[0] * point[:, 0] + plane[1] * point[:, 1] + plane[2] * point[:, 2] - plane[3]) / \
                   np.sqrt(plane[0] ** 2 + plane[1] ** 2 + plane[2] ** 2)
        # except RuntimeWarning:
        # distance[point[:,0]==-np.inf] = -np.inf
        # distance[point[:,0]==-np.inf] = -0
        return distance

    @staticmethod
    def distance_between_points(point1, point2):
        if point1.shape[1] == 4:
            distance = np.sqrt((point1[:, 0] - point2[:, 0]) ** 2 + (point1[:, 1] - point2[:, 1]) ** 2 + (
                        point1[:, 2] - point2[:, 2]) ** 2)
        elif point1.shape[0] == 4:
            distance = np.sqrt((point1[0, :] - point2[0, :]) ** 2 + (point1[1, :] - point2[1, :]) ** 2 + (
                    point1[2, :] - point2[2, :]) ** 2)
        # distance[point1[0,:]==np.nan] = 0
        # distance[distance == -np.inf] = 0
        return distance

    @staticmethod
    def parallel_plane(plane, distance):
        plane_p = np.copy(plane)
        plane_p[3] += distance * np.sqrt(plane[0] ** 2 + plane[1] ** 2 + plane[2] ** 2)
        return plane_p

    def acceptable_region(self, center_plane_face, vertices):
        self.min_distances_to_center = self.point_distance_to_plane(center_plane_face, vertices[0])
        self.max_distances_to_center = self.point_distance_to_plane(center_plane_face, vertices[1])


class DetectorArrayGeometryTest:
    def __init__(self, crystal_shape=None, number_of_detector=None):
        self.crystal_shape = crystal_shape
        self.number_of_detector = number_of_detector
        self.n_detectors = self.number_of_detector[0] * self.number_of_detector[1]
        crystals_centroids_init = np.zeros((self.n_detectors, 3))

        crystals_centroids_init[::2, 1] = 1.175
        crystals_centroids_init[1::2, 1] = -1.175
        crystals_centroids_init[:, 0] = 0
        crystals_centroids_init[::2, 2] = np.arange(0, self.n_detectors, 2)
        # crystals_centroids_init[::2, 2] += (crystals_centroids_init[::2, 2] - 1) * 0.28
        crystals_centroids_init[1::2, 2] = np.arange(0, self.n_detectors, 2)
        # crystals_centroids_init[1::2, 2] += (crystals_centroids_init[1::2, 2] - 1) * 0.28

        crystals_centroids_end = np.zeros((self.n_detectors, 3))
        crystals_centroids_end[::2, 1] = 1.175
        crystals_centroids_end[1::2, 1] = -1.175
        crystals_centroids_end[:, 0] = 90
        crystals_centroids_end[::2, 2] = np.arange(0, self.n_detectors, 2)
        # crystals_centroids_end[::2, 2] +=(crystals_centroids_end[::2, 2]-1)*0.28
        crystals_centroids_end[1::2, 2] = np.arange(0, self.n_detectors, 2)
        # crystals_centroids_end[1::2, 2] +=(crystals_centroids_end[1::2, 2]-1)*0.28
        self.crystals_centroids_init = crystals_centroids_init
        self.crystals_centroids_end = crystals_centroids_end


class PlotIntersectionData:
    def __init__(self, vertices=None, vertices_end=None, plane1=None, plane2=None, plane3=None):
        """

        """
        self.fig = plt.figure()
        self.ax = plt.axes(projection='3d')
        # ax = plt.gca(projection='3d')
        self.ax._axis3don = False
        self.crystal_active = 14
        self.vertices = vertices
        self.vertices_end = vertices_end
        self.plane1 = plane1
        self.plane2 = plane2
        self.plane3 = plane3
        self.intersection_line = PlanesCalc.two_plane_intersection(plane1, plane2)
        self.intersection_line = np.array(self.intersection_line)
        # print(results)

    def design_planes_coincidence(self):
        """

        """
        plane1 = self.plane1
        plane2 = self.plane2
        vertices = self.vertices
        vertices_end = self.vertices_end
        min_x = np.min(vertices[:, self.crystal_active, 0])
        max_x = np.max(vertices_end[:, self.crystal_active, 0])
        minmax = np.array([min_x, max_x])
        min_x = np.min(minmax)
        max_x = np.max(minmax)
        min_y = np.min(vertices[:, self.crystal_active, 1])
        max_y = np.max(vertices_end[:, self.crystal_active, 1])
        minmax = np.array([min_y, max_y])
        min_y = np.min(minmax)
        max_y = np.max(minmax)
        min_z = np.min(vertices[:, self.crystal_active, 2])
        max_z = np.max(vertices_end[:, self.crystal_active, 2])
        minmax = np.array([min_z, max_z])
        min_z = np.min(minmax)
        max_z = np.max(minmax)
        xx_plane1, yy_plane1 = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y))
        xx_plane2, zz_plane2 = np.meshgrid(np.arange(min_x - 1, max_x + 1), np.arange(min_y - 1, max_y + 1))

        z_plane1 = -(plane1[0, self.crystal_active] * xx_plane1 + plane1[1, self.crystal_active] * yy_plane1 - plane1[
            3, self.crystal_active]) / (plane1[2, self.crystal_active])

        y_plane2 = -(plane2[0, self.crystal_active] * xx_plane2 + plane2[2, self.crystal_active] * zz_plane2 - plane2[
            3, self.crystal_active]) / (plane2[1, self.crystal_active])

        # self.ax.plot_surface(xx_plane1, yy_plane1, z_plane1, alpha=0.5)
        z = np.arange(min_z, max_z + 1)
        x = self.intersection_line[0, self.crystal_active] * z + self.intersection_line[1, self.crystal_active]
        y = self.intersection_line[2, self.crystal_active] * z + self.intersection_line[3, self.crystal_active]
        self.ax.plot3D(x, y, z, linewidth=2)
        # self.ax.plot_surface(xx_plane2, y_plane2, zz_plane2, alpha=0.5)

    def active_crystal_vertices(self):
        """

        """
        vertices = self.vertices
        vertices_end = self.vertices_end
        vertice_active = self.crystal_active
        self.ax.scatter3D(vertices[:, vertice_active, 0], vertices[:, vertice_active, 1],
                          vertices[:, vertice_active, 2])
        self.ax.scatter3D(vertices_end[:, vertice_active, 0], vertices_end[:, vertice_active, 1],
                          vertices_end[:, vertice_active, 2])
        # self.ax.set_xlim(np.min(vertices_end[:, 0, 0:1]),np.max(vertices_end[:, 0, 0:1]))
        self.ax.set_xlim(-15, 105)
        self.ax.set_ylim(-10, 5)

    def design_active_crystal(self, i=0):
        """

        """
        i = self.crystal_active
        vertices = self.vertices
        vertices_end = self.vertices_end
        color = '#FE53BB'
        self.ax.plot3D(vertices_end[:, i, 0], vertices_end[:, i, 1], vertices_end[:, i, 2], color='#FE53BB')
        self.ax.plot3D(vertices_end[:, i - 2, 0], vertices_end[:, i - 2, 1], vertices_end[:, i - 2, 2], "--",
                       color='#FE53BB')
        # for i in range(16):
        #     if i == self.crystal_active:
        #         color = '#08F7FE'
        #         self.ax.plot3D(vertices_end[:, i, 0], vertices_end[:, i, 1], vertices_end[:, i, 2], color='#FE53BB', )
        #     else:
        #         self.ax.plot3D(vertices_end[:, i, 0], vertices_end[:, i, 1], vertices_end[:, i, 2], linestyle='dashed',
        #                        color='#08F7FE', )
        self.ax.plot3D(vertices[:, 0, 0], vertices[:, 0, 1], vertices[:, 0, 2], color='#08F7FE')
        self.ax.plot3D(vertices[:, 2, 0], vertices[:, 2, 1], vertices[:, 2, 2], "--", color='#08F7FE')

    def design_non_active_crystals(self):
        """

        """

    def design_intersection_points(self, coordinates_init, coordinates_end):
        list_crystals = [0, 1, 4]
        list_crystals_init = [4, 3, 2]
        self.ax.scatter3D(coordinates_init[list_crystals, self.crystal_active, 0],
                          coordinates_init[list_crystals, self.crystal_active, 1],
                          coordinates_init[list_crystals, self.crystal_active, 2], "o", color='#F5D300')
        self.ax.scatter3D(coordinates_end[list_crystals_init, self.crystal_active, 0],
                          coordinates_end[list_crystals_init, self.crystal_active, 1],
                          coordinates_end[list_crystals_init, self.crystal_active, 2], color='#F5D300')


if __name__ == "__main__":
    crystal_shape = [30, 2, 2]
    number_of_detectores = [32, 2]
    geometryTest = DetectorArrayGeometryTest(crystal_shape, number_of_detectores)
    crystals_centroids_init = geometryTest.crystals_centroids_init
    crystals_centroids_end = geometryTest.crystals_centroids_end
    init_cristal = 0
    final_cristal = 1
    # p1_list = np.tile(np.array([crystals_centroids_init[0]]),(final_cristal-init_cristal,1))
    p1_list = np.repeat(np.array(crystals_centroids_init[init_cristal:final_cristal]), 64, axis=0)
    # p1_list[2] = crystals_centroids_init[3]
    # p2_list = crystals_centroids_end[init_cristal:final_cristal]
    p2_list = np.tile(crystals_centroids_end[init_cristal:final_cristal], (64, 1))

    planes = PlanesCalc(p1_list, p2_list, crystal_shape=crystal_shape, number_of_detectores=number_of_detectores)

    [plane1, plane2, plane3] = planes.planes_coincidence()

    [planeA_init, planeB_init, planeC_init, planeD_init, planeE_init, planeF_init, vertices] = planes.crystal_planes(
        p1_list)

    [planeA_end, planeB_end, planeC_end, planeD_end, planeE_end, planeF_end, vertices_end] = planes.crystal_planes(p2_list)

    [planes_centralA, planes_centralB, planes_centralC] = planes.central_crystal_planes(p1_list)
    planeA_init = planes.parallel_plane(planes_centralA, crystal_shape[1] / 2)
    planeB_init = planes.parallel_plane(planes_centralA, -crystal_shape[1] / 2)
    planeC_init = planes.parallel_plane(planes_centralB, crystal_shape[2] / 2)
    planeD_init = planes.parallel_plane(planes_centralB, -crystal_shape[2] / 2)
    planeE_init = planes.parallel_plane(planes_centralC, crystal_shape[0] / 2)
    planeF_init = planes.parallel_plane(planes_centralC, -crystal_shape[0] / 2)

    planes_init = [planeA_init, planeB_init, planeC_init, planeD_init, planeE_init, planeF_init]
    planes_end = [planeA_end, planeB_end, planeC_end, planeD_end, planeE_end, planeF_end]
    # parallel planes
    maximum_parallel_plane1 = np.zeros((vertices.shape[0], vertices.shape[1]))
    maximum_parallel_plane2 = np.zeros((vertices.shape[0], vertices.shape[1]))
    minimum_parallel_plane3 = np.zeros((vertices.shape[0], vertices.shape[1]))
    maximum_parallel_plane3 = np.zeros((vertices.shape[0], vertices.shape[1]))
    for ver in range(len(vertices)):
        maximum_parallel_plane1[ver] = planes.point_distance_to_plane(plane1, vertices[ver])
        maximum_parallel_plane2[ver] = planes.point_distance_to_plane(plane2, vertices[ver])
        # minimum_parallel_plane3[ver] = planes.point_distance_to_plane(plane3, ver)
        maximum_parallel_plane3[ver] = planes.point_distance_to_plane(plane3, vertices[ver])

    maximum_parallel_plane1 = np.max(maximum_parallel_plane1, axis=0)
    maximum_parallel_plane2 = np.max(maximum_parallel_plane2, axis=0)
    minimum_parallel_plane3 = np.min(maximum_parallel_plane3, axis=0)
    maximum_parallel_plane3 = np.max(maximum_parallel_plane3, axis=0)
    # dist_y_array = np.arange(-maximum_parallel_plane1,maximum_parallel_plane1,0.1)

    # for i in range(len(dist_y_total)):
    dist_y_array = np.arange(-0.5 - np.max(maximum_parallel_plane1[:]), np.max(maximum_parallel_plane1[:] + 0.5), 0.04)
    # dist_y_array = np.array([0.1])
    dist_z_array = np.arange(-np.max(maximum_parallel_plane2[:]), np.max(maximum_parallel_plane2[:] + 0.05), 0.05)
    # dist_z_array = np.array([np.max(maximum_parallel_plane2[:])-0.4])
    dist_z_array = np.array([0])

    parallel_plane1 = np.copy(plane1)
    parallel_plane2 = np.copy(plane2)
    probability = np.zeros((len(dist_y_array) * len(dist_z_array), len(p1_list)))

    probability_no_at = np.zeros((len(dist_y_array) * len(dist_z_array), len(p1_list)))
    d_t = np.ones((len(dist_y_array) * len(dist_z_array), len(p1_list)))
    d_at_t = np.ones((len(dist_y_array) * len(dist_z_array), len(p1_list)))
    probability_2D = np.zeros((len(dist_y_array), len(dist_z_array), len(planeA_end[0])))
    d_t_2D = np.zeros((len(dist_z_array), len(dist_y_array), len(planeA_end[0])))
    d_at_t_2D = np.zeros((len(dist_z_array), len(dist_y_array), len(planeA_end[0])))

    distance_plane1point = np.zeros((len(planes_init), len(p1_list)))
    distance_plane2point = np.zeros((len(planes_init), len(p1_list)))
    distance_plane3point = np.zeros((len(planes_init), len(p1_list)))
    distance_plane4point = np.zeros((len(planes_init), len(p1_list)))
    attenuation_coeff = 0.15391619493736075  # mm-1
    main_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "dataFiles")
    att = np.loadtxt(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "dataFiles",
                                  "LYSO_photoelectric_absortion_1x.csv"), delimiter=",")

    # plt.plot(att[att[:,0]>300,0], att[att[:,0]>300,1])
    # plt.plot(att[att[:,1]<5,0], att[att[:,1]<5,1])
    att = att[att[:, 1] < 5, :]

    x_values_inter = np.arange(int(np.min(att[:, 0])), int(np.max(att[:, 0])), 1)
    atte_values_to_save = np.zeros((len(x_values_inter), 2))
    atte_values_to_save[:, 0] = x_values_inter
    atte_values_to_save[:, 1] = np.interp(x_values_inter, att[:, 0], att[:, 1])
    # np.save(os.path.join(main_path,"linear_attenuation.npy"), atte_values_to_save)
    # plt.scatter(x_values_inter, attenuation_values_interpolation)
    # plt.show()

    # att = att[2::3,:]
    att = att[10:11, :]
    probability_vs_energy = np.zeros((len(dist_y_array) * len(dist_z_array), len(p1_list), len(att)))
    en = 0

    for at in att:
        attenuation_coeff = at[1]
        y_el = 0
        for dist_y in dist_y_array:

            parallel_plane1 = planes.parallel_plane(plane1, dist_y)
            # parallel_plane1[3] += dist_y * np.sqrt(plane1[0] ** 2 + plane1[1] ** 2 + plane1[2] ** 2)
            z_el = 0

            for dist_z in dist_z_array:
                intersection_point = np.ones((4, 4, len(p1_list))) * (-np.inf)
                parallel_plane2 = planes.parallel_plane(plane2, dist_z)
                # parallel_plane2[3] = plane2[3]
                # parallel_plane2[3] += dist_z * np.sqrt(plane2[0] ** 2 + plane2[1] ** 2 + plane2[2] ** 2)

                coordinates_init = np.zeros((len(planes_init), len(p1_list), 3))
                coordinates_end = np.zeros((len(planes_init), len(p1_list), 3))

                for p in range(len(planes_init)):
                    [x, y, z] = planes.three_plane_intersection(parallel_plane1, parallel_plane2, planes_init[p])
                    coordinates_init[p, :, 0] = x
                    coordinates_init[p, :, 1] = y
                    coordinates_init[p, :, 2] = z

                    distance_plane1point[p] = np.round(
                        planes.point_distance_to_plane(planes_centralA, coordinates_init[p, :, :]), 4)
                    distance_plane2point[p] = np.round(
                        planes.point_distance_to_plane(planes_centralB, coordinates_init[p, :, :]), 4)
                    distance_plane3point[p] = np.round(planes.point_distance_to_plane(plane3, coordinates_init[p, :, :]), 4)
                    distance_plane4point[p] = np.round(
                        planes.point_distance_to_plane(planes_centralC, coordinates_init[p, :, :]), 4)
                    # if np.abs(distance_plane1point[0]) <= maximum_parallel_plane1[0] and np.abs(distance_plane2point[0]) <= maximum_parallel_plane2[0]:
                    #      print("plane 1 - point {} : {}".format(p, distance_plane1point))
                    # print("plane 2 - point {} : {}".format(p, distance_plane2point))
                    # if maximum_parallel_plane3[0] >= np.abs(distance_plane3point[0]) >= minimum_parallel_plane3[0]:
                    #     print("plane 3 - point {} : {}".format(p, distance_plane3point))

                    [x_end, y_end, z_end] = planes.three_plane_intersection(parallel_plane1, parallel_plane2, planes_end[p])
                    coordinates_end[p, :, 0] = x_end
                    coordinates_end[p, :, 1] = y_end
                    coordinates_end[p, :, 2] = z_end

                # [coordinates_init , indexes] = np.unique(np.round(coordinates_init, 4), axis=0,return_index=True)
                # distance_plane1point = distance_plane1point[indexes]
                # distance_plane2point = distance_plane2point[indexes]
                # distance_plane3point = distance_plane3point[indexes]

                condition_1 = minimum_parallel_plane3[:] == distance_plane3point[:]
                # condition_2 = (maximum_parallel_plane3[:] >= distance_plane3point[:]) & (distance_plane3point[:] >= minimum_parallel_plane3[:])
                # condition_2 = (maximum_parallel_plane3[:] >= distance_plane3point[:]) & (distance_plane3point[:] > minimum_parallel_plane3[:])
                condition_2 = (distance_plane4point[:] <= crystal_shape[0] / 2)
                condition_3 = (distance_plane1point[:] <= crystal_shape[2] / 2)
                condition_4 = (distance_plane2point[:] <= crystal_shape[1] / 2)
                # condition_4 = (distance_plane1point[:] == crystal_shape[2]/2)
                # condition_5 = (distance_plane2point[:] == crystal_shape[1]/2)
                condition = condition_1 | (condition_2 & condition_3 & condition_4)
                number_of_points_intersecting = np.count_nonzero(condition.T, axis=1)

                # print(np.count_nonzero(distance_plane3point[:] == minimum_parallel_plane3[:], axis=0))
                # print(np.count_nonzero(distance_plane2point[:] == crystal_shape[1]/2, axis=0))
                # print(np.count_nonzero(distance_plane1point[:] == crystal_shape[2]/2, axis=0))
                # print("--------------------")
                # print("minumum {}".format(minimum_parallel_plane3))
                # print("value A {}".format(np.abs(distance_plane1point[:])))
                # print("value B{}".format(np.abs(distance_plane2point[:])))
                # print("value C{}".format(np.abs(distance_plane3point[:])))
                # print("maximum {}".format(maximum_parallel_plane3))
                # coordinates_init = coordinates_init.transpose(1, 0, 2)
                # distance_plane3point = distance_plane3point.transpose(1, 0)
                coordinates_init_temp = coordinates_init.transpose()[:, condition.T]
                coordinates_init_temp = coordinates_init_temp.T
                distance_plane3point_temp = distance_plane3point.T[condition.T]

                init = 0
                end = 0
                el = 0
                for l in range(len(number_of_points_intersecting)):
                    init = end
                    end += number_of_points_intersecting[l]
                    if number_of_points_intersecting[l] != 0:
                        intersection_point[:number_of_points_intersecting[l], :3, el] = coordinates_init_temp[init:end, :]
                        intersection_point[:number_of_points_intersecting[l], 3, el] = distance_plane3point_temp[init:end]
                        el += 1

                # intersection_point=np.unique(np.round(intersection_point, 4), axis=0)
                # coordinates_init[(maximum_parallel_plane3[:] >= np.abs(distance_plane3point[:])) & (
                #             np.abs(distance_plane3point[:]) >= minimum_parallel_plane3[:]), :] = np.nan
                #
                #     coordinates_init[:,i,:]= coordinates_init[arg_sort[::-1,i],i,:]
                # distance_plane3point_temp = distance_plane3point[(distance_plane3point[:] == minimum_parallel_plane3[:]) | (distance_plane1point[:] == crystal_shape[2]/2) ]
                # intersection_point[:,np.sqrt(intersection_point[:,0,:]**2+intersection_point[:,1,:]**2+intersection_point[:,2,:]**2)==0] = np.nan
                # distance_plane3point_temp = planes.point_distance_to_plane(plane3, intersection_point)
                # intersection_point[:,3,:]=np.round(distance_plane3point_temp,4)
                indexes = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
                for ind in range(len(indexes)):
                    intersection_point[indexes[ind, 1], :,
                    intersection_point[indexes[ind, 0], 3, :] == intersection_point[indexes[ind, 1], 3, :]] = -np.inf
                # distance_plane3point_temp = np.sort(distance_plane3point_temp, axis=0)
                arg_sort = np.argsort(intersection_point[:, 3, :], axis=0)

                # print(arg_sort)
                intersection_point_f = np.copy(intersection_point)
                for x in range(arg_sort.shape[0]):
                    for y in range(arg_sort.shape[1]):
                        intersection_point[x, :, y] = intersection_point_f[arg_sort[x, y], :, y]

                # print(planes.point_distance_to_plane(plane3, intersection_point))

                d = planes.distance_between_points(intersection_point[2, :, :], intersection_point[3, :, :])

                d[d == np.inf] = 0
                d[d == -np.inf] = 0

                # d = (distance_plane3point_temp[2] - distance_plane3point_temp[1])
                # d_attenuation = (distance_plane3point_temp[1] - distance_plane3point_temp[0])

                d_attenuation = planes.distance_between_points(intersection_point[1, :, :], intersection_point[2, :, :])
                d_attenuation[d_attenuation == np.inf] = 0
                d_attenuation[d_attenuation == -np.inf] = 0
                # d_attenuation[d_attenuation == -np.inf] = 0

                # d_end = planes.distance_between_points(coordinates_end[0,:,:],coordinates_end[1,:,:])
                # d_attenuation_end = planes.distance_between_points(coordinates_end[1,:,:],coordinates_end[2,:,:])

                probability[len(dist_z_array) * y_el + z_el, :] = (1 - np.exp(-attenuation_coeff * d)) * np.exp(
                    -attenuation_coeff * d_attenuation)
                probability[len(dist_z_array) * y_el + z_el, maximum_parallel_plane1 <= np.abs(dist_y)] = 0

                # probability[len(dist_z_array)*y_el+z_el,maximum_parallel_plane2>=np.abs(dist_z)] = 0

                probability_no_at[len(dist_z_array) * y_el + z_el, :] = (1 - np.exp(-attenuation_coeff * d))
                probability_no_at[len(dist_z_array) * y_el + z_el, maximum_parallel_plane1 <= np.abs(dist_y)] = 0
                d_t[len(dist_z_array) * y_el + z_el, :] = d
                d_at_t[len(dist_z_array) * y_el + z_el, :] = d_attenuation
                d_t_2D[z_el, y_el, :] = np.nan_to_num(d)
                # d_t_2D[:, np.abs(y_el)>np.max(maximum_parallel_plane2[:]),:] = 0
                d_at_t_2D[z_el, y_el, :] = np.nan_to_num(d_attenuation)

                z_el += 1
            y_el += 1

        probability_vs_energy[:, :, en] = probability[:, :]
        en += 1

    probability_2D = (1 - np.exp(-attenuation_coeff * d_t_2D)) * np.exp(-attenuation_coeff * d_at_t_2D)
    probability_2D[probability_2D == np.nan] = 0

    probability_2D_no_at = (1 - np.exp(-attenuation_coeff * d_t_2D))
    probability_2D_no_at[probability_2D_no_at == np.nan] = 0
    fig_2 = plt.figure()
    # differ = d_t[:-1,30]-d_t[1:, 30]
    # differ = np.round(np.diff(d_t[:,:]),4)
    differ = np.round(np.diff(d_t_2D, axis=1), 4)
    differ_at = np.round(np.diff(d_at_t_2D, axis=1), 4)
    asign = np.sign(differ)
    asign_at = np.sign(differ_at)

    # signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    # # Does not detect all times the begining
    # indexes_signchange = np.where(signchange == 1)
    # print(indexes_signchange)
    # plt.scatter(dist_y_array,asign[:,0]*dist_y_array)
    # a = signchange*np.tile(dist_y_array,signchange.shape[0], axis=0)
    # b = signchange*d_t[:-1,47]
    # print(a[a!=0])
    # print(b[b!=0])
    # plt.plot(dist_y_array,d_t_2D[0,:,3968 ])
    # plt.show()
    # np.abs(np.diff(asign, axis=1))[14,:,14]*dist_y_array[:-2]
    d_t_2D_tp = np.copy(d_t_2D[0, 1:-1, :]).T
    peaks = np.abs(np.diff(asign, axis=1))
    number_of_points = np.count_nonzero(peaks, axis=1)
    peaks_loc = np.where(peaks == 2)

    peaks[peaks_loc[0], peaks_loc[1] + 1, peaks_loc[2]] = 1
    peaks[peaks_loc[0], peaks_loc[1], peaks_loc[2]] -= 1

    dist_array_multiple = np.tile(dist_y_array[1:-1], (peaks.shape[2], 1))
    dist_x_points = dist_array_multiple[peaks[0].T != 0]
    dist_x_points = np.reshape(dist_x_points, (peaks.shape[2], 4))
    dist_y_points = d_t_2D_tp[[peaks[0].T != 0]]
    dist_y_points = np.reshape(dist_y_points, (peaks.shape[2], 4))
    m_values = np.diff(dist_y_points, axis=1) / np.diff(dist_x_points, axis=1)
    m_values = np.delete(m_values, 1, axis=1)
    b_values = np.copy(m_values)
    b_values[:, 0] = - m_values[:, 0] * dist_x_points[:, 0]
    b_values[:, 1] = - m_values[:, 1] * dist_x_points[:, 3]
    inflex_points_x = dist_x_points[:, 1:3]
    max_D = dist_y_points[:, 1]
    # np.save(os.path.join(main_path,"m_values.npy"), m_values)
    # np.save(os.path.join(main_path,"b_values.npy"), b_values)
    # np.save(os.path.join(main_path,"inflex_points_x.npy"), inflex_points_x)
    # np.save(os.path.join(main_path,"max_D.npy"), max_D)

    d_t_2D_at_tp = np.copy(d_at_t_2D[0, 1:-1, :]).T
    peaks_at = np.abs(np.diff(asign_at, axis=1))
    peaks_at[peaks_at > 1] = 1
    number_of_points_at = np.count_nonzero(peaks_at, axis=1)
    peaks_at[0, 0, number_of_points_at[0] == 0] = 1
    peaks_at[0, 10, number_of_points_at[0] == 0] = 1
    peaks_at[0, -1, number_of_points_at[0] == 0] = 1
    # peaks_loc = np.where(peaks_at==0)
    #
    # peaks[peaks_loc[0], peaks_loc[1]+1, peaks_loc[2]] = 1
    # peaks[peaks_loc[0], peaks_loc[1], peaks_loc[2]] -= 1

    # dist_array_multiple = np.tile(dist_y_array[1:-1],(peaks.shape[2],1))
    dist_x_points_at = dist_array_multiple[peaks_at[0].T == 1]
    dist_x_points_at = np.reshape(dist_x_points_at, (peaks_at.shape[2], 3))
    dist_y_points_at = d_t_2D_at_tp[peaks_at[0].T == 1]
    dist_y_points_at = np.reshape(dist_y_points_at, (peaks_at.shape[2], 3))
    dist_x_points_at = dist_x_points_at[:, 0:2]
    dist_y_points_at = dist_y_points_at[:, 0:2]

    m_values_at = np.diff(dist_y_points_at, axis=1) / np.diff(dist_x_points_at, axis=1)
    # m_values = np.delete(m_values,1 ,axis=1)
    # b_values_at = np.copy(m_values_at)
    b_values_at = - m_values_at[:, 0] * dist_x_points_at[:, 0]
    # b_values_at = - m_values_at*0
    # b_values[:, 1] = - m_values[:,1]*dist_x_points[:,3]
    np.save(os.path.join(main_path, "m_values_at.npy"), m_values_at[:, 0])
    np.save(os.path.join(main_path, "b_values_at.npy"), b_values_at)

    # print(d_t_2D_tp)
    list_c = np.arange(200, 228)
    # list_c = [6,7,62,63]
    for j in list_c:
        # for i in range(differ.shape[0]):
        for i in range(1):
            # verifi = max_D[j]* np.ones(dist_y_array.shape)
            verifi_dec = m_values_at[j][0] * dist_y_array + b_values_at[j]
            # for i in range(differ.shape[1]):
            #     plt.plot(dist_y_array[:-1],differ[i,:, j], label= str(dist_z_array[i]))
            #     plt.scatter(dist_y_array[:-1],asign[i,:, j], label= str(dist_z_array[i]))
            #     plt.scatter(dist_y_array[:-1],signchange[i,:, j], label= str(dist_z_array[i]))
            #     plt.plot(dist_y_array[:-1],np.diff(d_t_2D[i,:, j]), label= str(dist_z_array[i]))
            #     plt.plot(dist_y_array,d_t_2D[i,:,j ], label= str(j))
            plt.plot(dist_y_array, d_at_t_2D[i, :, j], label=str(j))
            # plt.scatter(dist_y_array[2:],peaks_at[i,:,j ], label= str(j))
            plt.scatter(dist_x_points_at[j, :], dist_y_points_at[j, :], label=str(j))
            # plt.plot(dist_y_array,verifi, label= str(j))
            plt.plot(dist_y_array, verifi_dec, label=str(j))
            a = np.abs(np.diff(asign, axis=1))[i, :, j]
            # print("Crystal{}: {}".format(j,len(a[a!=0])))
            # plt.scatter(dist_y_array[:-2],a, label= str(dist_z_array[i]))
            # plt.plot(dist_z_array,d_t_2D[:,i,j ], label= str(dist_y_array[i]))
    # # plt.scatter(dist_y_array[:-1],signchange)
    # plt.xlabel("y_dist")
    # plt.ylabel("lenght crystal")
    # plt.ylim(0,30)
    # plt.legend()
    # probability[:,dimaximum_parallel_plane1[i]] = 0
    # fig_2 = plt.figure()
    # ax_2 = plt.axes()
    # # dist_y, dist_z, probability = np.meshgrid(dist_y_array, dist_z_array, probability[0])
    # probability_2D[probability_2D==np.nan] = 1
    # ax_2.imshow(probability_2D[:,:,15])
    #

    plt.style.use("seaborn-dark")
    # plt.style.use("seaborn-darkgrid")
    plt.style.use("dark_background")
    # plt.rc('axes', labelsize=12)
    # for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
    #     plt.rcParams[param] = '#212946'  # bluish dark greyfor param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
    #     plt.rcParams[param] = '0.9'  # very light greyax.grid(color='#2A3459')  # bluish dark grey, but slightly lighter than background
    #
    # plt.style.use("dark_background")
    # for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
    #     plt.rcParams[param] = '0.9'  # very light greyfor param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
    #     plt.rcParams[param] = '#212946'  # bluish dark grey
    #
    colors = [
        '#08F7FE',  # teal/cyan
        '#FE53BB',  # pink
        '#F5D300',  # yellow
        '#00ff41',  # matrix green
        # matrix green
        '#4361ee',  # matrix green
        '#4cc9f0',
        '#ab1725',
        '#f43d9b',
        '#f72585',  # matrix green
        '#7209b7',  # matrix green
        '#3a0ca3',
        # matrix green
    ]
    # # df = pd.DataFrame({'A': [1, 3, 9, 5, 2, 1, 1],
    # #                    'B': [4, 5, 5, 7, 9, 8, 6]})
    fig, ax = plt.subplots()

    ax.grid(color='#2A3459')
    # ax.set_xlim([ax.get_xlim()[0] - 0.2, ax.get_xlim()[1] + 0.2])  # to not have the markers cut off
    ax.set_ylim(0, 1.2 * np.max(probability_no_at[:, :]))

    label = [str(i + init_cristal + 1) for i in range((final_cristal - init_cristal))]

    # plt.plot(dist_y_array, probability[:,:], "--", label="attenuation")
    list_crystals_for_chart = [0, 1]
    # for i in range(probability_no_at.shape[1]):
    # dist_z_array *= -1
    dist_array = dist_y_array
    c = 0

    cut_z = [0]
    # for j in range(probability_2D.shape[1]):
    # # for j in cut_z:
    #     probability= probability_2D[:,j,:]
    #     probability_no_at = probability_2D_no_at[:,j,:]
    #     for i in list_crystals_for_chart:
    #         ax.plot(dist_array, probability_no_at[:,i], "--", label="DoI without attenuation\n 1 to {} crystal\n".format(label[i]), color=colors[c])
    #         ax.fill_between(x=dist_array,
    #                         y1=probability_no_at[:, i],
    #                         # y2=[0] * len(df),
    #                         color=colors[c],
    #                         alpha=0.15)
    #         ax.plot(dist_array, probability[:,i], "--", label=" DoI \n 1 to {} crystal\n".format(label[i]),color=colors[c+1])
    #         ax.fill_between(x=dist_array,
    #                         y1=probability[:,i],
    #                         # y2=[0] * len(df),
    #                         color=colors[c+1],
    #                         alpha=0.2)
    #         c +=2
    #
    # hfont = {'fontname': 'Gill Sans MT'}
    # plt.legend(prop={'family': 'Gill Sans MT'}, loc='upper left')
    # # plt.legend()
    # plt.xlabel("Distance to the center plane NCC", **hfont, size=14)
    # plt.ylabel("IDRF", **hfont, size=14)
    # plt.ylim(0,1)
    # plt.xlim(-2,2)

    # location_text = np.array([[-7.5,1.02],[-7.3,0.90],[-6.6,0.6],[-5,0.2],[-0.25,0.1]])
    # for i in range(probability_vs_energy.shape[2]):
    #     # ax.plot(dist_array, probability_no_at[:,i], "--", label="without attenuation - 1 to {} crystal".format(label[i]), color=colors[c])
    #     # ax.fill_between(x=dist_array,
    #     #                 y1=probability_no_at[:, i],
    #     #                 # y2=[0] * len(df),
    #     #                 color=colors[c],
    #     #                 alpha=0.15)
    #     ax.plot(dist_array, probability_vs_energy[:,62,i], "--", label=" {} keV".format(str(np.round(att[i,0],0))), color=colors[c+2])
    #     ax.text(location_text[i,0], location_text[i,1]," {} keV".format(str(int(np.round(att[i,0],0)))), rotation=0, color=colors[c+2], size=11,family= 'Gill Sans MT')
    #     # ax.fill_between(x=dist_array,
    #     #                 y1=probability[:,0,i],
    #     #                 # y2=[0] * len(df),
    #     #                 color=colors[c+1],
    #     #                 alpha=0.2)
    #     c +=1
    #
    # hfont = {'fontname': 'Gill Sans MT'}
    # # plt.legend(prop={'family': 'Gill Sans MT'}, loc='upper left')
    # # plt.legend()
    # plt.xlabel("Distance to the center plane CC", **hfont, size=14)
    # plt.ylabel("IDRF", **hfont, size=14)
    # plt.ylim(0,1)
    # plt.xlim(-7.5,7.5)
    # plt.title("Energy dependency in DOI estimation between 16 crystals in the axial direction", **hfont)

    # plt.xlabel("Distance to the center TOR plane XZ", **hfont, size=14)
    # plt.ylabel("Probability", **hfont, size=14)
    # plt.title("Energy dependency in DOI estimation between 16 crystals in the axial direction", **hfont)

    # fig3, ax3 = plt.subplots()
    # for i in range(d_t.shape[1]):
    #     plt.plot(dist_array,d_t[:,i],label = i)
    #     # plt.plot(dist_array, d_at_t[:, i])
    # plt.legend()
    # plt.scatter(dist_array,d_at_t[:,i])
    # plotintersection = PlotIntersectionData(vertices, vertices_end, plane1, plane2)
    # plotintersection.design_planes_coincidence()
    # plotintersection.design_active_crystal()
    # # plotintersection.active_crystal_vertices()
    # plotintersection.design_intersection_points(coordinates_init,coordinates_end)
    plt.show()

#%%
