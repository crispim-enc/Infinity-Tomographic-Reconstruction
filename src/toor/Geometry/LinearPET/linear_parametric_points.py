# *******************************************************
# * FILE: linear_parametric_points.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

import numpy as np


class SetParametricCoordinates:
    def __init__(self, listMode=None, geometry_file=None, header=None, sensivity_Matrix_calculation=False,
                 point_location="crystal_center", crystal_thick=2, crystal_depth=10, shuffle=True, FoV=45,
                 distance_between_motors=30, distance_crystals=60):

        initial_geometry_init_point = np.arange(1,33)*crystal_thick
        r_init_module = distance_between_motors
        r_end_module = -distance_between_motors
        bot = np.deg2rad(listMode[:, 1])
        y_stepper = listMode[:, 0]
        # _____________INIT_POINT____________
        A = np.array([[np.cos(bot), -np.sin(bot), r_init_module*np.cos(bot)],
                      [np.sin(bot), np.cos(bot), r_init_module*np.sin(bot)],
                      [0, 0, 1]])

        B = np.array([0,
                      y_stepper,
                      1])

        dot_vector = self.dotA_B(A, B)
        self.xi = dot_vector[0]
        self.yi = dot_vector[1]
        self.zi = initial_geometry_init_point[(listMode[:, 2]).astype(int)]

        # _____________END_POINT____________
        C = np.array([[np.cos(bot), -np.sin(bot), r_end_module*np.cos(bot)],
                      [np.sin(bot), np.cos(bot), r_end_module*np.sin(bot)],
                      [0, 0, 1]])

        D = np.array([0,
                      y_stepper,
                      1])

        dot_vector = self.dotA_B(C, D)

        self.xf = dot_vector[0]
        self.yf = dot_vector[1]
        self.zf = initial_geometry_init_point[(listMode[:, 3]).astype(int)]

        # _____________MID_POINT____________
        E = np.array([[np.cos(bot), -np.sin(bot), r_init_module*np.cos(bot)],
                      [np.sin(bot), np.cos(bot), r_init_module*np.sin(bot)],
                      [0, 0, 1]])

        F = np.array([0,
                      y_stepper+crystal_thick/2,
                      1])

        dot_vector = self.dotA_B(E, F)

        self.midpoint = np.array([np.zeros(bot.shape[0]), np.zeros(bot.shape[0]), np.zeros(bot.shape[0])], dtype=np.float32)
        self.midpoint[0] = dot_vector[0]
        self.midpoint[1] = dot_vector[1]
        self.midpoint[2] = initial_geometry_init_point[(listMode[:, 2]).astype(int)]

        # _____________FAREST_POINT____________
        G = np.array([[np.cos(bot), -np.sin(bot), 0],
                      [np.sin(bot), np.cos(bot), 0],
                      [0, 0, 1]])

        H = np.array([crystal_depth/2+r_init_module,
                      y_stepper,
                      1])

        dot_vector = self.dotA_B(G, H)

        self.farest_vertex = np.array([np.zeros(bot.shape[0]), np.zeros(bot.shape[0]), np.zeros(bot.shape[0])],
                                 dtype=np.float32)
        self.farest_vertex[0] = dot_vector[0]
        self.farest_vertex[1] = dot_vector[1]
        self.farest_vertex[2] = initial_geometry_init_point[(listMode[:, 2]).astype(int)]

        # _____________CLOSEST_POINT____________
        self.closest_vertex = self.farest_vertex
        self._transform_into_positive_values(crystal_thick=crystal_thick)
        # I = np.array([[np.cos(bot), -np.sin(bot), r_init_module * np.cos(bot)],
        #               [np.sin(bot), np.cos(bot), r_init_module * np.sin(bot)],
        #               [0, 0, 1]])
        #
        # J = np.array([-crystal_depth / 2,
        #               y_stepper,
        #               1])
        #
        # dot_vector = self.dotA_B(I, J)
        #
        # self.farest_vertex = np.array([np.zeros(bot.shape[0]), np.zeros(bot.shape[0]), np.zeros(bot.shape[0])],
        #                               dtype=np.float32)
        # self.farest_vertex = dot_vector[0]
        # self.farest_vertex = dot_vector[1]
        # self.farest_vertex = initial_geometry_init_point[(listMode[:, 2] - 1).astype(int)]


    def dotA_B(self, A, B):
        dotA1_B2 = np.array([A[0, 0] * B[0] + A[0, 1] * B[1] + A[0, 2] * B[2],
                           A[1, 0] * B[0] + A[1, 1] * B[1] + A[1, 2] * B[2],
                           A[2, 0] * B[0] + A[2, 1] * B[1] + A[2, 2] * B[2]], dtype=np.float32)

        return dotA1_B2

    def _transform_into_positive_values(self, FoV=None, crystal_thick=None):
        # self.midpoint[0] = self.midpoint[0] + FoV # np.abs(np.min(self.xf))
        # self.midpoint[1] = self.midpoint[1] + FoV # np.abs(np.min(self.yf))
        # self.xi = self.xi + FoV # np.abs(np.min(self.xf))
        # self.xf = self.xf + FoV # np.abs(np.min(self.xf))
        # self.yi = self.yi + FoV #np.abs(np.min(self.yf))
        # self.yf = self.yf + FoV #np.abs(np.min(self.yf))
        x = np.zeros((len(self.xi), 2))
        x[:, 0] = self.xi
        x[:, 1] = self.xf
        y = np.zeros((len(self.xi), 2))
        y[:, 0] = self.yi
        y[:, 1] = self.yf

        z = np.zeros((len(self.zi), 2))
        z[:, 0] = self.zi
        z[:, 1] = self.zf

        self.midpoint[0] = self.midpoint[0] + np.abs(np.min(x))
        self.midpoint[1] = self.midpoint[1] + np.abs(np.min(y))
        self.farest_vertex[0] = self.farest_vertex[0] + np.abs(np.min(x))
        self.farest_vertex[1] = self.farest_vertex[1] + np.abs(np.min(y))
        self.closest_vertex[0] += np.abs(np.min(x))
        self.closest_vertex[1] += np.abs(np.min(y))
        self.xi = self.xi + np.abs(np.min(x))
        self.xf = self.xf + np.abs(np.min(x))
        self.yi = self.yi + np.abs(np.min(y))
        self.yf = self.yf + np.abs(np.min(y))

        if np.min(z) < 0:

            self.zi = self.zi + np.abs(np.min(z)-crystal_thick/2) #+ crystal_thick/2*(np.random.rand()-0.5) # to be able to project the last half of the last crystal
            self.zf = self.zf + np.abs(np.min(z)-crystal_thick/2) #+ crystal_thick/2*(np.random.rand()-0.5)