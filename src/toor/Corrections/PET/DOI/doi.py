# *******************************************************
# * FILE: doi.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

import numpy as np
import matplotlib.pyplot as plt
import os
# from toor.Geometry.matrixgeometrycorrection import MatrixGeometryCorrection


class AdaptativeDOIMapping:
    def __init__(self, listMode=None):
        self.main_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                                      "dataFiles")
        self.m_values = None
        self.b_values = None
        self.max_D = None
        self.inflex_points_x = None
        self.linear_attenuation = None
        self.listMode = listMode
        self.m_values_listMode = None
        self.b_values_listMode = None
        self.max_D_listMode = None
        self.inflex_points_x_listMode = None
        self.linear_attenuation_listMode = None
        self.m_values_at = None
        self.m_values_at_listMode = None
        self.b_values_at_listMode = None

    def load_doi_files(self):
        self.m_values = np.load(os.path.join(self.main_path, "m_values.npy"))
        self.m_values_at = np.load(os.path.join(self.main_path, "m_values_at.npy"))
        self.b_values = np.load(os.path.join(self.main_path, "b_values.npy"))
        self.b_values_at = np.load(os.path.join(self.main_path, "b_values_at.npy"))
        self.max_D = np.load(os.path.join(self.main_path, "max_D.npy"))
        self.inflex_points_x = np.load(os.path.join(self.main_path, "inflex_points_x.npy"))
        self.linear_attenuation = np.load(os.path.join(self.main_path, "linear_attenuation.npy"))

    def generate_listmode_doi_values(self):
        vindex = np.array(((self.listMode[:, 2] - 1) * (self.listMode[:, 3] - 1) + (self.listMode[:, 3]) - 1)).astype(
            int)
        number_of_events = len(self.listMode)
        self.m_values_listMode = self.m_values[vindex]
        self.m_values_listMode = np.ascontiguousarray(
            np.reshape(self.m_values_listMode, (self.m_values_listMode.shape[0] * self.m_values_listMode.shape[1])),
            dtype=np.float32)
        self.m_values_at_listMode = self.m_values_at[vindex]
        self.m_values_at_listMode = np.ascontiguousarray(self.m_values_at_listMode, dtype=np.float32)
        self.b_values_listMode = self.b_values[vindex]
        self.b_values_listMode = np.ascontiguousarray(
            np.reshape(self.b_values_listMode, (self.b_values_listMode.shape[0] * self.b_values_listMode.shape[1])),
            dtype=np.float32)
        self.b_values_at_listMode = self.b_values_at[vindex]
        self.b_values_at_listMode = np.ascontiguousarray(self.b_values_at_listMode, dtype=np.float32)
        self.max_D_listMode = self.max_D[vindex]
        self.max_D_listMode = np.ascontiguousarray(self.max_D_listMode, dtype=np.float32)
        self.inflex_points_x_listMode = self.inflex_points_x[vindex]
        self.inflex_points_x_listMode = np.ascontiguousarray(np.reshape(self.inflex_points_x_listMode, (
                    self.inflex_points_x_listMode.shape[0] * self.inflex_points_x_listMode.shape[1])), dtype=np.float32)
        self.linear_attenuation_crystal_A_listMode = self.linear_attenuation[
            np.array(self.listMode[:, 0] - np.min(self.linear_attenuation[:, 0])).astype(int)]
        self.linear_attenuation_crystal_A_listMode = np.ascontiguousarray(
            self.linear_attenuation_crystal_A_listMode[:, 1], dtype=np.float32)
        self.linear_attenuation_crystal_B_listMode = self.linear_attenuation[
            np.array(self.listMode[:, 0] - np.min(self.linear_attenuation[:, 0])).astype(int)]
        self.linear_attenuation_crystal_B_listMode = np.ascontiguousarray(
            self.linear_attenuation_crystal_B_listMode[:, 1],
            dtype=np.float32)


## Data C:/Users/pedro/Downloads/Bright_and_fast_scintillations_of_an_inorganic_hal.pdf
if __name__ == "__main__":
    attenuation_coeff = .0869  # mm-1
    crystal_shape = np.array([20, 4, 4])
    detectors_arrays = [16, 2]
    crystals_centers = np.array([60, 0, 0])
    crystal_face_center = np.copy(crystals_centers)
    crystal_face_center[0] = crystal_face_center[0] - crystal_shape[0] / 2
    vertices_crystal = crystals_centers + crystal_shape / 2
    coincidence_point = np.array([0, 0, 0])

    vector_a = crystals_centers - vertices_crystal
    vector_a = np.tile(vector_a, (2, 1))
    vector_b = coincidence_point - vertices_crystal
    vector_b = np.tile(vector_b, (2, 1))
    nf_vector_a = (np.sqrt(vector_a[:, 0] ** 2 + vector_a[:, 1] ** 2 + vector_a[:, 2] ** 2))
    nf_vector_b = (np.sqrt(vector_b[:, 0] ** 2 + vector_b[:, 1] ** 2 + vector_b[:, 2] ** 2))
    for coor in range(3):
        vector_a[:, coor] = vector_a[:, coor] / nf_vector_a
        vector_b[:, coor] = vector_b[:, coor] / nf_vector_b
    angle = np.arccos(
        vector_a[:, 0] * vector_b[:, 0] + vector_a[:, 1] * vector_b[:, 1] + vector_a[:, 2] * vector_b[:, 2])
    print(np.rad2deg(angle))
    # crystals_positions = MatrixGeometryCorrection(detectors_arrays)

    angle_limts = np.rad2deg(np.arctan(np.array([45., -1., -1.]) - coincidence_point))

    angle_emission = np.deg2rad(45)
    angle = np.deg2rad(np.arange(-45, 45, 0.5))
    angle = np.deg2rad(30)

    x1 = crystal_shape[1] * np.cos(angle)
    x2 = crystal_shape[0] * np.sin(angle)
    x3 = (crystal_shape[0] * np.sin(angle) + crystal_shape[1] * np.cos(angle))
    x = np.arange(0, x3, 0.1)
    # d = np.ones(x.shape)*crystal_shape[0]/(np.cos(angle))

    d = x / (np.sin(angle) * np.cos(angle))
    # d =

    # d = np.min([crystal_shape[1]/(np.cos(angle)),crystal_shape[0]/(np.sin(angle))], axis=0)*x
    # d = x*np.sin(angle)/np.cos(angle)
    d[x >= x1] = np.min([crystal_shape[0] / (np.cos(angle)), crystal_shape[1] / (np.sin(angle))], axis=0)
    d[x >= x2] = (x3 - x[x >= x2]) / (np.sin(angle) * np.cos(angle))
    # d[x>=x3] = 0
    d_attenuation = x / (np.sin(angle) * np.cos(angle)) - d
    d_attenuation[x <= x1] = 0
    # d_attenuation[x>x3] =0

    print(d)
    print(d_attenuation)
    # d_attenuation = 0

    probability = (1 - np.exp(-attenuation_coeff * d)) * np.exp(-attenuation_coeff * d_attenuation)
    d_attenuation = 0
    probability_with_no_attenuation = (1 - np.exp(-attenuation_coeff * d)) * np.exp(-attenuation_coeff * d_attenuation)
    distribution_crystal = np.ones((int(np.ceil(crystal_shape[0] / 0.1)), int(np.ceil(crystal_shape[1] / 0.1))))

    x_range = np.arange(0, crystal_shape[0], 0.1)
    y_range = np.arange(0, crystal_shape[1], 0.1)

    # Create 3 EMPTY arrays with images size.
    im_index_x = np.empty((int(np.ceil(crystal_shape[0] / 0.1)), int(np.ceil(crystal_shape[1] / 0.1))))
    im_index_y = np.empty((int(np.ceil(crystal_shape[0] / 0.1)), int(np.ceil(crystal_shape[1] / 0.1))))

    # Repeat values in one direction. Like x_axis only grows in x_axis (0, 1, 2 ... number of pixels)
    # but repeat these values on y an z axis
    im_index_x[:] = x_range[..., None]
    im_index_y[:] = y_range[None, ...]

    rotation_matrix = np.array([[np.sin(angle), np.cos(angle)],
                                [np.cos(angle), -np.sin(angle)]])
    im_x = im_index_x * rotation_matrix[0, 0] + im_index_y * rotation_matrix[1, 0]
    im_y = im_index_x * rotation_matrix[0, 1] + im_index_y * rotation_matrix[1, 1]

    d_im = im_x / (np.sin(angle) * np.cos(angle))
    d_im[im_x >= x1] = np.min([crystal_shape[0] / (np.cos(angle)), crystal_shape[1] / (np.sin(angle))], axis=0)
    d_im[im_x >= x2] = (x3 - im_x[im_x >= x2]) / (np.sin(angle) * np.cos(angle))
    d_im_attenuation = im_x / (np.sin(angle) * np.cos(angle)) - d_im
    d_im_attenuation[im_x <= x1] = 0

    p = (1 - np.exp(-attenuation_coeff * d_im)) * np.exp(-attenuation_coeff * d_im_attenuation)
    x -= np.max(x) / 2

    plt.figure(1)
    plt.imshow(p, "jet", interpolation=None, origin="lower")

    plt.figure(2)
    plt.plot(x, probability, '--', linewidth=2, markersize=12, label='with attenuation')
    plt.plot(x, probability_with_no_attenuation, '--', linewidth=2, markersize=12, label='without attenuation')
    plt.xlabel("Distance crystal (mm)")
    plt.ylabel("Probability")
    plt.title("Angle {}º".format(np.around(np.rad2deg(angle))))
    plt.legend()
    plt.show()
