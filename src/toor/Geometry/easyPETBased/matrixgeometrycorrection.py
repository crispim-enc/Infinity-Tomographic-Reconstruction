import os
import numpy as np


class MatrixGeometryCorrection:
    def __init__(self, operation='r', crystals_dimensions = None, crystal_matrix = None, distance_between_motors = None, detectors_distances =None, file_path = None):
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            raise FileNotFoundError

        if operation == 'w':
            if (crystals_dimensions or crystal_matrix or distance_between_motors or detectors_distances) is None:
                return
            self.real_size_crystal_dimensions = crystals_dimensions
            self.crystal_matrix = crystal_matrix
            self.distance_between_motors = distance_between_motors
            self.numberofcrystals = crystal_matrix[0] * crystal_matrix[1]
            self.detectors_distances = detectors_distances

            #directory = os.path.dirname(os.path.abspath(__file__))
            #self.file_path = '{}/calibrationdata/x_{}__y_{}'.format(directory,self.crystal_matrix[0], self.crystal_matrix[1])

            # self.real_size_crystal_dimensions = [2, 2, 30, 0.28, 0.14, 0.175, 0.14,0.175]  # mm [pitch_x, pitch_y, length_crystal(z), reflector_exterior, reflector_interior_x, reflector_interior_y]
            # self.distance_between_motors = 60  # mm
            # self.detectors_distances = [60, 0.1, 0, 60]
            self._generate_matrix()
            self._write()
        elif operation == 'r':
            self._read()

    def _write(self):
        print('writing')
        file_name = os.path.join(self.file_path, 'Geometry matrix.geo')
        np.savetxt(file_name, self.coordinates, delimiter=",")

    def _read(self):
        print('Reading GEOMETRY FiLE')
        file_name = os.path.join(self.file_path, 'Geometry matrix.geo')
        self.coordinates = np.loadtxt(file_name, delimiter=',')

    def _generate_matrix(self):
        """ all distances are calculated to the center of the FOV ---- its consider that the system only has 2 blocks
        of crystals one of each side"""
        self.coordinates = np.zeros((self.numberofcrystals*2, 3))

        for side in range(2):
            reflector_depth_between_crystals_x = self.real_size_crystal_dimensions[4 + 2 * side]
            reflector_depth_between_crystals_y = self.real_size_crystal_dimensions[5 + 2 * side]
            additional_increment_y = self.real_size_crystal_dimensions[1] / 2 + reflector_depth_between_crystals_y  # Side A --- espessura interior y [4]  Side B ---- [6]
            total_size_array_Z = self.real_size_crystal_dimensions[1]*self.crystal_matrix[0]+(2*reflector_depth_between_crystals_x)*(self.crystal_matrix[0]-1)
            if side == 0:
                crystal_2_center_fov = -(self.detectors_distances[2] + self.distance_between_motors+self.real_size_crystal_dimensions[2]/2)
            elif side == 1:
                crystal_2_center_fov = self.detectors_distances[3] - self.distance_between_motors + self.real_size_crystal_dimensions[2]/2
            for i in range(self.crystal_matrix[0]):
                for j in range(self.crystal_matrix[1]):
                    self.coordinates[i * (self.crystal_matrix[1]) + j + self.numberofcrystals * side, 0] = crystal_2_center_fov
                    self.coordinates[i*(self.crystal_matrix[1])+j+self.numberofcrystals*side,1] = (j-self.crystal_matrix[1]/2)*(self.real_size_crystal_dimensions[0]+2*reflector_depth_between_crystals_y) + additional_increment_y +self.detectors_distances[1]
                    self.coordinates[i * (self.crystal_matrix[1]) + j + self.numberofcrystals * side, 2] =i*(self.real_size_crystal_dimensions[0]+2*reflector_depth_between_crystals_x) + self.real_size_crystal_dimensions[1]/2 - total_size_array_Z/2
