# *******************************************************
# * FILE: CPU.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

import numpy as np
import os
from array import array
import time


class IterativeAlgorithmCPU(object):
    def __init__(self, EM_obj=None):

        self.normalization_map = EM_obj.normalization_map
        self.number_of_pixels_x = EM_obj.number_of_pixels_x
        self.number_of_pixels_y = EM_obj.number_of_pixels_y
        self.number_of_pixels_z = EM_obj.number_of_pixels_z

        self.im_final = None

    def _orthogonal_distance_single_core_single_thread(self):
        print('___CPU___')
        #
        # ------------------------
        # IMAGE RECONSTRUCTION CPU
        # ------------------------
        # number_of_events = 100

        im_final = np.ascontiguousarray(
            np.ones((self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z),
                    dtype=np.float32))

        normalization_matrix = np.ascontiguousarray(
            np.ones((self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z),
                    dtype=np.float32))

        # self.number_of_events =

        for it in range(self.number_of_iterations):
            # for sb in range(number_of_subsets):
            print(np.sum(im_final))
            adjust_value = np.zeros(([self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z]))
            for i in range(0, self.number_of_events):
                # Creates an image filled with ones
                im_t = np.ones(([self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z]))

                im_copy = np.copy(im_final)
                tec = time.time()
                # print('The equation is {0}x + {1}y + {2}z = {3}'.format(self.a[i], sb[i], c[i], d[i]))
                # print('Top: {}º ;  BOT: {}º ;IDA: {}; IDB: {}; '.format(parametric_coordinates.listMode[i, 5],
                #                                                         parametric_coordinates.listMode[i, 4],
                #                                                         parametric_coordinates.listMode[i, 3],
                #                                                         parametric_coordinates.listMode[i, 2]))
                # print('Coordinates point 1: {0}x + {1}y + {2}z '.format(xi[i], yi[i], zi[i]))
                # print('Coordinates point 2: {0}x + {1}y + {2}z '.format(xf[i], yf[i], zf[i]))
                # print('----------------------------------------')

                if i % 1000 == 0:
                    # Just for checking reconstruction process
                    print('Event: {}'.format(i))

                # For each plane equation it is calcuted the value for each image point
                # example: plane equation 46x+800y+10z=10
                # for the matrix point (1,10,25) you will get a value of 40*1+800*10+10*250-10.
                # this value will then be compared to plane A value and define the VOR
                value = self.im_index_x * self.a[i] + self.im_index_y * self.b[i] \
                        + self.im_index_z * self.c[i] - self.d[i]

                value_normal = self.im_index_x * self.a_normal[i] + self.im_index_y * self.b_normal[i] \
                               + self.im_index_z * self.c_normal[i] - self.d_normal[i]

                value_cf = self.im_index_x * self.a_cf[i] + self.im_index_y * self.b_cf[i] \
                           + self.im_index_z * self.c_cf[i] - self.d_cf[i]

                d2 = self.half_crystal_pitch_xy * np.sqrt((self.a[i] ** 2 + self.b[i] ** 2 + self.c[i] ** 2))

                d2_normal = self.half_crystal_pitch_xy * np.sqrt(
                    (self.a_normal[i] ** 2 + self.b_normal[i] ** 2 + self.c_normal[i] ** 2))

                d2_cf = (self.distance_between_array_pixel / 2) * np.sqrt((self.a_cf[i] ** 2 + self.b_cf[i] ** 2
                                                                           + self.c_cf[i] ** 2))

                # The indexes with values greater than plane A and plane C are set to zero
                # The indexes with values smaller than plane B and plane D are also set to zero
                # This creates a VOR
                # im_copy[im_t[(value < d2) & (value >= -d2) & (value_normal < d2_normal) &(value_normal >= -d2_normal) & (value_cf < d2_cf) & (value_cf >= -d2_cf)]]
                im_copy[((value > d2) | (value < -d2)) | ((value_normal > d2_normal) | (value_normal < -d2_normal)) | (
                        (value_cf > d2_cf) | (value_cf < -d2_cf))] = 0
                projector = (1 - (np.sqrt(value * value + value_normal * value_normal + value_cf * value_cf)) / np.sqrt(
                    d2 * d2 + d2_normal * d2_normal + d2_cf * d2_cf))

                sum_vor = np.sum(im_copy * projector)

                adjust_value[
                    (value < d2) & (value >= -d2) & (value_normal < d2_normal) & (value_normal >= -d2_normal) & (
                            value_cf < d2_cf) & (value_cf >= -d2_cf)] += projector[(value < d2) & (value >= -d2) & (
                        value_normal < d2_normal) & (value_normal >= -d2_normal) & (value_cf < d2_cf) & (
                                                                                           value_cf >= -d2_cf)] / sum_vor

                # im_final = im_final + (im_final * im_t / np.sum(norm_im * im_t)) * np.sum(
                #     norm_im / np.sum(norm_im * im_t * im_final))  # np.sum(im_t*im_final)
                # print(np.sum(im_final*im_t/np.sum(norm_im*im_t)))
            im_final[normalization_matrix != 0] = im_final[normalization_matrix != 0] * \
                                                  adjust_value[normalization_matrix != 0] / (
                                                      normalization_matrix[normalization_matrix != 0])
            self._save_image_by_it(im_final, it, 0)

        self.im_final = im_final

    def _save_image_by_it(self, im, it, sb):
        directory = os.path.dirname(os.path.abspath(__file__))
        file_name = self.directory + "/EasyPETScan_it{}_sb{}".format(it, sb)
        print(file_name)
        volume = im.astype(np.float32)
        length = 1
        for i in volume.shape:
            length *= i
        # length = volume.shape[0] * volume.shape[2] * volume.shape[1]
        if len(volume.shape) > 1:
            data = np.reshape(volume, [1, length], order='F')
        else:
            data = volume
        shapeIm = volume.shape

        output_file = open(file_name, 'wb')
        arr = array('f', data[0])

        arr.tofile(output_file)
        output_file.close()
        # file_name = directory+"/Acquisitions/SENS_0_5"
        # file_name = directory+"/Acquisitions/sens_0_25"
