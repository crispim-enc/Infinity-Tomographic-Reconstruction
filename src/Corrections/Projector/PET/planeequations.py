import numpy as np
import matplotlib.pyplot as plt
from src.Corrections.DOI import CrystalPlanes


class ParallelepipedProjector:
    def __init__(self, parametric_coordinates, pixelSizeXY=0.5, pixelSizeXYZ=0.5, crystal_width=2, crystal_height=2,
                 reflector_xy=0.35, reflector_z=0.28, FoV=45, crystal_depth=30, crystals_geometry=[32, 2],
                 bool_consider_reflector_in_z_projection=False, bool_consider_reflector_in_xy_projection=False,
                 distance_crystals=60, max_center_position=None):

        FoV = FoV / pixelSizeXY
        if bool_consider_reflector_in_z_projection:
            crystal_height = crystal_height + reflector_z
        else:
            crystal_height = crystal_height

        if bool_consider_reflector_in_xy_projection:
            crystal_width = crystal_width + reflector_xy
        else:
            crystal_width = crystal_width
        print("ParallelepipedProjector: Init")
        self.x_min_f = None
        self.x_max_f = None
        self.y_min_f = None
        self.y_max_f = None
        self.z_min_f = None
        self.z_max_f = None
        self.pixelSizeXY = pixelSizeXY
        self.pixelSizeXYZ = pixelSizeXYZ
        self.half_crystal_pitch_xy = np.float32(0.5 * crystal_width / pixelSizeXY)
        self.half_crystal_pitch_z = np.float32(0.5 * crystal_height / pixelSizeXYZ)
        self.distance_between_array_pixel = np.float32((distance_crystals) / pixelSizeXY)
        # self.distance_between_array_pixel = np.float32(40 / pixelSizeXYZ)

        # self.crystal_planes = CrystalPlanes(parametric_coordinates_object=parametric_coordinates,
        #                                pixelSizeXY=pixelSizeXY, pixelSizeXYZ=pixelSizeXYZ)
        # self.crystal_planes.multiply_by_pixel_size()
        # self.crystal_planes.crystals_central_planes()
        # Convert coordinates to postive values inside te image
        v1, v1_normal, v2, v3, v4, p1_list, p4_list, p5_list, p9_list = self.calculateVectors(parametric_coordinates, pixelSizeXY, pixelSizeXYZ)
        self.calculateVolume(FoV, crystal_height, pixelSizeXYZ)
        self.planeParameters(v1, v1_normal, v2, v3, v4, p1_list, p4_list, p5_list, p9_list)

        self.number_of_events = len(self.a)
        print("Number of events: {}".format(self.number_of_events))

    def calculateVolume(self, FoV, crystal_height, pixelSizeXYZ):
        print("Calculate Volume")
        self.number_of_pixels_x = int(np.ceil(FoV) + 1)
        self.number_of_pixels_y = int(np.ceil(FoV) + 1)
        self.number_of_pixels_z = int(np.ceil(self.max_z + crystal_height / pixelSizeXYZ))
        print("Max_x fov: {}".format(self.max_x))
        print("Max_y fov: {}".format(self.max_y))
        print("Min_z fov: {}".format(self.min_z))
        print("Max_z fov: {}".format(self.max_z))
        # Create a int range array from 0 to number of pixels)
        self.x_range_lim = [np.floor((self.max_x - FoV) / 2), np.ceil((self.max_x - FoV) / 2 + np.ceil(FoV))]
        self.y_range_lim = [np.floor((self.max_y - FoV) / 2), np.ceil((self.max_y - FoV) / 2 + np.ceil(FoV))]
        self.z_range_lim = [np.floor(self.min_z - 0.5 * crystal_height/pixelSizeXYZ), np.ceil(self.max_z + 0.5*crystal_height/pixelSizeXYZ)+1]

        x_range = np.arange(self.x_range_lim[0], self.x_range_lim[1],
                            dtype=np.int32)
        y_range = np.arange(self.y_range_lim[0], self.y_range_lim[1],
                            dtype=np.int32)

        z_range = np.arange(self.z_range_lim[0], self.z_range_lim[1],
                            dtype=np.int32)

        self.number_of_pixels_x = int(self.x_range_lim[1] - self.x_range_lim[0])
        self.number_of_pixels_y = int(self.y_range_lim[1] - self.y_range_lim[0])
        self.number_of_pixels_z = int(self.z_range_lim[1] - self.z_range_lim[0])

        # Create 3 EMPTY arrays with images size.
        self.im_index_x = np.ascontiguousarray(
            np.empty((self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z), dtype=np.int32))
        self.im_index_y = np.ascontiguousarray(
            np.empty((self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z), dtype=np.int32))
        self.im_index_z = np.ascontiguousarray(
            np.empty((self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z), dtype=np.int32))
        # Repeat values in one direction. Like x_axis only grows in x_axis (0, 1, 2 ... number of pixels)
        # but repeat these values on y an z axis
        self.im_index_x[:] = x_range[..., None, None]
        self.im_index_y[:] = y_range[None, ..., None]
        self.im_index_z[:] = z_range[None, None, ...]
        print(self.im_index_x.shape)

    def calculateVectors(self, parametric_coordinates, pixelSizeXY, pixelSizeXYZ):
        print("Calculate Vectors")
        # Stack all coordinates in one array
        # 4 points are needed to make two orthogonal planes.
        # p1 and p2 are the crystals centers
        # p3 is a point that belongs to the crystal  with the same Z value as p1
        # p4 is a point that belongs to the crystal it belongs to the normal plane of plane formed by p1, p2 and p3
        xi = (parametric_coordinates.xi / pixelSizeXY).astype(np.float32)
        xf = (parametric_coordinates.xf / pixelSizeXY).astype(np.float32)
        yi = (parametric_coordinates.yi / pixelSizeXY).astype(np.float32)
        yf = (parametric_coordinates.yf / pixelSizeXY).astype(np.float32)
        zi = (parametric_coordinates.zi / pixelSizeXYZ).astype(np.float32)
        zf = (parametric_coordinates.zf / pixelSizeXYZ).astype(np.float32)

        self.max_x = np.array([xi.max(), xf.max()]).max()
        self.max_y = np.array([yi.max(), yf.max()]).max()
        self.max_z = np.array([zi.max(), zf.max()]).max()
        self.min_z = np.array([zi.min(), zf.min()]).min()
        p1_list = np.column_stack((xi, yi, zi))
        self.p1_list = p1_list
        p2_list = np.column_stack((xf, yf, zf))
        p3_list = np.column_stack((((parametric_coordinates.midpoint[0]) / pixelSizeXY).astype(np.float32),
                                   ((parametric_coordinates.midpoint[1]) / pixelSizeXY).astype(np.float32),
                                   ((parametric_coordinates.midpoint[2]) / pixelSizeXY).astype(np.float32)))

        p9_list = np.column_stack(((parametric_coordinates.farest_vertex[0] / pixelSizeXY).astype(np.float32),
                                   (parametric_coordinates.farest_vertex[1] / pixelSizeXY).astype(np.float32),
                                   (parametric_coordinates.farest_vertex[2] / pixelSizeXYZ).astype(np.float32)))

        p4_list = p1_list.copy()
        # p5_list = p1_list.copy()
        p4_list[:, 2] = p4_list[:, 2] + self.half_crystal_pitch_z
        p5_list = (p1_list + p2_list) / 2
        p7_list = p5_list.copy()
        p6_list = p5_list + p3_list - p1_list

        p7_x = p7_list[:, 2]
        p7_z = -p7_list[:, 0]
        p7_list[:, 0] = p7_x
        p7_list[:, 2] = p7_z

        # p7_list[:, 2] = p7_list[:, 2] + self.half_crystal_pitch_z
        self.x_min_f = np.ascontiguousarray(np.min(np.vstack([xi, xf]), axis=0) - 2, dtype=np.short)
        self.z_min_f = np.ascontiguousarray(np.min(np.vstack([zi, zf]), axis=0) - 2, dtype=np.short)
        self.z_max_f = np.ascontiguousarray(np.max(np.vstack([zi, zf]), axis=0) + 2, dtype=np.short)
        v1, v1_normal, v2, v3, v4 = self.calculateNormalizedVectors(p1_list, p2_list, p3_list, p4_list, p5_list,
                                                                    p6_list, p7_list)
        return v1, v1_normal, v2, v3, v4, p1_list, p4_list, p5_list, p9_list



    def calculateNormalizedVectors(self, p1_list, p2_list, p3_list, p4_list, p5_list, p6_list, p7_list):
        # Calculate the vectors
        print("Calculate Normalized Vectors")
        v1_normal = p4_list - p1_list
        v1 = p2_list - p1_list
        v2 = p3_list - p1_list

        v3 = p5_list - p6_list
        v4 = p7_list - p5_list
        nf_v1 = ParallelepipedProjector.normVector(v1)
        nf_v1_normal = ParallelepipedProjector.normVector(v1_normal)
        nf_v2 = ParallelepipedProjector.normVector(v2)
        nf_v3 = ParallelepipedProjector.normVector(v3)
        nf_v4 = ParallelepipedProjector.normVector(v4)

        for coor in range(3):
            v1[:, coor] = v1[:, coor] / nf_v1
            v1_normal[:, coor] = v1_normal[:, coor] / nf_v1_normal
            v2[:, coor] = v2[:, coor] / nf_v2
            v3[:, coor] = v3[:, coor] / nf_v3
            v4[:, coor] = v4[:, coor] / nf_v4

        return v1, v1_normal, v2, v3, v4

    def calcuteVectorsImagePlane(self):
        p1 = np.array([self.x_range_lim[0], self.y_range_lim[0], self.z_range_lim[0]])
        p2 = np.array([self.x_range_lim[1], self.y_range_lim[0], self.z_range_lim[0]])
        p3 = np.array([self.x_range_lim[0], self.y_range_lim[1], self.z_range_lim[0]])
        p4 = np.array([self.x_range_lim[1], self.y_range_lim[1], self.z_range_lim[0]])
        p5 = np.array([self.x_range_lim[0], self.y_range_lim[0], self.z_range_lim[1]])
        p6 = np.array([self.x_range_lim[1], self.y_range_lim[0], self.z_range_lim[1]])
        p7 = np.array([self.x_range_lim[0], self.y_range_lim[1], self.z_range_lim[1]])
        p8 = np.array([self.x_range_lim[1], self.y_range_lim[1], self.z_range_lim[1]])
        points_face_image = np.array([p1, p2, p3, p4, p5, p6, p7, p8])
        vector_image_xx = p2 - p1
        vector_image_yy = p3 - p1
        vector_image_zz = p5 - p1
        vector_image_xx_zmax = p6 - p8
        vector_image_yy_zmax = p7 - p8
        vector_image_zz_zmax = p4 - p8

        nf_vector_image_xx = ParallelepipedProjector.normVector(vector_image_xx)
        nf_vector_image_yy = ParallelepipedProjector.normVector(vector_image_yy)
        nf_vector_image_zz = ParallelepipedProjector.normVector(vector_image_zz)
        nf_vector_image_xx_zmax = ParallelepipedProjector.normVector(vector_image_xx_zmax)
        nf_vector_image_yy_zmax = ParallelepipedProjector.normVector(vector_image_yy_zmax)
        nf_vector_image_zz_zmax = ParallelepipedProjector.normVector(vector_image_zz_zmax)

        for coor in range(3):
            vector_image_xx[coor] /= nf_vector_image_xx
            vector_image_yy[coor] /= nf_vector_image_yy
            vector_image_zz[coor] /= nf_vector_image_zz
            vector_image_xx_zmax[coor] /= nf_vector_image_xx_zmax
            vector_image_yy_zmax[coor] /= nf_vector_image_yy_zmax
            vector_image_zz_zmax[coor] /= nf_vector_image_zz_zmax

        return vector_image_xx, vector_image_yy, vector_image_zz, vector_image_xx_zmax, vector_image_yy_zmax, vector_image_zz_zmax, points_face_image

    def planesImage(self):
        vector_image_xx, vector_image_yy, vector_image_zz, vector_image_xx_zmax, vector_image_yy_zmax, vector_image_zz_zmax, points_face_image = self.calcuteVectorsImagePlane()
        face_image_A_plane = ParallelepipedProjector.planeEquation(vector_image_xx, vector_image_yy, points_face_image[0])
        face_image_B_plane = ParallelepipedProjector.planeEquation(vector_image_xx, vector_image_zz, points_face_image[0])
        face_image_C_plane = ParallelepipedProjector.planeEquation(vector_image_yy, vector_image_zz, points_face_image[0])
        face_image_D_plane = ParallelepipedProjector.planeEquation(vector_image_xx_zmax, vector_image_yy_zmax, points_face_image[7])
        face_image_E_plane = ParallelepipedProjector.planeEquation(vector_image_xx_zmax, vector_image_zz_zmax, points_face_image[7])
        face_image_F_plane = ParallelepipedProjector.planeEquation(vector_image_yy_zmax, vector_image_zz_zmax, points_face_image[7])
        planes_face_image = np.array([face_image_A_plane, face_image_B_plane, face_image_C_plane, face_image_D_plane, face_image_E_plane, face_image_F_plane])
        return planes_face_image

    @staticmethod
    def normVector(v1):
        try:
            return (np.sqrt(v1[:, 0] ** 2 + v1[:, 1] ** 2 + v1[:, 2] ** 2)).astype(np.float32)
        except IndexError:
            return (np.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)).astype(np.float32)

    def planeParameters(self, v1, v1_normal, v2, v3, v4, p1_list, p4_list, p5_list, p9_list):
        print("Calculate Plane Parameters")
        self.v1 = v1
        self.min_max_position_of_the_event_in_image()
        # These operations gives the equations of planes
        self.a, self.b, self.c, self.d = ParallelepipedProjector.planeEquation(v1, v2, p1_list)
        # self.distance_to_central_plane = np.ascontiguousarray(np.abs(
        #     self.a * p9_list[:, 0] + self.b * p9_list[:, 1] + self.c * p9_list[:, 2] - self.d) / np.sqrt(
        #     self.a ** 2 + self.b ** 2 + self.c ** 2) + self.half_crystal_pitch_z, dtype=np.float32)

        self.distance_to_central_plane = np.ascontiguousarray(np.abs(
            self.a * p9_list[:, 0] + self.b * p9_list[:, 1] + self.c * p9_list[:, 2] - self.d) / np.sqrt(
            self.a ** 2 + self.b ** 2 + self.c ** 2), dtype=np.float32)

        self.a_normal, self.b_normal, self.c_normal, self.d_normal = ParallelepipedProjector.planeEquation(v1_normal,
                                                                                                           v1,
                                                                                                           p4_list)
        # self.distance_to_central_plane_normal = np.ascontiguousarray(np.abs(
        #     self.a_normal * p9_list[:, 0] + self.b_normal * p9_list[:, 1] + self.c_normal * p9_list[:,
        #                                                                                               2] - self.d_normal) / np.sqrt(
        #     self.a_normal ** 2 + self.b_normal ** 2 + self.c_normal ** 2) + self.half_crystal_pitch_xy,
        #                                                              dtype=np.float32)

        self.distance_to_central_plane_normal = np.ascontiguousarray(np.abs(
            self.a_normal * p9_list[:, 0] + self.b_normal * p9_list[:, 1] + self.c_normal * p9_list[:,
                                                                                            2] - self.d_normal) / np.sqrt(
            self.a_normal ** 2 + self.b_normal ** 2 + self.c_normal ** 2),
                                                                     dtype=np.float32)

        self.a_cf, self.b_cf, self.c_cf, self.d_cf = ParallelepipedProjector.planeEquation(v4, v3, p5_list)
        # self.distance_to_central_plane_cf = np.ascontiguousarray(np.abs(
        #     self.a_cf * p9_list[:, 0] + self.b_cf * p9_list[:, 1] + self.c_cf * p9_list[:,
        #                                                                                   2] - self.d_cf) / np.sqrt(
        #     self.a_cf ** 2 + self.b_cf ** 2 + self.c_cf ** 2), dtype=np.float32)

        self.distance_to_central_plane_cf = np.ascontiguousarray(np.abs(
            self.a_cf * p9_list[:, 0] + self.b_cf * p9_list[:, 1] + self.c_cf * p9_list[:,
                                                                                2] - self.d_cf) / np.sqrt(
            self.a_cf ** 2 + self.b_cf ** 2 + self.c_cf ** 2), dtype=np.float32)

        print("END- Equation Planes")

    @staticmethod
    def planeEquation(v1, v2, p1):
        try:
            cp = np.cross(v2, v1).astype(np.float32)
            a = np.ascontiguousarray(cp[:, 0], dtype=np.float32)
            b = np.ascontiguousarray(cp[:, 1], dtype=np.float32)
            c = np.ascontiguousarray(cp[:, 2], dtype=np.float32)
            d = np.ascontiguousarray(np.array(cp[:, 0] * p1[:, 0] + cp[:, 1] * p1[:, 1] + cp[:, 2] * p1[:, 2],
                                              dtype=np.float32))
        except IndexError:
            cp = np.cross(v2, v1).astype(np.float32)
            a = np.ascontiguousarray(cp[0], dtype=np.float32)
            b = np.ascontiguousarray(cp[1], dtype=np.float32)
            c = np.ascontiguousarray(cp[2], dtype=np.float32)
            d = np.ascontiguousarray(np.array(cp[0] * p1[0] + cp[1] * p1[1] + cp[2] * p1[2], dtype=np.float32))

        return a, b, c, d

    def cut_current_frame(self, frame_start, frame_end):
        self.distance_to_central_plane = self.distance_to_central_plane[frame_start:frame_end]
        self.distance_to_central_plane_normal = self.distance_to_central_plane_normal[frame_start:frame_end]
        self.distance_to_central_plane_cf = self.distance_to_central_plane_cf[frame_start:frame_end]
        # self.p1_list = self.p1_list[frame_start:frame_end]
        # self.p2_list = self.p2_list[frame_start:frame_end]
        # self.p3_list = self.p3_list[frame_start:frame_end]
        # self.p4_list = self.p4_list[frame_start:frame_end]

        self.a = self.a[frame_start:frame_end]
        self.b = self.b[frame_start:frame_end]
        self.c = self.c[frame_start:frame_end]
        self.d = self.d[frame_start:frame_end]

        self.a_normal = self.a_normal[frame_start:frame_end]
        self.b_normal = self.b_normal[frame_start:frame_end]
        self.c_normal = self.c_normal[frame_start:frame_end]
        self.d_normal = self.d_normal[frame_start:frame_end]

        self.a_cf = self.a_cf[frame_start:frame_end]
        self.b_cf = self.b_cf[frame_start:frame_end]
        self.c_cf = self.c_cf[frame_start:frame_end]
        self.d_cf = self.d_cf[frame_start:frame_end]
        # self.p1_list = self.p1_list[frame_start:frame_end]
        # self.p2_list = self.p2_list[frame_start:frame_end]
        # sle.p3_list = self.p3_list[frame_start:frame_end]
        # self.p4_list = self.p4_list[frame_start:frame_end]
        # self.p5_list = self.p5_list[frame_start:frame_end]
        # self.p6_list = self.p6_list[frame_start:frame_end]
        # self.p7_list = self.p7_list[frame_start:frame_end]
        # self.p9_list = self.p9_list[frame_start:frame_end]

    def min_max_position_of_the_event_in_image(self):
        # calculate the intersection of the coincidence vector v1 with image planes
        planes_face_image = self.planesImage()
        # calculate the determinant
        i = 0
        intersections = np.zeros((planes_face_image.shape[0],self.p1_list.shape[0], 3), dtype=np.float32)
        for plane in planes_face_image:
            # t = -(plane[3]+plane[0]*self.p1_list[:,0]+plane[1]*self.p1_list[:,1]+plane[2]*self.p1_list[:,2])/(plane[0]*plane[0]+plane[1]*plane[1]+plane[2]*plane[2])
            t = -(-plane[3] + self.p1_list[:, 0] * plane[0] + self.p1_list[:, 1] * plane[1] + self.p1_list[:, 2] * plane[2])/(plane[0]*self.v1[:,0]+plane[1]*self.v1[:,1]+plane[2]*self.v1[:,2])
            intersections[i] = self.p1_list + self.v1 * t[:, np.newaxis]
            # for each plane set intersection the value of np.nan if the intersection is out of the image in the 3 directions, x_rangelim, y_rangelim, z_rangelim
            mask_x = (intersections[i, :, 0] < self.x_range_lim[0]) | (intersections[i, :, 0] > self.x_range_lim[1])
            mask_y = (intersections[i, :, 1] < self.y_range_lim[0]) | (intersections[i, :, 1] > self.y_range_lim[1])
            mask_z = (intersections[i, :, 2] < self.z_range_lim[0]) | (intersections[i, :, 2] > self.z_range_lim[1])
            mask = mask_x | mask_y | mask_z

            intersections[i, mask] = np.nan




            # t = -(plane[3] + np.dot(plane[:3], self.v1)) / np.dot(plane[:3], plane[:3])
            # intersections[i] = self.v1.T + plane[:3] * t
            i += 1
        # if the value is bigger than range in each direction of the image, the intersection is set no np.nan
        # intersections[np.where((intersections[:, :, 0] <= self.x_range_lim[0]) | (intersections[:, :, 0] >= self.x_range_lim[1]))] = np.nan
        # intersections[np.where((intersections[:, :, 1] <= self.y_range_lim[0]) | (intersections[:, :, 1] >= self.y_range_lim[1]))] = np.nan
        # intersections[np.where((intersections[:, :, 2] <= self.z_range_lim[0]) | (intersections[:, :, 2] >= self.z_range_lim[1]))] = np.nan
        # calculate max and min of the intersection not nan
        x_min = np.nanmin(intersections[:, :, 0], axis=0)
        x_max = np.nanmax(intersections[:, :, 0], axis=0)
        y_min = np.nanmin(intersections[:, :, 1], axis=0)
        y_max = np.nanmax(intersections[:, :, 1], axis=0)
        z_min = np.nanmin(intersections[:, :, 2], axis=0)
        z_max = np.nanmax(intersections[:, :, 2], axis=0)

        x_min[np.argwhere(np.isnan(x_min))] = self.x_range_lim[0]
        x_max[np.argwhere(np.isnan(x_max))] = self.x_range_lim[1]
        y_min[np.argwhere(np.isnan(y_min))] = self.y_range_lim[0]
        y_max[np.argwhere(np.isnan(y_max))] = self.y_range_lim[1]
        z_min[np.argwhere(np.isnan(z_min))] = self.z_range_lim[0]
        z_max[np.argwhere(np.isnan(z_max))] = self.z_range_lim[1]
        self.x_min_f = np.ascontiguousarray(x_min, dtype=np.short)
        self.x_max_f = np.ascontiguousarray(x_max, dtype=np.short)
        self.y_min_f = np.ascontiguousarray(y_min, dtype=np.short)
        self.y_max_f = np.ascontiguousarray(y_max, dtype=np.short)
        self.z_min_f = np.ascontiguousarray(z_min, dtype=np.short)
        self.z_max_f = np.ascontiguousarray(z_max, dtype=np.short)

    def plots(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = ['red', 'red', 'green', 'blue', 'yellow', 'black', 'brown']
        points = [self.p1_list, self.p2_list, self.p3_list, self.p4_list, self.p5_list, self.p6_list, self.p7_list]
        for point, color in zip(points, colors):
            ax.scatter(point[:, 0], point[:, 1], point[:, 2], color=color)

        plt.show()
