import numpy as np
import matplotlib.pyplot as plt
# from src.Designer.geometricdesigner import GeometryDesigner


class SetParametricCoordinates:
    def __init__(self, listMode=None, geometry_file=None, simulation_files=False, point_location="crystal_center",
                 crystal_height=2, crystal_width=2, crystal_depth=20, shuffle=False, FoV=45, distance_between_motors=30,
                 distance_crystals=60, recon2D=False, number_of_neighbours="Auto", generated_files=False,
                 transform_into_positive=True, normalization=False, centers_for_doi=False):

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
        self.shuffle = shuffle
        self.FoV = FoV
        self.simulation_files = simulation_files
        self.centers_for_doi = centers_for_doi

        print('Número de eventos: {}'.format(len(listMode)))
        geometry_file = geometry_file.astype(np.float32)
        geometry_file_a_side = geometry_file[:int(len(geometry_file) / 2), :]
        geometry_file_b_side = geometry_file[int(len(geometry_file) / 2):, :]
        nrCrystals_per_side = int(len(geometry_file) / 2)

        if simulation_files is True:
            if normalization:
                crystal_distance_to_center_fov_sideA = [geometry_file_a_side[(listMode[:, 2] - 1).astype(np.int32), i] for i in
                                                        range(3)]  # em mm
                crystal_distance_to_center_fov_sideB = [
                    geometry_file_b_side[(listMode[:, 3] - 1).astype(np.int32), i] for i in
                    range(3)]  # em mm

                top = -np.deg2rad(listMode[:, 1], dtype=np.float32)  # sim0ulation

                bot = np.deg2rad(listMode[:, 0], dtype=np.float32)


            else:
                crystal_distance_to_center_fov_sideA = [geometry_file_a_side[(listMode[:, 2] - 1).astype(np.int32), i] for i in
                                                        range(3)]  # em mm
                crystal_distance_to_center_fov_sideB = [
                    geometry_file_b_side[(listMode[:, 3] - 1).astype(np.int32), i] for i in
                    range(3)]  # em mm

                top = -np.deg2rad(listMode[:, 5], dtype=np.float32)  # sim0ulation

                bot = np.deg2rad(listMode[:, 4], dtype=np.float32)

        else:
            if normalization:
                crystal_distance_to_center_fov_sideA = [geometry_file[(listMode[:, 2] - 1).astype(np.int32), i] for i in
                                                        range(3)]  # em mm
                crystal_distance_to_center_fov_sideB = [
                    geometry_file[(listMode[:, 3] - 1 + nrCrystals_per_side).astype(np.int32), i] for i in
                    range(3)]  # em mm

                top = -np.deg2rad(listMode[:, 1], dtype=np.float32)  # sim0ulation

                bot = np.deg2rad(listMode[:, 0]+90, dtype=np.float32)
            else:
                crystal_distance_to_center_fov_sideA = [geometry_file[(listMode[:, 2] - 1).astype(np.int32), i] for i in
                                                        range(3)]  # em mm
                crystal_distance_to_center_fov_sideB = [
                    geometry_file[(listMode[:, 3] - 1 + nrCrystals_per_side).astype(np.int32), i] for i in range(3)]  # em mm

                top = np.deg2rad(listMode[:, 5], dtype=np.float32)  # real
                # top = np.deg2rad(listMode[:, 5],dtype=np.float32) # simulation

                bot = np.deg2rad(listMode[:, 4] + 90, dtype=np.float32)

        if point_location == "crystal_face":
            r_a = np.float32(distance_between_motors)
            # Initial Points - Crystal on the side of top motor positions

            rot_angle_init_points = np.pi / 2 - np.copy(top)

            rotation_matrix = np.array([[np.cos(bot, dtype=np.float32), -np.sin(bot, dtype=np.float32)],
                                        [np.sin(bot, dtype=np.float32), np.cos(bot, dtype=np.float32)]
                                        ], dtype=np.float32)

            initial_point = np.array([crystal_distance_to_center_fov_sideA[1] * np.cos(rot_angle_init_points),
                                      crystal_distance_to_center_fov_sideA[1] * np.sin(rot_angle_init_points)],
                                     dtype=np.float32)

            rotpoint = np.array([rotation_matrix[0, 0] * initial_point[0] + rotation_matrix[0, 1] * initial_point[1],
                                 rotation_matrix[1, 0] * initial_point[0] + rotation_matrix[1, 1] * initial_point[1]])

            RP = np.array([r_a * np.cos(bot), r_a * np.sin(bot), np.zeros(bot.shape[0])])
            self.midpoint = np.copy(RP)
            RP[0] = RP[0] + rotpoint[0]
            RP[1] = RP[1] + rotpoint[1]
            self.xi = RP[0]
            self.yi = RP[1]
            self.zi = crystal_distance_to_center_fov_sideA[2]
            self.midpoint[2] = self.zi
            top = np.pi - np.copy(top)
            self.farest_vertex = np.copy(RP)

            self.closest_vertex = np.copy(RP)

            # -------------END POINTS--------------
            # End Points - Crystal on the other side of top motor positions
            zav = np.float32(np.arctan(crystal_distance_to_center_fov_sideB[1] / distance_crystals))
            # zav = np.float64(np.arctan(crystal_distance_to_center_fov_sideB[1] / (r_a+r_b)))
            vtr = np.float32((distance_crystals ** 2 + crystal_distance_to_center_fov_sideB[1] ** 2) ** 0.5)
            # zav = np.tan()

            A = np.array([[np.cos(bot), -np.sin(bot), distance_between_motors * np.cos(bot)],
                          [np.sin(bot), np.cos(bot), distance_between_motors * np.sin(bot)],
                          [0, 0, 1]])

            B = np.array([np.cos(top + zav) * vtr,
                          np.sin(top + zav) * vtr,
                          1])

            dotA_B = np.array([A[0, 0] * B[0] + A[0, 1] * B[1] + A[0, 2] * B[2],
                               A[1, 0] * B[0] + A[1, 1] * B[1] + A[1, 2] * B[2],
                               A[2, 0] * B[0] + A[2, 1] * B[1] + A[2, 2] * B[2]], dtype=np.float32)

            self.xf = dotA_B[0]
            self.yf = dotA_B[1]
            self.zf = crystal_distance_to_center_fov_sideB[2]

        elif point_location == "crystal_center":
            self.beginPoints(top, bot, distance_between_motors, crystal_depth, crystal_width, crystal_height,
                    crystal_distance_to_center_fov_sideA, crystal_distance_to_center_fov_sideB)

            self.endPoints(top, bot, crystal_distance_to_center_fov_sideB, crystal_depth, crystal_width,
                           distance_crystals, distance_between_motors)

        if transform_into_positive:
            self._transform_into_positive_values(FoV, crystal_width)
        if not number_of_neighbours == "Auto":
            self.filter_neighbours(crystal_width, number_of_neighbours)

        if recon2D:
            self.filter_neighbours(crystal_width, neighbours=0)

    def beginPoints(self, top, bot, distance_between_motors, crystal_depth, crystal_width, crystal_height,
                    crystal_distance_to_center_fov_sideA, crystal_distance_to_center_fov_sideB):
        r_a = np.float32( distance_between_motors)

        half_crystal_depth = crystal_depth / 2
        half_crystal_width = crystal_width / 2

        ang_to_crystal_center = np.arctan(crystal_distance_to_center_fov_sideA[1] / half_crystal_depth,
                                          dtype=np.float32)

        point_rotation_to_center_crystal = np.sqrt(
            crystal_distance_to_center_fov_sideA[1] ** 2 + half_crystal_depth ** 2, dtype=np.float32)

        initial_point = np.array([point_rotation_to_center_crystal * np.cos(top + ang_to_crystal_center),
                                  point_rotation_to_center_crystal * np.sin(top + ang_to_crystal_center)],
                                 dtype=np.float32)

        distance_to_crystal_mid_point = np.sqrt(
            np.abs((crystal_distance_to_center_fov_sideA[1]) - half_crystal_width) ** 2
            + half_crystal_depth ** 2)
        ang_to_crystal_mid_point = np.arctan((crystal_distance_to_center_fov_sideA[
                                                  1] - half_crystal_width * np.sign(
            crystal_distance_to_center_fov_sideA[1])) / half_crystal_depth,
                                             dtype=np.float32)

        initial_mid_point = np.array(
            [distance_to_crystal_mid_point * np.cos(top + ang_to_crystal_mid_point),
             distance_to_crystal_mid_point * np.sin(top + ang_to_crystal_mid_point)],
            dtype=np.float32)

        ang_to_crystal_farest_vertex = np.arctan(
            (crystal_distance_to_center_fov_sideA[1]-np.sign(crystal_distance_to_center_fov_sideA[1])*crystal_width/2) / crystal_depth,
            dtype=np.float32)

        distance_to_crystal_farest_vertex = np.sqrt(
            crystal_distance_to_center_fov_sideA[1] ** 2 + crystal_depth ** 2)

        initial_farest_vertex = np.array(
            [distance_to_crystal_farest_vertex * np.cos(top + ang_to_crystal_farest_vertex),
             distance_to_crystal_farest_vertex * np.sin(top + ang_to_crystal_farest_vertex)],
            dtype=np.float32)

        central_crystal_point = self._rotate_point(bot, initial_point)
        mid_point = self._rotate_point(bot, initial_mid_point)
        farest_vertex = self._rotate_point(bot, initial_farest_vertex)

        RP = np.array([r_a * np.cos(bot), r_a * np.sin(bot), np.zeros(bot.shape[0])],
                      dtype=np.float32)  # rotation point
        self.origin_system_wz = RP
        self.xi = RP[0] + central_crystal_point[0]
        self.yi = RP[1] + central_crystal_point[1]
        self.zi = crystal_distance_to_center_fov_sideA[2]

        self.crystal_centerA = np.copy(RP)
        self.crystal_centerA[0] += central_crystal_point[0]
        self.crystal_centerA[1] += central_crystal_point[1]
        self.crystal_centerA[2] += crystal_distance_to_center_fov_sideA[2]

        self.midpoint = np.copy(RP)
        self.midpoint[0] += mid_point[0]
        self.midpoint[1] += mid_point[1]
        self.midpoint[2] = crystal_distance_to_center_fov_sideA[2]

        self.farest_vertex = np.copy(RP)
        self.farest_vertex[0] += farest_vertex[0]
        self.farest_vertex[1] += farest_vertex[1]
        orientation = np.sign(crystal_distance_to_center_fov_sideA[2] - crystal_distance_to_center_fov_sideB[2])
        orientation[orientation == 0] = 1
        self.farest_vertex[2] = (crystal_distance_to_center_fov_sideA[2] -orientation*crystal_height/2)

        # necessary for crystall planes
        if self.centers_for_doi:
            ang_to_crystal_front_face = np.float32(np.pi / 2) + np.copy(top)
            center_left_face_initial = np.array([(crystal_depth / 2) * np.cos(top),
                                                 # only valid for easypet wit 2 cristals
                                                 (crystal_depth / 2) * np.sin(top)],
                                                dtype=np.float32)

            center_frontal_face_initial_point = np.array(
                [crystal_distance_to_center_fov_sideA[1] * np.cos(ang_to_crystal_front_face),
                 crystal_distance_to_center_fov_sideA[1] * np.sin(ang_to_crystal_front_face)],
                dtype=np.float32)

            center_frontal_face_sideA = self._rotate_point(bot, center_frontal_face_initial_point)
            center_left_face_sideA = self._rotate_point(bot, center_left_face_initial)
            self.center_frontal_face_sideA = np.copy(RP)
            self.center_frontal_face_sideA[0] += center_frontal_face_sideA[0]
            self.center_frontal_face_sideA[1] += center_frontal_face_sideA[1]
            self.center_frontal_face_sideA[2] += crystal_distance_to_center_fov_sideA[2]

            self.center_left_face_sideA = np.copy(RP)
            self.center_left_face_sideA[0] += center_left_face_sideA[0]
            self.center_left_face_sideA[1] += center_left_face_sideA[1]
            self.center_left_face_sideA[2] += crystal_distance_to_center_fov_sideA[2]

            self.center_bottom_face_sideA = np.copy(self.midpoint)
            self.center_bottom_face_sideA[2] -= crystal_height / 2

    def endPoints(self, top, bot, crystal_distance_to_center_fov_sideB, crystal_depth, crystal_width,
                  distance_crystals, distance_between_motors):
        # -------------END POINTS--------------
        top = np.pi + top
        half_crystal_depth = crystal_depth / 2
        half_crystal_width = crystal_width / 2
        # End Points - Crystal on the other side of top motor positions
        zav = np.float32(
            np.arctan(crystal_distance_to_center_fov_sideB[1] / (distance_crystals + half_crystal_depth)))
        vtr = np.float32(
            ((distance_crystals + half_crystal_depth) ** 2 + crystal_distance_to_center_fov_sideB[1] ** 2) ** 0.5)

        A = np.array([[np.cos(bot), -np.sin(bot), distance_between_motors * np.cos(bot)],
                      [np.sin(bot), np.cos(bot), distance_between_motors * np.sin(bot)],
                      [np.zeros(len(bot)), np.zeros(len(bot)), np.ones(len(bot))]], dtype=np.float32)

        B = np.array([np.cos(top - zav) * vtr,
                      np.sin(top - zav) * vtr,
                      np.ones(len(bot))], dtype=np.float32)

        dotA_B = np.array([A[0, 0] * B[0] + A[0, 1] * B[1] + A[0, 2] * B[2],
                           A[1, 0] * B[0] + A[1, 1] * B[1] + A[1, 2] * B[2],
                           A[2, 0] * B[0] + A[2, 1] * B[1] + A[2, 2] * B[2]], dtype=np.float32)
        # teste
        # B = np.array([np.cos(top - zav) * vtr,
        #               np.sin(top - zav) * vtr,
        #               np.ones(len(bot))], dtype=np.float32)
        # A00 = np.cos(bot).astype(np.float32)
        # A01 = np.sin(bot).astype(np.float32)
        #
        # self.xf = A00 * B[0] - A01 * B[1] + distance_between_motors * A00 * B[2]
        # self.yf = A01 * B[0] + A00 * B[1] + distance_between_motors * A01 * B[2]

        self.xf = dotA_B[0]
        self.yf = dotA_B[1]
        self.zf = crystal_distance_to_center_fov_sideB[2]

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

    def filter_neighbours(self, crystal_width, neighbours=4):
        index = np.where(np.abs(self.zi - self.zf) < crystal_width * (neighbours + 1))
        self.xi = self.xi[index]
        self.xf = self.xf[index]
        self.yi = self.yi[index]
        self.yf = self.yf[index]
        self.zi = self.zi[index]
        self.zf = self.zf[index]
        self.midpoint = self.midpoint[:, index[0]]

    def cutCurrentFrame(self, frame_start, frame_end):
        self.xi = self.xi[frame_start:frame_end]
        self.yi = self.yi[frame_start:frame_end]
        self.zi = self.zi[frame_start:frame_end]
        self.farest_vertex = self.farest_vertex[frame_start:frame_end]
        self.midpoint = self.midpoint[frame_start:frame_end]
        self.xf = self.xf[frame_start:frame_end]
        self.yf = self.yf[frame_start:frame_end]
        self.zf = self.zf[frame_start:frame_end]

    def _check_geometry(self):

        dt = len(self.xi)
        dt = -1
        x = np.zeros((len(self.xi) * 2))
        x[:len(self.xi)] = self.xi
        x[len(self.xi):] = self.xf

        y = np.zeros((len(self.yi) * 2))
        y[:len(self.yi)] = self.yi
        y[len(self.yi):] = self.yf

        z = np.zeros((len(self.zi) * 2))
        z[:len(self.zi)] = self.zi
        z[len(self.zi):] = self.zf

        # t = np.zeros((len(self.zi) * 2))
        # t[:len(self.zi)] = rot_angle_init_points
        # t[len(self.zi):] = rot_angle_init_points

        # GeometryDesigner(geometry_vector=[x[0:dt], y[0:dt], z[0:dt]])

    def _plots(self):
        print('plots')
        x = np.zeros((len(self.xi), 2))
        x[:, 0] = self.xi
        x[:, 1] = self.xf
        y = np.zeros((len(self.xi), 2))
        y[:, 0] = self.yi
        y[:, 1] = self.yf
        z = np.zeros((len(self.xi), 2))
        z[:, 0] = self.zi
        z[:, 1] = self.zf
        #
        #
        # # D= np.sqrt((self.xf-self.xi)**2+(self.yi-self.yf)**2)
        #
        t = 3
        p = 2
        array = 0

        # # # toc = time.time()
        # # # print('nR VOR: {}'.format(len(listMode)))
        # # # print('TIME: {}'.format(toc-tic))
        # # xa = x[(self.listMode[:, t])%2 == 0,array]
        # # ya = y[(self.listMode[:, t])%2 == 0, array]
        # # za = z[(self.listMode[:, t])%2 == 0,array]
        # # xb = x[(self.listMode[:, t])%2 != 0,array]
        # # yb = y[(self.listMode[:, t])%2 != 0,array]
        # # zb = z[(self.listMode[:, t])%2 != 0, array]
        # #
        # # xc = x[(self.listMode[:, p]) % 2 == 0, array+1]
        # # yc = y[(self.listMode[:, p]) % 2 == 0, array+1]
        # # zc = z[(self.listMode[:, p]) % 2 == 0, array+1]
        # # xd = x[(self.listMode[:, p]) % 2 != 0, array+1]
        # # yd = y[(self.listMode[:, p]) % 2 != 0, array+1]
        # # zd = z[(self.listMode[:, p]) % 2 != 0, array+1]
        # # fig2 = plt.figure()
        #
        # #ax2 = fig2.add_subplot(111)
        # # x_t = point_p3[0][0]
        # # y_t = point_p3[1][0]
        # xe = x[(self.listMode[:, t]) % 2 == 0,:]
        # ye = y[(self.listMode[:, t]) % 2 == 0,:]
        # ze = z[(self.listMode[:, t]) % 2 == 0, :]
        # xf = x[(self.listMode[:, t]) % 2 != 0, :]
        # yf = y[(self.listMode[:, t]) % 2 != 0, :]
        # zf = z[(self.listMode[:, t]) % 2 != 0, :]
        # #
        fig = plt.figure()
        # #a = self.listMode[2:4]
        # #a = self.listMode[self.listMode[:,2] % 2==0,2:4]
        # a = [self.listMode[np.where(self.listMode[:,2] % 2==0),3],self.listMode[self.listMode[:,2] % 2!=0,3],
        #      self.listMode[np.where(self.listMode[:,2] % 2==0),2],self.listMode[self.listMode[:,2] % 2!=0,2]]
        # ax2 =fig.add_subplot(111, projection='3d')
        ax2 = fig.add_subplot(111)
        # list_ax =[0]*32
        # for i in range(0,len(list_ax)):
        #
        #     list_ax[i] = fig.add_subplot(8,4,i+1)
        #     b = self.listMode[np.where(self.listMode[:,2]==(i+1)),0]
        #     c = self.listMode[np.where(self.listMode[:,3]==(i+1)),1]
        #     list_ax[i].hist(b[0], 100)
        #     list_ax[i].hist(c[0], 100)
        #
        # fenergy, ((axenergyA, axenergyB)) = plt.subplots(2, 1)
        #
        # axenergyA.hist(self.listMode[:, 0], 100)
        # axenergyB.hist(self.listMode[:, 1], 100)
        #
        #
        #
        # f5, (axsino) = plt.subplots(1, 1)
        # number_values_top = len(np.unique(self.listMode[:,5]))  # int(header[5] /(round(header[3], 3) / header[4]))
        # number_values_bot = len(np.unique(self.listMode[:,4]))
        # array_bot = np.round(np.linspace(np.min((self.listMode[:,4])), np.max((self.listMode[:,4])), number_values_bot), 4) - (
        #             360 / (number_values_bot)) / 2
        # array_top = np.round(np.linspace(np.min((self.listMode[:,5])), np.max((self.listMode[:,5])), number_values_top), 4)
        # sinogram = axsino.hist2d((self.listMode[:,4]), (self.listMode[:,5]), bins=[array_bot, array_top])
        #
        # fmotors, ((axmotortop, axmotorbot)) = plt.subplots(2, 1)
        #
        # axmotorbot.hist(self.listMode[:,4], number_values_bot)
        # axmotortop.hist(self.listMode[:,5], number_values_top)

        # plt.show()
        ax2.scatter(x[:, 0], y[:, 0], color='red')
        ax2.scatter(x[:, 1], y[:, 1], color='blue')
        ax2.scatter(self.midpoint[0], self.midpoint[1], color='green')
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.xi[0], self.yi[0], self.zi[0], label="init")
        ax.scatter3D(self.xf[0], self.yf[0], self.zf[0], label="end")
        ax.scatter3D(self.midpoint[0, 0], self.midpoint[1, 0], self.midpoint[2, 0], label="midpoint")
        ax.scatter3D(self.farest_vertex[0, 0], self.farest_vertex[1, 0], self.farest_vertex[2, 0], label="farest")
        # ax.scatter3D(self.closest_vertex[0,0], self.closest_vertex[1,0],self.closest_vertex[2,0],  label= "closest")
        ax.legend()

        # ax2.scatter(xb, yb, color='green')
        # #for i in range(0, len(x), 2):
        # # plt.plot(x, y, 'ro-')
        # # ax2.scatter(x_t[0], y_t[0], color ='red')
        # # r = distance_between_motors*2
        # # theta = np.arange(0,360,1)
        # # x = r*np.cos(theta)
        # # y = r*np.sin(theta)-distance_between_motors
        # #
        # ax2.set_ylim(-60, 0)
        # ax2.set_xlim(-60, 0)
        # # ax2.set_aspect('equal', 'box')
        plt.show()


# if __name__ == '__main__':
    # filename = "/Users/alexis/PycharmProjects/3DScanner/3DScan
