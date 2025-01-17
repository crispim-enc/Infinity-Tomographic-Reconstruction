import os
import numpy as np
from scipy.interpolate import interp1d


class AdaptiveNormalizationMatrix:
    def __init__(self, data_in=None, number_of_crystals=None, data_in_s=None, number_of_reps=10,
                 rangeTopMotor=99, begin_range_botMotor=0, end_rangeBotMotor=360,
                 stepTopmotor=0.225, stepBotMotor=3.6, recon_2D=False):
        # number_of_reps = 20
        self._probability_uniform_phantom = None
        self.data_in = data_in
        self.number_of_crystals = number_of_crystals
        self.data_in_s = data_in
        self.number_of_reps = number_of_reps
        self.rangeTopMotor = rangeTopMotor
        self.begin_range_botMotor = begin_range_botMotor
        self.end_rangeBotMotor = end_rangeBotMotor
        self.stepTopmotor = stepTopmotor
        self.stepBotMotor = stepBotMotor
        self.recon_2D = recon_2D
        self.total_counts = None
        self.reading_data = None
        self.main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.probability_file_path = os.path.join(self.main_dir,
            'system_configurations', 'x_{}__y_{}'.format(self.number_of_crystals[0], self.number_of_crystals[1]),
            'crystals_detection_probability.npy')
        self.probability_top_file_path = os.path.join(self.main_dir,
            'system_configurations', 'x_{}__y_{}'.format(self.number_of_crystals[0], self.number_of_crystals[1]),
            'top_motor_probability.npy')
        self.top_positions_file_path = os.path.join(self.main_dir,
            'system_configurations', 'x_{}__y_{}'.format(self.number_of_crystals[0], self.number_of_crystals[1]),
            'top_motor_positions.npy')

    def matrix_without_reduction(self):
        self.reading_data = np.ones(
            (len(self.angles) * len(self.sideA_id) * len(self.sideB_id), 7), dtype=np.float32)
        self.reading_data[:, 4] = np.repeat(self.angles[:, 0], len(self.sideA_id) * len(self.sideB_id))
        self.reading_data[:, 5] = np.repeat(self.angles[:, 1], len(self.sideA_id) * len(self.sideB_id))
        self.reading_data[:, 2] = np.tile(np.repeat(self.sideA_id, len(self.sideB_id)), len(self.angles[:, 0]))
        self.reading_data[:, 3] = np.tile(self.sideB_id, len(self.sideB_id) * len(self.angles[:, 0]))
        self.reading_data[:, 0] = 511
        self.reading_data[:, 1] = 511
        # self.every_possible_position_array[:, 2] = np.tile(
        #     np.repeat(self.sideA_id, len(self.sideB_id)), len(self.angles))
        # self.every_possible_position_array[:, 0] = np.tile(np.repeat(self.sideA_id, len(self.sideB_id)),
        #                                                    len(self.topmotors_position) * len(self.botmotors_position))
        # self.every_possible_position_array[:, 1] = np.tile(self.sideB_id, len(self.topmotors_position) * len(
        #     self.botmotors_position) * len(self.sideB_id))

    def normalization_LM(self):
        if self.data_in is None:
            # used for pre-calculation_normalization
            top = np.arange(-self.rangeTopMotor / 2, self.rangeTopMotor / 2 + self.stepTopmotor, self.stepTopmotor)
            bot = np.arange(self.begin_range_botMotor, self.end_rangeBotMotor, self.stepBotMotor)
        else:
            top = np.unique(self.data_in[:, 5])
            bot = np.unique(self.data_in[:, 4])
            # self.stepTopmotor = np.diff(top).min()
        # self.angles = np.vstack({tuple(e) for e in data_in[:,4:6]})
        # self.topmotors_position =self.angles[:,1]
        # self.botmotors_position =angles[:,0]
        # self.reading_data = np.zeros ((len(angles)*32))

        print("Normalization TOP :{}".format(len(top)))
        print("Normalization BOT :{}".format(len(bot)))
        self.reading_data = np.ones(
            (len(top) * len(bot) * self.number_of_reps, 4), dtype=np.float32)
        # self.sideA_id = np.arange(1, number_of_crystals[0] * number_of_crystals[1] + 1, dtype=np.int8)
        # self.sideB_id = np.arange(1, number_of_crystals[0] * number_of_crystals[1] + 1, dtype=np.int8)
        # self.sideA_id = np.arange(1, np.max(self.data_in[:, 2]) - np.min(self.data_in[:, 2]) + 1, dtype=np.int8)
        # self.sideB_id = np.arange(1, np.max(self.data_in[:, 2]) - np.min(self.data_in[:, 2]) + 1, dtype=np.int8)
        # self.matrix_without_reduction()
        # number_of_reps= 400

        self.reading_data[:,1] = np.repeat(top, len(bot) * self.number_of_reps)
        probability_top = np.load(self.probability_top_file_path)
        top_norm = np.load(self.top_positions_file_path)

        inter_top_positions = interp1d(top_norm, probability_top, fill_value="extrapolate")
        top_new_conditions = np.unique(top)
        probability_top = inter_top_positions(top_new_conditions)
        probability_top += np.abs(probability_top.min())
        probability_top /= np.sum(probability_top)
        # self.reading_data[:, 1] = np.random.choice(len(probability_top), len(self.reading_data),
        #                                            p=probability_top) * self.stepTopmotor - top.max()
        # #uniform random

        # self.reading_data[:, 1] = np.random.uniform(top.min(), top.max(), len(self.reading_data))
        self.reading_data[:, 1] = np.repeat(top, len(bot) * self.number_of_reps)
        self.reading_data[:, 0] = np.tile(
            np.repeat(bot, self.number_of_reps), len(top))
        x = np.arange(-self.number_of_crystals[0], self.number_of_crystals[0])
        # xU, xL = x + 0.5, x - 0.5
        # prob = ss.norm.cdf(xU, scale=3) - ss.norm.cdf(xL, scale=3)
        # prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
        # nums = np.random.choice(x, size=len(self.reading_data), p=prob)
        #
        # self.reading_data[:,2] = np.random.randint(1,number_of_crystals[0]*number_of_crystals[1],len(self.reading_data))
        # self.reading_data[:,3] = np.random.randint(1,number_of_crystals[0]*number_of_crystals[1],len(self.reading_data))
        # if self.recon_2D:
        #     self.reading_data[:, 2] = np.random.randint(np.min(self.data_in[:, 2]), np.max(data_in[:, 2]) + 1,
        #                                                 len(self.reading_data))
        #     self.reading_data[:, 3] = self.reading_data[:, 2]
        # else:
        #     self.reading_data[:, 2] = np.random.randint(np.min(data_in[:, 2]), np.max(data_in[:, 2]) + 1,
        #                                                 len(self.reading_data))
        #     self.reading_data[:, 3] = np.random.randint(np.min(data_in[:, 3]), np.max(data_in[:, 3]) + 1,
        #                                                 len(self.reading_data))
        # self.reading_data[:,2] = np.random.poisson(np.max(data_in[:,2])/2, len(self.reading_data))
        # self.reading_data[:,3] = np.random.poisson(np.max(data_in[:,3])/2, len(self.reading_data))

        # self.reading_data = self.reading_data[self.reading_data[:,2]<np.max(data_in[:,2])]
        # self.reading_data = self.reading_data[self.reading_data[:,3]<np.max(data_in[:,3])]
        # b= np.load("C:\\Users\\pedro.encarnacao\\Desktop\\b.npy")
        # t = np.load("C:\\Users\\pedro.encarnacao\\Desktop\\t.npy")

        # differ_abs = (data_in[:, 3]-data_in[:, 2])
        # differ_abs = differ_abs[differ_abs%2 == 0]
        # unique_values = np.unique(differ_abs)
        # unique_values = np.insert(unique_values,0,unique_values[0]-1) # add boundaries
        # unique_values = np.append(unique_values,unique_values[-1]+1) # add boundaries
        # probability = np.histogram(differ_abs, unique_values)[0] / len(differ_abs)
        total_crystals = self.number_of_crystals[0] * self.number_of_crystals[1]

        # # probability = np.histogram((data_in[:, 3]-data_in[:,2])/(data_in[:,3]+data_in[:,2]),64)[0]/len(data_in)
        # # probability = np.histogram((data_in[:, 3]+data_in[:,2])/2,64)[0]/len(data_in)
        prob = self.probability_uniform_phantom()
        value = np.random.choice((self.number_of_crystals[0] * self.number_of_crystals[1]) ** 2, len(self.reading_data),
                                 p=prob)
        self.reading_data[:, 3] = np.array(value % total_crystals + 1, dtype=np.int32)
        # self.reading_data[:,3] = np.random.choice(len(unique_values)-1, len(self.reading_data), p=probability)
        self.reading_data[:, 2] = np.array((value // total_crystals) + 1,
                                           dtype=np.int32)

        # remove id diff larger than 30
        # diff_vector = np.abs(self.reading_data[:, 3] - self.reading_data[:, 2])
        # self.reading_data = self.reading_data[diff_vector <= 32]
        # # # d = np.random.choice(len(t), len(self.reading_data), p=b / np.sum(b))
        # # id =
        # self.reading_data[:,2:4]=t[d]
        # self.reading_data[:, 0] = 511
        # self.reading_data[:, 1] = 511

        self.total_counts = len(self.reading_data)

        # ind = np.lexsort((self.reading_data[:, 3], self.reading_data[:, 2]))
        # self.reading_data = self.reading_data[ind]
        # diff_vector = np.abs(self.reading_data[:, 3] - self.reading_data[:, 2])
        # sum_vector = self.reading_data[:, 3] + self.reading_data[:, 2]+diff_vector
        # # sum_vector = np.array(sum_vector, dtype=np.int32)
        # # sum_vector=sum_vector[np.lexsort((sum_vector,diff_vector))]
        # ind = np.lexsort((sum_vector, self.reading_data[:, 3], self.reading_data[:, 2],self.reading_data[:, 5],self.reading_data[:, 4]))
        # self.reading_data = self.reading_data[ind]
        # self.reading_data = np.unique(self.reading_data, axis=0)

        print("Normalization MAtrix number events :{}".format(self.total_counts))

    def write_probability_phantom(self):
        """

        """

        total_crystals = self.number_of_crystals[0] * self.number_of_crystals[1]
        # isolate 0º on top motor
        data_in = self.data_in[np.abs(np.round(self.data_in[:, 5],5)) <= 10]
        probability = \
            np.histogram((data_in[:, 2] - 1) * total_crystals + (data_in[:, 3] - 1),
                         total_crystals ** 2,
                         (0, total_crystals ** 2))[0] / len(data_in)
        # probability = \
        #     np.histogram((self.data_in[:, 2] - 1) * total_crystals + (self.data_in[:, 3] - 1),
        #                  total_crystals ** 2,
        #                  (0, total_crystals ** 2))[0] / len(self.data_in)
        np.save(self.probability_file_path, probability)

        top = np.unique(self.data_in[:, 5])
        probability_top = np.histogram(self.data_in[:, 5], len(top))[0] / len(self.data_in)

        np.save(self.probability_top_file_path, probability_top)
        np.save(self.top_positions_file_path, top)
        return probability

    def probability_uniform_phantom(self):
        # try:
        #     self._probability_uniform_phantom = np.load(self.probability_file_path)
        # except FileNotFoundError:
        # self._probability_uniform_phantom = self.write_probability_phantom()

        try:
            self._probability_uniform_phantom = np.load(self.probability_file_path)

            print("probabi")
        except FileNotFoundError:
            comb = int((self.number_of_crystals[0] * self.number_of_crystals[1])**2)
            self._probability_uniform_phantom = np.ones(comb)/comb
            # np.save(self.probability_file_path, self._probability_uniform_phantom)
            # print("write")
        return self._probability_uniform_phantom

    def _load_standard_normalization_matrix(self, normalized=True, simulation=False):
        sens_name = 'SENS_XY_{}um_XYZ_{}um'.format(int(self.pixelSizeXY * 1000), int(self.pixelSizeXYZ * 1000))
        file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'system_configurations',
                                 'x_{}__y_{}'.format(
                                     self.crystals_geometry[0], self.crystals_geometry[1]), 'normalization_matrix',
                                 'Sensitivity', sens_name)

        #### TEM QUE SE GRAVAR NUM HEADER AS DIMENSOES DA IMAGEM DE SENSIBILIDADE

        # size_shape = [48, 48, 31]
        if self.crystals_geometry[0] == 16:
            if self.pixelSizeXY == 1:
                size_shape = [60, 60, 33]
            if self.pixelSizeXY == 0.5:
                size_shape = [120, 120, 65]

            if self.pixelSizeXY == 0.3:
                size_shape = [200, 200, 108]

            if self.pixelSizeXY == 0.25:
                size_shape = [240, 240, 145]

        elif self.crystals_geometry[0] == 32:
            if self.pixelSizeXY == 1:
                size_shape = [60, 60, 33]
            if self.pixelSizeXY == 0.5:
                # size_shape = [120, 120, 65]
                size_shape = [120, 120, 129]

            if self.pixelSizeXY == 0.3:
                size_shape = [200, 200, 215]
            if self.pixelSizeXY == 0.25:
                size_shape = [240, 240, 145]
        # size_shape = [int(60*self.pixelSizeXY),int(60*self.pixelSizeXY),int(37*self.pixelSizeXYZ)]
        sizefile = size_shape[0] * size_shape[1] * size_shape[2]
        output_file = open(file_name, 'rb')
        a = array('f')
        a.fromfile(output_file, sizefile)
        output_file.close()
        sensitivity_matrix = np.array(a)
        sensitivity_matrix = sensitivity_matrix.reshape((size_shape[0], size_shape[1], size_shape[2]), order='F')

        if normalized:
            sensitivity_matrix = sensitivity_matrix / np.max(
                sensitivity_matrix)  # Passar para dentro da Função de matrix de sensibilidade
        else:
            sensitivity_matrix = sensitivity_matrix / np.sum(
                sensitivity_matrix)  # Passar para dentro da Função de matrix de sensibilidade

        return sensitivity_matrix

    def _save_on_fly_normalization_matrix(self):
        normalization_header = {
            'type_of_projector': self.type_of_projector,
            'recon2D': reading_hardware_parameters.u_board_version,
            'number_of_neighbours': reading_hardware_parameters.module_control,
            'map_precedent': reading_hardware_parameters.array_crystal_x,
            'beta': reading_hardware_parameters.array_crystal_y,
            'local_median_v_number': reading_hardware_parameters.angle_bot_rotation,
            'angle_top_correction': reading_hardware_parameters.angle_top_correction,
            'multiplexed': reading_hardware_parameters.multiplexed,
            'reading_method': reading_hardware_parameters.reading_method,
            'number_adc_channel': reading_hardware_parameters.number_adc_channel,
            'bed_version': reading_hardware_parameters.bed_version,
            'bed_diameter': reading_hardware_parameters.bed_diameter,
            'pc_communication': reading_hardware_parameters.pc_communication,
            'baudrate': reading_hardware_parameters.baudrate,
            'motor_bot': reading_hardware_parameters.motor_bot,
            'motor_top': reading_hardware_parameters.motor_top,
            'bed_motor': reading_hardware_parameters.bed_motor,
            'fourth_motor': reading_hardware_parameters.fourth_motor,
            'capable4CT': reading_hardware_parameters.capable4CT,
            'crystal_pitch_x': reading_hardware_parameters.crystal_pitch_x,
            'crystal_pitch_y': reading_hardware_parameters.crystal_pitch_y,
            'crystal_length': reading_hardware_parameters.crystal_length,
            'reflector_exterior_thic': reading_hardware_parameters.reflector_exterior_thic,
            'reflector_interior_A_x': reading_hardware_parameters.reflector_interior_A_x,
            'reflector_interior_A_y': reading_hardware_parameters.reflector_interior_A_y,
            'reflector_interior_B_x': reading_hardware_parameters.reflector_interior_B_x,
            'reflector_interior_B_y': reading_hardware_parameters.reflector_interior_B_y,
            'distance_between_motors': reading_hardware_parameters.distance_between_motors,
            'distance_between_crystals': reading_hardware_parameters.distance_between_crystals,
            'centercrystals2topmotor_x_sideA': reading_hardware_parameters.centercrystals2topmotor_x_sideA,
            'centercrystals2topmotor_x_sideB': reading_hardware_parameters.centercrystals2topmotor_x_sideB,
            'centercrystals2topmotor_y': reading_hardware_parameters.centercrystals2topmotor_y
        }

        systemConfigurations = json.dumps(systemConfigurations)
        systemConfigurations_info = array('u', systemConfigurations)
        systemConfigurations_info_size = [len(systemConfigurations_info)]
        systemConfigurations_info_size = array('i', systemConfigurations_info_size)
        volume = self.im.astype(np.float32)
        length = volume.shape[0] * volume.shape[2] * volume.shape[1]
        data = np.reshape(volume, [1, length], order='F')

        shapeIm = volume.shape

        output_file = open(os.path.join(study_path, 'static_image',
                                        '{}_ IMAGE {}.T'.format(os.path.basename(study_path), volume.shape)), 'wb')
        arr = array('f', data[0])

        arr.tofile(output_file)
        output_file.close()
