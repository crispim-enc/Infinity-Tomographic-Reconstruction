# *******************************************************
# * FILE: generateallpositions.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

import numpy as np


class GenerateEveryPossibleVolumePositions:
    """Angular measurements matrix
    This function calculate all possible easypet line of responses"""

    def __init__(self, rangeTopMotor=60, begin_range_botMotor=0, end_rangeBotMotor=360, stepTopmotor=0.9, stepBotMotor= 3.6,
                 number_of_crystals=[32, 2], reduction = True):
        self.rangeTopMotor = rangeTopMotor
        self.begin_range_botMotor = begin_range_botMotor
        self.end_rangeBotMotor = end_rangeBotMotor
        self.stepTopmotor = stepTopmotor
        self.stepBotMotor = stepBotMotor
        self.number_of_crystals = number_of_crystals
        self.reduction = reduction



    def matrix_without_reduction(self):
        topmotors_position = np.arange(-self.rangeTopMotor / 2, self.rangeTopMotor / 2 + self.stepTopmotor, self.stepTopmotor,
                                            dtype=np.float32)
        botmotors_position = np.arange(self.begin_range_botMotor, self.end_rangeBotMotor, self.stepBotMotor, dtype=np.float32)
        sideA_id = np.arange(1, self.number_of_crystals[0] * self.number_of_crystals[1] + 1, dtype=np.int8)
        sideB_id = np.arange(1, self.number_of_crystals[0] * self.number_of_crystals[1] + 1, dtype=np.int8)

        self.every_possible_position_array = np.zeros(
            (len(topmotors_position) * len(botmotors_position) * len(sideA_id) * len(sideB_id), 4), dtype=np.float32)
        self.every_possible_position_array[:, 3] = np.repeat(topmotors_position,
                                                             len(botmotors_position) * len(sideA_id) * len(
                                                                 sideB_id))
        self.every_possible_position_array[:, 2] = np.tile(
            np.repeat(botmotors_position, len(sideA_id) * len(sideB_id)), len(topmotors_position))
        self.every_possible_position_array[:, 0] = np.tile(np.repeat(sideA_id, len(sideB_id)),
                                                           len(topmotors_position) * len(botmotors_position))
        self.every_possible_position_array[:, 1] = np.tile(sideB_id, len(topmotors_position) * len(
            botmotors_position) * len(sideB_id))

    def matrix_with_reduction(self):
        number_of_blocks = 2
        total_number_id_positions = 124  # block_shape[0] * block_shape[1] * number_of_blocks
        self.every_possible_position_array = np.zeros(
            (len(self.topmotors_position) * len(self.botmotors_position) * total_number_id_positions, 4))

        self.total_possible_id = np.zeros((len(self.sideA_id) * len(self.sideB_id), 2))
        self.total_possible_id[:, 0] = np.repeat(self.sideA_id, len(self.sideB_id))
        self.total_possible_id[:, 1] = np.tile(self.sideB_id, len(self.sideA_id))

        index_ida = np.where(self.total_possible_id[:, 0] <= 2)
        index_idb = np.where(self.total_possible_id[:, 1] <= 2)
        # # # #index_a4_100 = np.where(listMode[:, 3] < 1)
        indexes_intersection = np.union1d(index_ida, index_idb)

        self.total_possible_id = self.total_possible_id[indexes_intersection]
        # reading_data = np.delete(reading_data, indexes_intersection_B, axis=0)

        self.every_possible_position_array[:, 2] = np.repeat(self.botmotors_position,
                                                             total_number_id_positions * len(self.topmotors_position))

        self.every_possible_position_array[:, 3] = np.tile(np.repeat(self.topmotors_position,
                                                                     total_number_id_positions),
                                                           len(self.botmotors_position))

        # self.every_possible_position_array[:, 1] = np.tile(np.repeat(self.total_possible_id[:,0], len(self.total_possible_id[:,0])),
        #                                                    len(self.topmotors_position) * len(self.botmotors_position))
        self.every_possible_position_array[:, 0] = np.tile(self.total_possible_id[:, 0],
                                                           len(self.topmotors_position) * len(
                                                               self.botmotors_position))

        self.every_possible_position_array[:, 1] = np.tile(self.total_possible_id[:, 1],
                                                           len(self.topmotors_position) * len(
                                                               self.botmotors_position))
        #
        #
        # self.every_possible_position_array = self.every_possible_position_array[self.every_possible_position_array[:,0]==self.every_possible_position_array[:,1]-2,:]

        # self.every_possible_position_array[1, 0:2] = 5
        #self.every_possible_position_array[0, 0:2] = 32

