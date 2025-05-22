import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class AutomaticNormalization:
    def __init__(self, tor_file=None):
        if tor_file is None:
            raise FileExistsError("tor_file  cannot be None")

        self.torFile = tor_file
        # self.

    def normalizationLM(self):
        pass


class DualRotationNormalizationSystem(AutomaticNormalization):
    def __init__(self, tor_file=None, getNumberOfDetectorsFromFile=True):
        super().__init__(tor_file=tor_file)
        self._fanMotorStep = self.torFile.fileBodyData.minDiff[
            self.torFile.fileBodyData.listmodeFields.index("FAN_MOTOR")]
        self._axialMotorStep = self.torFile.fileBodyData.minDiff[
            self.torFile.fileBodyData.listmodeFields.index("AXIAL_MOTOR")]
        self._rangeTopMotor = (np.abs(self.torFile.fileBodyData.max[self.torFile.fileBodyData.
                                      listmodeFields.index("FAN_MOTOR")]) +
                               np.abs(self.torFile.fileBodyData.min[self.torFile.fileBodyData.
                                      listmodeFields.index("FAN_MOTOR")]))
        self._beginAxialMotorPosition = self.torFile.fileBodyData.max[
            self.torFile.fileBodyData.listmodeFields.index("AXIAL_MOTOR")]
        self._endAxialMotorPosition = self.torFile.fileBodyData.min[
            self.torFile.fileBodyData.listmodeFields.index("AXIAL_MOTOR")]
        if getNumberOfDetectorsFromFile:
            self._numberOfDetectorsSideA = int(self.torFile.fileBodyData.max[
                                               self.torFile.fileBodyData.listmodeFields.index("IDA")] + 1)
            self._numberOfDetectorsSideB = int(self.torFile.fileBodyData.max[
                                               self.torFile.fileBodyData.listmodeFields.index("IDB")] + 1)
        else:
            self._numberOfDetectorsSideA = int(self.torFile.systemInfo.numberOfDetectorModulesSideA *
                                            self.torFile.systemInfo.detectorModulesSideA[0].
                                            totalNumberHighEnergyLightDetectors)

            self._numberOfDetectorsSideB = int(self.torFile.systemInfo.numberOfDetectorModulesSideB *
                                            self.torFile.systemInfo.detectorModulesSideB[0].
                                            totalNumberHighEnergyLightDetectors)

        self._calculateAllPositionsAllDetectors = True
        self._percentageOfDetectorsCalculated = 0.2
        self._generateMotorPositions = True
        self._energyPeakKey = None
        self._listModeForNormalization = None
        self._numberOfEventsListMode = None
        self._fieldsListMode = None
        self._tiledProbabilityOfDetection = None

    @property
    def tiledProbabilityOfDetection(self):
        """
        Get the tiled probability of detection
        :return:
        """
        return self._tiledProbabilityOfDetection

    @property
    def fieldsListMode(self):
        """
        Get the fields list mode
        :return:
        """
        return self._fieldsListMode

    @property
    def numberOfDetectorsSideA(self):
        return self._numberOfDetectorsSideA

    @property
    def numberOfDetectorsSideB(self):
        return self._numberOfDetectorsSideB

    @property
    def numberOfEventsListMode(self):
        return self._numberOfEventsListMode

    @property
    def listModeForNormalization(self):
        return self._listModeForNormalization


    @property
    def energyPeakKey(self):
        """
        Get the energy peak key
        :return:
        """
        return self._energyPeakKey

    def setEnergyPeakKey(self, energyPeakKey):
        """
        Set the energy peak key
        :param energyPeakKey:
        :return:
        """
        if not isinstance(energyPeakKey, str):
            raise ValueError("energyPeakKey must be a string")
        self._energyPeakKey = energyPeakKey

    def printMotorVariables(self):
        """
        Print the motor variables
        :return:
        """
        print("Fan Motor Step: ", self._fanMotorStep)
        print("Axial Motor Step: ", self._axialMotorStep)
        print("Range Top Motor: ", self._rangeTopMotor)
        print("Begin Axial Motor Position: ", self._beginAxialMotorPosition)
        print("End Axial Motor Position: ", self._endAxialMotorPosition)
        print("Calculate All Positions All Detectors: ", self._calculateAllPositionsAllDetectors)
        print("Generate Motor Positions: ", self._generateMotorPositions)

    @property
    def fanMotorStep(self):
        """
        Fan motor step
        :return:
        """
        return self._fanMotorStep

    @property
    def axialMotorStep(self):
        """
        Axial motor step
        :return:
        """
        return self._axialMotorStep

    @property
    def calculateAllPositionsAllDetectors(self):
        """
        Calculate all positions for all detectors. If false the unique positions are determined
        :return:
        """
        return self._calculateAllPositionsAllDetectors

    def setCalculateAllPositionsAllDetectors(self, value):
        """
        Set calculate all positions for all detectors
        :param value:
        :return:
        """
        if not isinstance(value, bool):
            raise ValueError("value must be a boolean")
        self._calculateAllPositionsAllDetectors = value

    @property
    def generateMotorPositions(self):
        """
        Generate motor positions
        :return:
        """
        return self._generateMotorPositions

    def setGenerateMotorPositions(self, value):
        """
        Set generate motor positions
        :param value:
        :return:
        """
        if not isinstance(value, bool):
            raise ValueError("value must be a boolean")
        self._generateMotorPositions = value

    def normalizationLM(self):
        """
        Normalization for the dual rotation system
        :return:
        """
        if self.torFile.fileBodyData.listmodeFields is None:
            raise ValueError("listmodeFields cannot be None")
        if self.torFile.fileBodyData.listmodeFields == 0:
            raise ValueError("listmodeFields cannot be 0")

        if self._generateMotorPositions:

            # used for pre-calculation_normalization
            fanMotor = np.arange(-self._rangeTopMotor / 2, self._rangeTopMotor / 2 + self._fanMotorStep,
                                 self._fanMotorStep)
            if self._beginAxialMotorPosition > self._endAxialMotorPosition:
                self._beginAxialMotorPosition, self._endAxialMotorPosition = self._endAxialMotorPosition, \
                                                                             self._beginAxialMotorPosition
            axialMotor = np.arange(self._beginAxialMotorPosition, self._endAxialMotorPosition,
                                   self._axialMotorStep)
        else:
            fanMotor = np.unique(self.torFile.fileBodyData[self.torFile.fileBodyData.listmodeFields.index("FAN_MOTOR")])
            axialMotor = np.unique(
                self.torFile.fileBodyData[self.torFile.fileBodyData.listmodeFields.index("AXIAL_MOTOR")])

        index_ = self.torFile.calibrations.systemSensitivity.fields.index(self._energyPeakKey)
        detectorSensitivity = self.torFile.calibrations.systemSensitivity.probabilityOfDetection[index_]
        detectorSensitivity = detectorSensitivity / np.sum(detectorSensitivity)

        if self.torFile.systemInfo.deviceType == "CT":
            self._fieldsListMode = ["AXIAL_MOTOR","FAN_MOTOR", "IDB"]
            self._listModeForNormalization = np.ones(
                (len(fanMotor) * len(axialMotor) * self._numberOfDetectorsSideB, 3), dtype=np.float32)

            self._listModeForNormalization[:, 1] = np.repeat(fanMotor, len(axialMotor) * self._numberOfDetectorsSideB)

            if self._calculateAllPositionsAllDetectors:

                self._listModeForNormalization[:, 0] = np.tile(
                    np.repeat(axialMotor, self._numberOfDetectorsSideB), len(fanMotor))
                value = np.random.choice((self._numberOfDetectorsSideB), len(self._listModeForNormalization),
                                         p=detectorSensitivity)

                # self._listModeForNormalization[:, 2] = value
                self._listModeForNormalization[:,2] = np.tile(np.arange(0,self._numberOfDetectorsSideB),len(axialMotor)*len(fanMotor))
                self._tiledProbabilityOfDetection = np.tile(detectorSensitivity,len(axialMotor)*len(fanMotor))

        return self._listModeForNormalization

    def __getitem__(self, key):
        """
        Get an item from the listmode data
        """
        if isinstance(key, str):
            return self._listModeForNormalization[:, self._fieldsListMode.index(key)]
        return self._listModeForNormalization[key]





class NormalizationCT:
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
                                                  'system_configurations',
                                                  'x_{}__y_{}'.format(self.number_of_crystals[0],
                                                                      self.number_of_crystals[1]),
                                                  'crystals_detection_probability.npy')
        self.probability_top_file_path = os.path.join(self.main_dir,
                                                      'system_configurations',
                                                      'x_{}__y_{}'.format(self.number_of_crystals[0],
                                                                          self.number_of_crystals[1]),
                                                      'top_motor_probability.npy')
        self.top_positions_file_path = os.path.join(self.main_dir,
                                                    'system_configurations',
                                                    'x_{}__y_{}'.format(self.number_of_crystals[0],
                                                                        self.number_of_crystals[1]),
                                                    'top_motor_positions.npy')

    def write_probability_phantom(self):
        """

        """

        total_crystals = self.number_of_crystals[0] * self.number_of_crystals[1]
        probability = \
            np.histogram((self.data_in[:, 2] - 1) * total_crystals + (self.data_in[:, 3] - 1),
                         total_crystals ** 2,
                         (0, total_crystals ** 2))[0] / len(self.data_in)
        np.save(self.probability_file_path, probability)

        top = np.unique(self.data_in[:, 5])
        probability_top = np.histogram(self.data_in[:, 5], len(top))[0] / len(self.data_in)

        np.save(self.probability_top_file_path, probability_top)
        np.save(self.top_positions_file_path, top)
        return probability

    def matrix_without_reduction(self):
        top = np.arange(-self.rangeTopMotor / 2, self.rangeTopMotor / 2 + self.stepTopmotor, self.stepTopmotor)
        bot = np.arange(self.begin_range_botMotor, self.end_rangeBotMotor, self.stepBotMotor)

        self.sideB_id = np.arange(1, 33)
        self.sideA_id = np.array([1])
        number_of_crystals_B = 32
        number_of_crystals_A = 1

        self.reading_data = np.ones(
            (len(top) * len(bot) * len(self.sideA_id) * len(self.sideB_id), 7), dtype=np.float32)
        self.reading_data[:, 5] = np.tile(np.repeat(top, len(self.sideA_id) * len(self.sideB_id)), len(bot))
        self.reading_data[:, 4] = np.repeat(bot, len(self.sideA_id) * len(self.sideB_id) * len(top))
        self.reading_data[:, 2] = np.tile(np.repeat(self.sideA_id, len(self.sideB_id)), len(top) * len(bot))
        self.reading_data[:, 3] = np.tile(self.sideB_id, len(self.sideA_id) * len(bot) * len(top))
        self.reading_data[:, 0] = 59.5
        self.reading_data[:, 1] = 59.5
        print(self.reading_data)

    def probability_uniform_phantom(self):
        # try:
        #     self._probability_uniform_phantom = np.load(self.probability_file_path)
        # except FileNotFoundError:
        # self._probability_uniform_phantom = self.write_probability_phantom()

        try:
            self._probability_uniform_phantom = np.load(self.probability_file_path)

            print("probabi")
        except FileNotFoundError:
            comb = int((self.number_of_crystals[0] * self.number_of_crystals[1]) ** 2)
            self._probability_uniform_phantom = np.ones(comb) / comb
            # np.save(self.probability_file_path, self._probability_uniform_phantom)
            # print("write")
        return self._probability_uniform_phantom

    def normalizationLM(self):
        """

        """
        if self.data_in is None:
            # used for pre-calculation_normalization
            top = np.arange(-self.rangeTopMotor / 2, self.rangeTopMotor / 2 + self.stepTopmotor, self.stepTopmotor)
            bot = np.arange(self.begin_range_botMotor, self.end_rangeBotMotor, self.stepBotMotor)
        else:
            top = np.unique(self.data_in[:, 5])
            bot = np.unique(self.data_in[:, 4])

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
            (len(top) * len(bot) * 32, 4), dtype=np.float32)
        # self.sideA_id = np.arange(1, number_of_crystals[0] * number_of_crystals[1] + 1, dtype=np.int8)
        # self.sideB_id = np.arange(1, number_of_crystals[0] * number_of_crystals[1] + 1, dtype=np.int8)
        # self.sideA_id = np.arange(1, np.max(self.data_in[:, 2]) - np.min(self.data_in[:, 2]) + 1, dtype=np.int8)
        # self.sideB_id = np.arange(1, np.max(self.data_in[:, 2]) - np.min(self.data_in[:, 2]) + 1, dtype=np.int8)
        # self.matrix_without_reduction()
        # number_of_reps= 400

        self.reading_data[:, 1] = np.repeat(top, len(bot) * 32)
        # probability_top = np.load(self.probability_top_file_path)
        probability_top = np.random.uniform(0, 1, len(top))
        # top_norm = np.load(self.top_positions_file_path)

        # inter_top_positions = interp1d(top_norm, probability_top, fill_value="extrapolate")
        # top_new_conditions = np.unique(top)
        # probability_top = inter_top_positions(top_new_conditions)
        # probability_top += np.abs(probability_top.min())
        # probability_top /= np.sum(probability_top)
        # self.reading_data[:, 1] = np.random.choice(len(probability_top), len(self.reading_data),
        #                                            p=probability_top) * self.stepTopmotor - top.max()

        # uniform distribution in top
        # self.reading_data[:, 1] = np.random.uniform(top.min(), top.max(), len(self.reading_data))
        self.reading_data[:, 0] = np.tile(
            np.repeat(bot, 32), len(top))
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
        # prob = self.probability_uniform_phantom()
        # value = np.random.choice((self.number_of_crystals[0] * self.number_of_crystals[1]), len(self.reading_data),
        #                          p=prob)
        # self.reading_data[:, 3] = np.array(value % total_crystals + 1, dtype=np.int32)
        # self.reading_data[:, 2] = np.random.uniform(0,32, len(self.reading_data))
        self.sideB_id = np.arange(0, 32)
        self.reading_data[:, 0] = np.tile(
            np.repeat(bot, 32), len(top))
        self.reading_data[:, 2] = np.tile(np.tile(self.sideB_id, len(bot)), len(top))
        # self.reading_data[:,3] = np.random.choice(len(unique_values)-1, len(self.reading_data), p=probability)
        # self.reading_data[:, 3] = np.array((value // total_crystals) + 1,
        #                                    dtype=np.int32)
        # # # d = np.random.choice(len(t), len(self.reading_data), p=b / np.sum(b))
        # # id =
        # self.reading_data[:,2:4]=t[d]
        # self.reading_data[:, 0] = 511
        # self.reading_data[:, 1] = 511

        self.total_counts = len(self.reading_data)
        # np.save(os.path.join(self.main_dir, "outputs", "listmode_normalization_test.npy"), self.reading_data)

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


if __name__ == "__main__":
    normalization = AdaptiveNormalizationMatrix(number_of_crystals=32,
                                                rangeTopMotor=108, begin_range_botMotor=0, end_rangeBotMotor=360,
                                                stepTopmotor=0.225, stepBotMotor=1.8, recon_2D=False)
    normalization.matrix_without_reduction()
