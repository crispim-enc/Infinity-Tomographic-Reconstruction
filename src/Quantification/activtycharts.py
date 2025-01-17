import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from src.ImageReader import RawDataSetter
from src.EasyPETLinkInitializer.EasyPETDataReader import binary_data
from src.Corrections.PET.DecayCorrection import DecayCorrection


class ActivityMeasuredVsTheo:
    def __init__(self):
        [self.listMode, Version_binary, header, dates, otherinfo, self.acquisitionInfo, stringdata,
         self.systemConfigurations_info, energyfactor_info, peakMatrix_info] = binary_data().open(easypet_file)
        self.crystals_geometry = [self.systemConfigurations_info["array_crystal_x"],
                                  self.systemConfigurations_info["array_crystal_y"]]
        data = RawDataSetter(file_name=file_path, )
        # data = RawDataSetter(file_name=file_path, size_file_m=[88, 88, 130])
        data.read_files()
        volume = data.volume
        real_pixelSizeXYZ = (self.systemConfigurations_info["array_crystal_x"] * self.systemConfigurations_info[
            "crystal_pitch_x"] +
                             (self.systemConfigurations_info["array_crystal_x"] - 1) *
                             2 * self.systemConfigurations_info["reflector_interior_A_x"]) / volume.shape[2]
        self.volume_voxel = 1 * 1 * real_pixelSizeXYZ

        time_indexes = self.acquisitionInfo["Turn end index"]
        cg = [self.systemConfigurations_info["array_crystal_x"], self.systemConfigurations_info["array_crystal_y"]]
        counts = np.zeros(len(time_indexes)-1)
        counts_with_decay_correction = np.zeros(len(time_indexes)-1)
        correct_activity_vector = np.zeros(len(time_indexes)-1)
        self.decay_correction_class = DecayCorrection(listMode=self.listMode,
                                                      acquisition_info=self.acquisitionInfo,
                                                      correct_decay=True)

        self.decay_correction_class.list_mode_decay_correction()
        decay_factor_cutted = self.decay_correction_class.decay_factor
        # self.decay_correction_class.list_mode_decay_correction()

        for i in range(len(time_indexes)-6):

            listMode_temp = self.listMode[time_indexes[i]:time_indexes[-5]]
            self.scan_time = listMode_temp[-1, 6] - listMode_temp[0, 6]
            print(listMode_temp[-1, 6] )
            counts[i] = len(listMode_temp)/self.scan_time

            self.decay_correction_class = DecayCorrection(listMode=listMode_temp,
                                                          acquisition_info=self.acquisitionInfo,
                                                          correct_decay=True)
            corrected_activity = self.decay_correction_class.activity_on_subject_at_scanning_time()

            self.decay_correction_class.list_mode_decay_correction()
            decay_factor_cutted = self.decay_correction_class.decay_factor

            # counts_with_decay_correction[i] = np.sum(decay_factor_cutted[time_indexes[i]:time_indexes[-350]])
            counts_with_decay_correction[i] = np.sum(decay_factor_cutted)/self.scan_time
            counts[i] = len(listMode_temp)/self.scan_time
            # counts_with_decay_correction[i] = DecayCorrection._calculate_initial_activity(
            #     counts[i], listMode_temp[-1, 6], self.decay_correction_class.decay_half_life)

            correct_activity_vector[i] = corrected_activity
            # f = FactorQuantificationFromUniformPhantom(activity_phantom=corrected_activity,
            #                                            radiotracer_phantom=self.acquisitionInfo['Tracer'],
            #                                            positron_fraction_phantom=self.acquisitionInfo[
            #                                                'Positron Fraction'],
            #                                            phantom_volume=float(self.acquisitionInfo["Volume tracer"]),
            #                                            crystals_geometry=self.crystals_geometry,
            #                                            image_phantom=volume,
            #                                            acquisition_phantom_duration=self.scan_time,
            #                                            voxel_volume=self.volume_voxel, voxel_volume_unit="ml")
            # f.segment_region_phantom(voi= 8)
            # f.quantification_factor_calculation(bq_ml=True)
        plt.plot(decay_factor_cutted)
        plt.figure()
        ax = plt.axes()
        # ax.twinx()
        plt.plot(correct_activity_vector/37000, counts_with_decay_correction,".", label="with decay correction")
        plt.plot(correct_activity_vector/37000, counts,".", label="without decay correction")
        plt.legend()
        # plt.yscale('log')

        # secax = ax.secondary_xaxis('left', functions=(correct_activity_vector/37000, counts))
        # secax.xaxis.set_minor_locator(AutoMinorLocator())
        # secax.set_xlabel('$X_{other}$')


        plt.xlabel("Activity predicted $\mu$Ci")
        plt.ylabel("CPS")
        plt.show()

    def remove_turns_for_reconstruction(self, ):
        """
        Generalizar para timestamps !!
        """
        time_init_cut = self.remove_turns["Init time"]
        time_end_cut = self.remove_turns["End time"]
        time_indexes = self.acquisitionInfo["Turn end index"]
        time_per_turns = self.reading_data[time_indexes, 6]
        diff_init_time = np.abs(time_init_cut - time_per_turns)
        diff_init_index = time_indexes[np.where(diff_init_time == np.min(diff_init_time))[0][0]]
        diff_end_time = np.abs(time_end_cut - time_per_turns)
        diff_end_index = time_indexes[np.where(diff_end_time == np.min(diff_end_time))[0][0]]
        reading_data_cutted = self.reading_data[diff_init_index:diff_end_index, :]
        decay_factor_cutted = self.decay_factor[diff_init_index:diff_end_index]
        print("Image cutted between: {} and {} s".format(reading_data_cutted[0, 6], reading_data_cutted[-1, 6]))
        return reading_data_cutted, decay_factor_cutted

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    # matplotlib.rcParams['font.family'] = "Gill Sans MT"
    file_path = filedialog.askopenfilename()
    easypet_folder = os.path.dirname(os.path.dirname(file_path))
    easypet_file = os.path.join(easypet_folder, "{}.easypet".format(os.path.basename(easypet_folder)))
    # easypet_file = file_path
    a = ActivityMeasuredVsTheo()
    # file_folder = path.join(file_folder, "static_image")



