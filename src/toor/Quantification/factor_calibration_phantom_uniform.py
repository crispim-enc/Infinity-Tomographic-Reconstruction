from array import array
import json
import numpy as np
import matplotlib.pyplot as plt

"""
Not integrated in the TOOR. For PET quantification factor calculation used the intelligent scan project
"""


class FactorQuantificationFromUniformPhantom:
    def __init__(self, activity_phantom=6663944, radiotracer_phantom="F18", positron_fraction_phantom=1,
                 positron_fraction_subject=1, phantom_volume=21,
                 radiotracer_subject="Cu64", image_phantom=None, acquisition_phantom_duration=None,
                 voxel_volume=None, voxel_volume_unit="ml", crystals_geometry=None):

        if crystals_geometry is None:
            crystals_geometry = [32, 2]
        self.counts_p_s_voxel = None
        self.counts_p_s_ml = None
        self.activity_phantom = activity_phantom
        self.radiotracer_phantom = radiotracer_phantom
        self.phantom_volume = phantom_volume
        self.image_phantom = image_phantom
        self.radiotracer_subject = radiotracer_subject
        self.acquisition_phantom_duration = acquisition_phantom_duration
        self.segmented_image = None
        self.concentration_phantom = None
        self.quantification_factor = None

        self.factor_phantom_radiotracer = float(positron_fraction_phantom)
        if self.factor_phantom_radiotracer == 0:
            self.factor_phantom_radiotracer = 0.967  # default F18

        self.factor_subject_radiotracer = float(positron_fraction_subject)
        if self.factor_subject_radiotracer == 0:
            self.factor_subject_radiotracer = 0.967

        if voxel_volume_unit == "mm^3":
            self.voxel_volume = voxel_volume * 0.001
        elif voxel_volume_unit == "ml":
            self.voxel_volume = voxel_volume
        else:
            print("please specify voxel volume units")
            return
        self.quantification_factor_per_ml = None
        self.default_system_quantification_factor = None
        self.file_name = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                      "system_configurations",
                                      "x_{}__y_{}".format(crystals_geometry[0], crystals_geometry[1]),
                                      "quatification_factor")
        self.radioisotope_dict = {
                                  "F18": {"Half-life": 6582.66,
                                          "Number decays": 1,
                                          "Decay 1 type": "beta+",
                                          "Decay 1 Energy": 633,
                                          "Decay 1 Percentage": 0.9886,
                                          },

                                  "Na22": {"Half-life": 6582.66,
                                           "Number decays": 2,
                                           "Decay 1 Energy": 546,
                                           "Decay 1 Percentage": 0.898,
                                           "Decay 2 Energy": 1254,
                                           "Decay 2 Percentage": 0.05},

                                  "Cu64": {"Half-life": 45720,
                                           "Number decays": 3,
                                           "Decay 1 type": "gamma",
                                           "Decay 1 Energy": 1346,
                                           "Decay 1 Percentage": 0.54,
                                           "Decay 2 type": "beta+",
                                           "Decay 2 Energy": 653,
                                           "Decay 2 Percentage":  0.178,
                                           "Decay 3 type": "beta-",
                                           "Decay 3 Energy": 579,
                                           "Decay 3 Percentage": 0.3848}
                                }

    def segment_region_phantom(self, voi=None, dx=5, dy=5, dz=5):
        if voi is None:
            self.segmented_image = self.image_phantom[int(self.image_phantom.shape[0]/2-dx):int(self.image_phantom.shape[0]/2+dx),
                                            int(self.image_phantom.shape[1]/2-dy):int(self.image_phantom.shape[1]/2+dy),
                                            int(self.image_phantom.shape[2]/2-dz):int(self.image_phantom.shape[2]/2+dz)]

        else:
            number_voxels = voi / self.voxel_volume
            number_voxels_per_side = int(number_voxels ** (1 / 3) / 2)
            self.segmented_image = self.image_phantom[
                               int(self.image_phantom.shape[0] / 2 - number_voxels_per_side):int(
                                   self.image_phantom.shape[0] / 2 + number_voxels_per_side),
                               int(self.image_phantom.shape[1] / 2 - number_voxels_per_side):int(
                                   self.image_phantom.shape[1] / 2 + number_voxels_per_side),
                               int(self.image_phantom.shape[2] / 2 - number_voxels_per_side):int(
                                   self.image_phantom.shape[2] / 2 + number_voxels_per_side)]

            self.mask = np.zeros(self.image_phantom.shape)
            self.mask[int(self.image_phantom.shape[0] / 2 - number_voxels_per_side):int(
                                   self.image_phantom.shape[0] / 2 + number_voxels_per_side),
                               int(self.image_phantom.shape[1] / 2 - number_voxels_per_side):int(
                                   self.image_phantom.shape[1] / 2 + number_voxels_per_side),
                               int(self.image_phantom.shape[2] / 2 - number_voxels_per_side):int(
                                   self.image_phantom.shape[2] / 2 + number_voxels_per_side)] = 1
            print("number_voxels: {}".format(number_voxels))
            print("number_voxels_per_side: {}".format(number_voxels_per_side))
        # l = (voi*1000)^(1/3)

    def quantification_factor_calculation(self, bq_ml=False):
        # self.counts_p_s_voxel = np.mean(self.segmented_image)/self.acquisition_phantom_duration
        if bq_ml:
            self.counts_p_s_voxel = 0
            self.counts_p_s_ml = np.mean(self.segmented_image)
        else:
            self.counts_p_s_voxel = np.mean(self.segmented_image)
            self.counts_p_s_ml = self.counts_p_s_voxel / self.voxel_volume
        self.concentration_phantom = self.factor_phantom_radiotracer * self.activity_phantom / self.phantom_volume
        self.quantification_factor = self.concentration_phantom / self.counts_p_s_ml
        self.quantification_factor_per_ml = self.quantification_factor / self.voxel_volume

        print("Phantom Mean: {} Counts/s/voxel".format(np.mean(self.counts_p_s_voxel)))
        print("Phantom Mean: {} Bq/ml".format(np.mean(self.counts_p_s_ml)))
        print("Factor_phantom_radiotracer: {}".format(self.factor_phantom_radiotracer))
        print("Activity phantom: {} Bq".format(self.activity_phantom))
        print("Activity Phantom concentration {} Bq/ml".format(self.concentration_phantom))
        print("Quantification Factor {} ".format(self.quantification_factor))
        print("Quantification Factor per mm3 {} ".format(self.quantification_factor_per_ml))

    def save_info(self):
        dict = {"Phantom Mean (Counts/s/voxel)": str(self.counts_p_s_voxel),
                "Phantom Mean (Counts/s/ml)": str(self.counts_p_s_ml),
                "Factor phantom radiotracer": str(self.factor_phantom_radiotracer),
                "Activity phantom": str(self.activity_phantom),
                "Activity Phantom concentration (Bq/ml)": str(self.concentration_phantom),
                "Quantification Factor": str(self.quantification_factor),
                "Quantification Factor per ml": str(self.quantification_factor_per_ml)}

        calibrationfactor = json.dumps(dict)
        calibrationfactor_info = array('u', calibrationfactor)
        calibrationfactor_info_size = [len(calibrationfactor_info)]
        calibrationfactor_info_size = array('i', calibrationfactor_info_size)
        with open(self.file_name, 'wb') as output_file:
            calibrationfactor_info_size.tofile(output_file)
            calibrationfactor_info.tofile(output_file)

    def load_info(self):
        """ """
        try:
            with open(self.file_name, "rb") as binary_file:
                size_header = np.fromfile(binary_file, dtype=np.int32, count=1)
                binary_file.seek(2)
                quantificationfactor = np.fromfile(binary_file, dtype='|S1', count=size_header[0] * 2 + 2).astype('|U1')
                quantificationfactor = quantificationfactor.tolist()
                quantificationfactor = ''.join(quantificationfactor)
                quantificationfactor = json.loads(quantificationfactor)
            self.default_system_quantification_factor = quantificationfactor
            self.quantification_factor = float(self.default_system_quantification_factor
                                               ["Quantification Factor per ml"]) * self.voxel_volume
            # self.quantification_factor = float(self.default_system_quantification_factor
            #                                    ["Quantification Factor"])
        except:
            print("No quantification factor found")
            self.quantification_factor = 1


    def calculation_radiotracer_factors(self, it_measure_only_beta_plus=False):
        if it_measure_only_beta_plus:
            self.factor_phantom_radiotracer = 1
            self.factor_subject_radiotracer = 1
        else:
            for i in range(self.radioisotope_dict[self.radiotracer_phantom]["Number decays"]):
                if self.radioisotope_dict[self.radiotracer_phantom]["Decay {} type".format(i)] == "beta+":
                    self.factor_phantom_radiotracer += self.radioisotope_dict[self.radiotracer_phantom][
                        "Decay {} Percentage".format(i)]

            for i in range(self.radioisotope_dict[self.radiotracer_phantom]["Number decays"]):
                if self.radioisotope_dict[self.radiotracer_phantom]["Decay {} type".format(i)] == "beta+":
                    self.factor_subject_radiotracer += self.radioisotope_dict[self.radiotracer_phantom][
                        "Decay {} Percentage".format(i)]

    def apply_quantification_factor_to_image(self, image):
        """ """
        return image*self.quantification_factor*self.factor_subject_radiotracer

    def plt_segmented_image(self):
        plt.figure()
        plt.imshow(np.max(self.segmented_image, axis=0))
        plt.show()


if __name__ == "__main__":
    import os
    import tkinter as tk
    from tkinter import filedialog
    from ImageReader import RawDataSetter
    from EasyPETLinkInitializer.EasyPETDataReader import binary_data

    root = tk.Tk()
    root.withdraw()
    # matplotlib.rcParams['font.family'] = "Gill Sans MT"
    file_path = filedialog.askopenfilename()
    easypet_folder = os.path.dirname(os.path.dirname(file_path))
    easypet_file = os.path.join(easypet_folder, "{}.easypet".format(os.path.basename(easypet_folder)))
    # file_folder = path.join(file_folder, "static_image")

    [listMode, Version_binary, header, dates, otherinfo, acquisitionInfo, stringdata,
     systemConfigurations_info, energyfactor_info, peakMatrix_info] = binary_data().open(easypet_file)
    data = RawDataSetter(file_name=file_path,)
    # data = RawDataSetter(file_name=file_path, size_file_m=[88, 88, 130])
    data.read_files()
    volume = data.volume
    real_pixelSizeXYZ = (systemConfigurations_info["array_crystal_x"] * systemConfigurations_info["crystal_pitch_x"] +
                         (systemConfigurations_info["array_crystal_x"] - 1) *
                         2*systemConfigurations_info["reflector_interior_A_x"]) / volume.shape[2]
    volume_voxel = 1 * 1 * real_pixelSizeXYZ*0.001

    cg = [systemConfigurations_info["array_crystal_x"], systemConfigurations_info["array_crystal_y"]]
    initial_activity = acquisitionInfo["Total Dose"]
    f = FactorQuantificationFromUniformPhantom(activity_phantom=initial_activity,
                                               radiotracer_phantom=acquisitionInfo['Tracer'],
                                               positron_fraction_phantom=acquisitionInfo['Positron Fraction'],
                                               phantom_volume=float(acquisitionInfo["Volume tracer"]),
                                               crystals_geometry=cg,
                                               image_phantom=volume,
                                               acquisition_phantom_duration=listMode[-1, 6],
                                               voxel_volume=volume_voxel, voxel_volume_unit="ml")

    # f = FactorQuantificationFromUniformPhantom(activity_phantom=  crystals_geometry=cg ,image_phantom=volume,
    #                                            acquisition_phantom_duration=listMode[-1, 6],
    #                                            voxel_volume=volume_voxel, voxel_volume_unit="mm^3")
    f.segment_region_phantom(dx=10, dy=10, dz=10)
    f.quantification_factor_calculation(bq_ml=True)
    # f.save_info()

    f.plt_segmented_image()
