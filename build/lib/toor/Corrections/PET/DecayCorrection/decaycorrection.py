# *******************************************************
# * FILE: decaycorrection.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

import time
import numpy as np


class DecayCorrection:
    def __init__(self, listMode=None, correct_decay=True, delay_time=0,
                 decay_half_life=None, radioisotope="Na22", acquisition_info=None):

        self.listMode = listMode
        self.correct_decay = correct_decay
        self.delay_time = delay_time
        self.decay_half_life = decay_half_life
        self.decay_factor = None
        self.radioisotope = radioisotope
        self.radioisotope_dict = {
            "F18": {"Half-life": 6582.66,
                    "Number decays": 1,
                    "Decay 1 type": "gamma",
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
                     "Decay 2 Percentage": 0.178,
                     "Decay 3 type": "beta-",
                     "Decay 3 Energy": 579,
                     "Decay 3 Percentage": 0.3848}
        }
        self.acquisition_info = acquisition_info
        self._activity_on_subject = None

        self._activity_on_subject_at_injection_time = None
        self._activity_on_subject_at_scanning_time = None
        self._initial_activity_at_injection_time = None
        self._residual_activity_at_injection_time = None

        if acquisition_info is None:
            self.decay_half_life = self.radioisotope_dict[self.radioisotope]["Half-life"]
        else:
            self.decay_half_life = float(acquisition_info["Half life"])
            if self.acquisition_info["Type of subject"] == "Quick Scan" or \
                    self.acquisition_info["Type of subject"] == "Radioactive source":
                self._initial_activity = 0
                self._residual_activity = 0
                self.deltatime_injection_and_sbegin = 0
                self.deltatime_injection_and_residual = 0
                self.deltatime_initial_and_injection = 0
            else:
                self._initial_activity = float(acquisition_info["Total Dose"])

                # if time.mktime(
                #         time.strptime(
                #             acquisition_info['Acquisition start time'], '%d %b %Y - %Hh %Mm %Ss')) - time.mktime(
                #     time.strptime("14 Mar 2022 - 00h 00m 00s", '%d %b %Y - %Hh %Mm %Ss'))<0:
                #     self._initial_activity = float(acquisition_info["Total Dose"])/1000
                if not 'Injection date time' in acquisition_info:
                    acquisition_info['Injection date time'] = acquisition_info['End date time']

                self._residual_activity = float(acquisition_info["Residual Activity"])

                self.deltatime_injection_and_sbegin = time.mktime(
                    time.strptime(acquisition_info['Acquisition start time'], '%d %b %Y - %Hh %Mm %Ss')) - time.mktime(
                    time.strptime(acquisition_info['Injection date time'], '%d.%m.%y %H:%M:%S'))+self.listMode[0,6]

                self.deltatime_injection_and_residual = time.mktime(time.strptime(
                    acquisition_info['Injection date time'], '%d.%m.%y %H:%M:%S')) - \
                                                        time.mktime(time.strptime(acquisition_info['End date time'],
                                                                                  '%d.%m.%y %H:%M:%S'))

                self.deltatime_initial_and_injection = time.mktime(time.strptime(
                    acquisition_info['Injection date time'], '%d.%m.%y %H:%M:%S')) - \
                                                       time.mktime(time.strptime(acquisition_info['Start date time'],
                                                                                 '%d.%m.%y %H:%M:%S'))

    def list_mode_decay_correction(self):

        if self.correct_decay:
            # self.decay_factor = np.exp(-(np.log(2)/self.decay_half_life)*((self.listMode[:,6]+self.delay_time)))
            time_array = (self.listMode[:, 6]-self.listMode[0, 6])+self.deltatime_injection_and_sbegin
            # print(time_array)
            time_array[time_array == 0] = np.min(time_array[np.nonzero(time_array)])
            factor = np.exp(-(np.log(2) * time_array / self.decay_half_life))
            # factor[factor < 1E-7] = np.max(factor[np.nonzero(factor[factor < 1E-7])])
            self.decay_factor = 1 / factor
            # print(np.sum(self.decay_factor))
        else:
            self.decay_factor = np.ones((len(self.listMode[:, 6])))

    def activity_on_subject_at_scanning_time(self):
        initial_activity_at_injection_time = self.initial_activity_at_injection_time()
        residual_activity_at_injection_time = self.residual_activity_at_injection_time()
        activity_on_subject_at_injection_time = self.activity_on_subject_at_injection_time()

        self._activity_on_subject_at_scanning_time = DecayCorrection.apply_decay(
            self._activity_on_subject_at_injection_time, self.deltatime_injection_and_sbegin, self.decay_half_life)
        print("initial_activity: {}".format(int(self._initial_activity)))
        print("residual_activity: {}".format(int(self._residual_activity)))
        print("initial_activity_at_injection_time: {}".format(int(initial_activity_at_injection_time)))
        print("residual_activity_at_injection_time: {}".format(int(residual_activity_at_injection_time)))
        print("activity_on_subject_at_injection_time: {}".format(int(activity_on_subject_at_injection_time)))
        print("activity_on_subject_at_scanning_time: {}".format(int(self._activity_on_subject_at_scanning_time)))
        return self._activity_on_subject_at_scanning_time

    def initial_activity_at_injection_time(self):
        self._initial_activity_at_injection_time = DecayCorrection.apply_decay(self._initial_activity,
                                                                               self.deltatime_initial_and_injection,
                                                                               self.decay_half_life)
        return self._initial_activity_at_injection_time

    def residual_activity_at_injection_time(self):
        self._residual_activity_at_injection_time = DecayCorrection.\
            _calculate_initial_activity(self._residual_activity, self.deltatime_injection_and_residual,
                                        self.decay_half_life)
        return self._residual_activity_at_injection_time

    def activity_on_subject_at_injection_time(self):
        self._activity_on_subject_at_injection_time = self._initial_activity_at_injection_time - \
                                                      self._residual_activity_at_injection_time
        return self._activity_on_subject_at_injection_time

    @staticmethod
    def apply_decay(A0, delta_t, half_life):
        return A0 * np.exp(-delta_t * np.log(2) / half_life)

    @staticmethod
    def _calculate_initial_activity(A, delta_t, half_life):
        return A / (np.exp(-delta_t * np.log(2) / half_life))

    @staticmethod
    def average_activity(A0, delta_t, half_life):
        ac_ave = (A0/np.log(2))*(half_life/delta_t)*(1-np.exp(-np.log(2)*delta_t/half_life))
        return ac_ave

    @staticmethod
    def calculate_delta_t(A0, A, half_life):
        return -half_life/np.log(2)*np.log(A/A0)
    # def initial_activity(self, value):
    #     if value != self._initial_activity:
    #         self._initial_activity = value
    #     return self._initial_activity


if __name__ == "__main__":
    import os
    import tkinter as tk
    from tkinter import filedialog

    from EasyPETLinkInitializer.EasyPETDataReader import binary_data

    root = tk.Tk()
    root.withdraw()
    # matplotlib.rcParams['font.family'] = "Gill Sans MT"
    file_path = filedialog.askopenfilename()
    easypet_folder = os.path.dirname(os.path.dirname(file_path))
    easypet_file = os.path.join(easypet_folder, "{}.easypet".format(os.path.basename(easypet_folder)))
    # file_folder = path.join(file_folder, "static_image")

    [listMode_, Version_binary, header, dates, otherinfo, acquisitionInfo, stringdata,
     systemConfigurations_info, energyfactor_info, peakMatrix_info] = binary_data().open(file_path)

    d = DecayCorrection(listMode_, acquisition_info=acquisitionInfo)
    d.activity_on_subject_at_scanning_time()
    print(time.strptime(acquisitionInfo['Acquisition start time'], '%d %b %Y - %Hh %Mm %Ss'))
