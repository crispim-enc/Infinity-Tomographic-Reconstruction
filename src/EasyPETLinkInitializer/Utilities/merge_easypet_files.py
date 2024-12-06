import shutil
import time

from array import array
import json
import os
import numpy as np
from src.EasyPETLinkInitializer.EasyPETDataReader import binary_data


class MergeEasyPETfiles:
    def __init__(self, files, new_directory=True):
        """

        """
        self.files = files
        self.init_time = None
        self.listMode_main = None
        self.version_binary_main = None
        self.header_main = None
        self.dates_main = None
        self.otherinfo_main = None
        self.acquisitionInfo_main = None
        self.stringdata_main = None
        self.system_configurations_main = None
        self.energy_factor_info_main = None
        self.peakMatrix_info_main = None
        self.start_time_main_scan = None
        self.output_file_name = None
        if new_directory:
            self._create_directory()
        else:
            self.output_file_name = self.files[0]

    def _create_directory(self):
        # name_file = os.path.basename(os.path.dirname((self.files[0])))º
        root_src_dir = os.path.dirname(self.files[0])
        root_dst_dir = "{} converted".format(os.path.dirname(self.files[0]))
        for src_dir, dirs, files in os.walk(root_src_dir):
            dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            for file_ in files:
                src_file = os.path.join(src_dir, file_)
                _, ext = os.path.splitext(src_file)
                if ext == '.easypetoriginal':
                    dst_newname_file = "{}converted Original data".format(os.path.basename(_).split("Original data")[0])
                    dst_file = os.path.join(dst_dir, "{}{}".format(dst_newname_file, ext))
                else:
                    dst_file = os.path.join(dst_dir, file_)
                if os.path.exists(dst_file):
                    os.remove(dst_file)
                shutil.copy(src_file, dst_file)
        # a =shutil.copytree(os.path.dirname((self.files[0]))," {} converted".format(os.path.dirname((self.files[0]))),dirs_exist_ok=True)
        # os.path.dirname((self.files[0]))
        self.output_file_name = os.path.join("{} converted".format(os.path.dirname((self.files[0]))), os.path.basename(self.files[0]))

    def _read_main_file(self):
        [self.listMode_main, self.version_binary_main, self.header_main, self.dates_main, self.otherinfo_main,
         self.acquisitionInfo_main, self.stringdata_main, self.system_configurations_main,
         self.energyfactor_info_main, self.peakMatrix_info_main] = binary_data().open(self.files[0])
        self.start_time_main_scan = time.mktime(
                    time.strptime(self.acquisitionInfo_main['Acquisition start time'], '%d %b %Y - %Hh %Mm %Ss'))

    def merge_list_mode(self):
        self._read_main_file()
        c = 1
        for file in self.files[1:]:
            [listmode, Version_binary, header, dates, otherinfo, acquisitionInfo, stringdata,
             systemConfigurations_info,
             energyfactor_info, peakMatrix_info] = binary_data().open(file)
            start_time_scan = time.mktime(
                    time.strptime(acquisitionInfo['Acquisition start time'], '%d %b %Y - %Hh %Mm %Ss'))

            diff_time = start_time_scan - self.start_time_main_scan

            listmode[:, 6] += diff_time
            if c == 1:
                update_index = [i+len(self.listMode_main) for i in acquisitionInfo["Turn end index"]]
            else:
                update_index = [i + len(listmode) for i in acquisitionInfo["Turn end index"]]
            self.acquisitionInfo_main["Turn end index"].extend(update_index)
            self.listMode_main = np.vstack([self.listMode_main, listmode])

    def write_easypet_format(self):
        step_bot = round(self.header_main[1], 4) / self.header_main[2]
        step_top = round(self.header_main[3], 4) / self.header_main[4]
        topRange = self.header_main[5]

        self.listMode_main[:, 5] = (self.listMode_main[:, 5] + topRange/2) / step_top
        self.listMode_main[:, 4] = self.listMode_main[:, 4] / step_bot

        crystalMatrix = np.array(self.listMode_main[:, 2:4], dtype='int16', copy=False)
        data4save = np.round(self.listMode_main[:, 4:6], 4).astype(dtype='int16', copy=False)
        EA_Corrected = self.listMode_main[:, 0].astype(dtype='int16', copy=False)
        EB_Corrected = self.listMode_main[:, 1].astype(dtype='int16', copy=False)
        timestamp = self.listMode_main[:, 6].astype(dtype='float32', copy=False)

        records = np.rec.fromarrays(
            (data4save[:, 0], data4save[:, 1], crystalMatrix[:, 0], crystalMatrix[:, 1],
             EA_Corrected, EB_Corrected, timestamp),
            names=('bot', 'top', 'id1', 'id2', 'EA', 'EB', 'timestamp'))

        header = array('f', self.header_main)
        dates = array('u', self.dates_main)
        Version_binary = array('u',  self.version_binary_main)

        acquisitionParameters = json.dumps(self.acquisitionInfo_main)
        acquisition_info = array('u', acquisitionParameters)
        acquisition_info_size = [len(acquisitionParameters)]
        acquisition_info_size = array('i', acquisition_info_size)

        stringdata = json.dumps(self.stringdata_main)
        stringdata = array('u', stringdata)
        stringdata_size = [len(stringdata)]
        stringdata_size = array('i', stringdata_size)

        systemConfigurations = json.dumps(self.system_configurations_main)
        systemConfigurations_info = array('u', systemConfigurations)
        systemConfigurations_info_size = [len(systemConfigurations_info)]
        systemConfigurations_info_size = array('i', systemConfigurations_info_size)

        # [peakMatrix, calibration_file, energyfactor] = calibration_points_init(
        #     [int(reading_hardware_parameters.array_crystal_x), int(reading_hardware_parameters.array_crystal_y)])

        energyfactor = self.energyfactor_info_main
        # energyfactor = energyfactor[0].tolist()
        # energyfactor_str = str(energyfactor).strip('[]')
        # energyfactor_str = energyfactor_str.replace(" ", "")
        energyfactor_size = array('i', [len(energyfactor)])
        energyfactor_info = array('u', energyfactor)

        peakMatrix_str = self.peakMatrix_info_main
        # peakMatrix = peakMatrix.flatten('F')
        # peakMatrix_str = str(peakMatrix).strip('[]')

        peakMatrix_size = array('i', [len(peakMatrix_str)])
        peakMatrix_info = array('u', peakMatrix_str)

        size_header = [2 * 9 + 4 + 24 + 102 + 4 + acquisition_info_size[0] * 2 + 4 + stringdata_size[0] * 2 +
                       4 + systemConfigurations_info_size[0] * 2 + 4 + energyfactor_size[0] * 2 +
                       4 + peakMatrix_size[0] * 2]
        size_header = array('i', size_header)

        with open(self.output_file_name, 'wb') as output_file:
            Version_binary.tofile(output_file)
            size_header.tofile(output_file)
            header.tofile(output_file)
            dates.tofile(output_file)
            acquisition_info_size.tofile(output_file)
            stringdata_size.tofile(output_file)
            systemConfigurations_info_size.tofile(output_file)
            energyfactor_size.tofile(output_file)
            peakMatrix_size.tofile(output_file)
            acquisition_info.tofile(output_file)
            stringdata.tofile(output_file)
            systemConfigurations_info.tofile(output_file)
            energyfactor_info.tofile(output_file)
            peakMatrix_info.tofile(output_file)
            records.tofile(output_file)

        output_file.close()




if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()

    list_of_directories = []

    file_path_1 = filedialog.askopenfilename()
    file_path_2 = filedialog.askopenfilename()
    list_of_directories.append(file_path_1)
    list_of_directories.append(file_path_2)

    merge = MergeEasyPETfiles(list_of_directories)
    merge.merge_list_mode()
    merge.write_easypet_format()


