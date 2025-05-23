import os.path
import tkinter as tk
import time
import json
import numpy as np
import pandas as pd
from pandas import ExcelFile
from array import array
from EasyPETLinkInitializer.EasyPETDataReader import binary_data
from EasyPETLinkInitializer.EasyPETDataReader.calibration_points_upload import calibration_points_init
from EasyPETLinkInitializer.EasyPETDataReader import time_discrimination


class RewriteHeader:
    def __init__(self, path_study=None, excel_file=None, rewrite_acquisitionInfo=True,
                 rewrite_system_configuration=True, crystal_geometry=None):
        if crystal_geometry is None:
            crystal_geometry = [32, 2]
        self.excel_file = excel_file
        self.main_folder = os.path.dirname(excel_file)
        self.rewrite_acquisitionInfo = rewrite_acquisitionInfo
        self.rewrite_system_configuration = rewrite_system_configuration
        self.crystal_geometry = crystal_geometry
        xls = ExcelFile(self.excel_file)
        self.dataframe = xls.parse(xls.sheet_names[0])
        self.indexes = None
        self.files_paths = None

    def get_file_name(self):
        print(self.dataframe.columns.values)
        # number_of_files = len(self.dataframe["Date "].values[self.dataframe["Date "].values != None])
        number_of_files = self.dataframe.Recuperado[self.dataframe.Recuperado.values != True].shape[0]
        indexes = np.where(self.dataframe.Recuperado.values != True)[0]
        indexes = indexes[pd.isna(self.dataframe["Date "][self.dataframe.Recuperado.values != True])==False]
        number_of_files = len(indexes)
        self.indexes = indexes
        self.files_paths = [None]*number_of_files
        # for id_file in range(len(self.files_paths)):
        id_ = 0
        for id_file in indexes:
            print(id_file)

            # if self.dataframe.Recuperado[id_file] != True:
            date = str(self.dataframe["Date "][id_file]).split(" ")[0]

            scan_time = self.dataframe["Scan Time"][id_file]

            date_and_scan_time = time.strptime("{} {}".format(date, scan_time), '%Y-%m-%d %H:%M:%S')
            folder_name = time.strftime('Easypet Scan %d %b %Y - %Hh %Mm %Ss', date_and_scan_time)
            file_name = "{}.easypet".format(folder_name)
            year_folder = str(date_and_scan_time.tm_year)
            year_month_folder = time.strftime("%Y-%m",date_and_scan_time)
            # self.files_paths[id_] = os.path.join(main_folder, "iCBR_Acquisitions",  year_folder, year_month_folder,
            #                          folder_name, file_name)
            self.files_paths[id_] = os.path.join(main_folder,
                                                 folder_name, file_name)

            print(self.files_paths[id_])
            id_ += 1

    def recreate_acquisitionInfo(self, id=0):
        acquisitionInfo = {}
        start_column = int(np.where(self.dataframe.columns.values == "Type of subject")[0])
        for name in self.dataframe.columns.values[start_column:]:
            acquisitionInfo[name] = "{}".format(self.dataframe[name][id])
            if name == "Start date time" or name == "End date time" or name == "Injection date time":
                date = str(self.dataframe["Date "][id]).split(" ")[0]
                start = self.dataframe[name][id]
                datetime_start = time.strptime("{} {}".format(date, start), '%Y-%m-%d %H:%M:%S')
                acquisitionInfo[name] = time.strftime('%d.%m.%y %H:%M:%S', datetime_start)
            elif name == "Acquisition start time":
                date = str(self.dataframe["Date "][id]).split(" ")[0]
                scan_time = self.dataframe["Scan Time"][id]
                date_and_scan_time = time.strptime("{} {}".format(date, scan_time), '%Y-%m-%d %H:%M:%S')
                acquisitionInfo[name] = time.strftime('%d %b %Y - %Hh %Mm %Ss', date_and_scan_time)
        return acquisitionInfo

    def recreate_system_configurations(self, systemConfigurations_info=None):
        systemConfigurations_info['array_crystal_x'] = self.crystal_geometry[0]
        systemConfigurations_info['array_crystal_y'] = self.crystal_geometry[1]
        return systemConfigurations_info

    def rewrite_file(self):
        i = 0
        for file_path in self.files_paths:
            print(file_path)
            filename = os.path.splitext(os.path.basename(file_path))
            filename = "{} teste{}".format(filename[0], filename[1])
            file_to_record = os.path.join(os.path.dirname(file_path), filename)

            try:
                [listmode, Version_binary, header, dates, otherinfo, acquisitionInfo, stringdata,
                 systemConfigurations_info,
                 energyfactor_info, peakMatrix_info] = binary_data().open(file_path)

                systemConfigurations_info = self.recreate_system_configurations(systemConfigurations_info)
                acquisitionInfo = self.recreate_acquisitionInfo(self.indexes[i])
                time_indexes = time_discrimination(listmode)
                acquisitionInfo["Turn end index"] = time_indexes.tolist()
                step_bot = round(header[1], 4) / header[2]
                step_top = round(header[3], 4) / header[4]
                topRange = header[5]

                listmode[:, 5] = (listmode[:, 5] + topRange/2) / step_top
                listmode[:, 4] = listmode[:, 4] / step_bot

                crystalMatrix = np.array(listmode[:, 2:4], dtype='int16', copy=False)
                data4save = np.round(listmode[:, 4:6], 4).astype(dtype='int16', copy=False)
                EA_Corrected = listmode[:, 0].astype(dtype='int16', copy=False)
                EB_Corrected = listmode[:, 1].astype(dtype='int16', copy=False)
                timestamp = listmode[:, 6].astype(dtype='float32', copy=False)

                records = np.rec.fromarrays(
                    (data4save[:, 0], data4save[:, 1], crystalMatrix[:, 0], crystalMatrix[:, 1],
                     EA_Corrected, EB_Corrected, timestamp),
                    names=('bot', 'top', 'id1', 'id2', 'EA', 'EB', 'timestamp'))

                print(header)

                header = array('f', header)
                dates = array('u', dates)

                Version_binary = 'Version 2'
                Version_binary = array('u', Version_binary)
                acquisitionParameters = json.dumps(acquisitionInfo)
                acquisition_info = array('u', acquisitionParameters)
                acquisition_info_size = [len(acquisitionParameters)]
                acquisition_info_size = array('i', acquisition_info_size)

                stringdata = json.dumps(stringdata)
                stringdata = array('u', stringdata)
                stringdata_size = [len(stringdata)]
                stringdata_size = array('i', stringdata_size)

                systemConfigurations = json.dumps(systemConfigurations_info)
                systemConfigurations_info = array('u', systemConfigurations)
                systemConfigurations_info_size = [len(systemConfigurations_info)]
                systemConfigurations_info_size = array('i', systemConfigurations_info_size)

                [peakMatrix, calibration_file, energyfactor] = calibration_points_init(self.crystal_geometry)

                # energyfactor = self.dummy_energyfactor_info


                energyfactor = energyfactor[0].tolist()
                energyfactor_str = str(energyfactor).strip('[]')
                energyfactor_str = energyfactor_str.replace(" ", "")
                energyfactor_size = array('i', [len(energyfactor_str)])
                energyfactor_info = array('u', energyfactor_str)
                # peakMatrix_str = self.dummy_peakMatrix_info
                peakMatrix = peakMatrix.flatten('F')
                peakMatrix_str = str(peakMatrix).strip('[]')

                peakMatrix_size = array('i', [len(peakMatrix_str)])
                peakMatrix_info = array('u', peakMatrix_str)

                size_header = [2 * 9 + 4 + 24 + 102 + 4 + acquisition_info_size[0] * 2 + 4 + stringdata_size[0] * 2 +
                               4 + systemConfigurations_info_size[0] * 2 + 4 + energyfactor_size[0] * 2 +
                               4 + peakMatrix_size[0] * 2]
                size_header = array('i', size_header)

                with open(file_to_record, 'wb') as output_file:
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
                self.dataframe.loc[self.indexes[i], 'Recuperado'] = True

            except FileNotFoundError:
                print("File Not Found: {}".format(file_path))
                self.dataframe.loc[self.indexes[i], 'Recuperado'] = False

            except TypeError as e:
                print("TypeError: {}".format(file_path))
                print(e)
                self.dataframe.loc[self.indexes[i], 'Recuperado'] = False

            except ValueError:
                print("ValueError: {}".format(file_path))
                self.dataframe.loc[self.indexes[i], 'Recuperado'] = False
        #
        # except ValueError as e:
        # print(e)

            i += 1
        self.dataframe.to_excel(os.path.join(os.path.dirname(self.excel_file),
                                             'Recuperado.xlsx'), sheet_name='new_sheet_name')





if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    # excel_file = "C:\\Users\\pedro\\Universidade de Aveiro\\Fabiana Ribeiro - iCBR\\iCBR-2022.xlsx"
    excel_file = "C:\\Users\\pedro.encarnacao\\Universidade de Aveiro\\Fabiana Ribeiro - iCBR\\iCBR-2022.xlsx"
    excel_file = "C:\\Users\\pedro.encarnacao\\OneDrive - Universidade de Aveiro\\ICNAS\\Acquisitions 2022\\ICNAS_recoverFile.xlsx"
    # excel_file = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\PhD\\iCBR-2022.xlsx"

    main_folder = os.path.dirname(excel_file)
    rh = RewriteHeader(excel_file=excel_file)
    rh.get_file_name()
    rh.recreate_acquisitionInfo()
    rh.rewrite_file()





    # file_path = filedialog.askopenfilename()
    # ValidationModule(file_path)