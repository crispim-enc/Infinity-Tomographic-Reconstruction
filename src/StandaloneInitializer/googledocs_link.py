import os
import sys
import time
import numpy as np
import pandas as pd
from pandas import ExcelFile
from src.StandaloneInitializer import ReconstructOpenFileTest


class SelectReconstructionConditionsFromExcel:
    def __init__(self, excel_file, main_directory=None, files_in_main_location=False):
        self.excel_file = excel_file
        self.main_directory = main_directory
        xls = ExcelFile(self.excel_file)
        self.dataframe = xls.parse(xls.sheet_names[0])
        self.files_in_main_location = files_in_main_location

    def find_paths(self):
        # file_paths = [[None ,None]] * len(self.dataframe.Nome_do_Ficheiro)
        file_paths = [[None, None, None] for _ in range( len(self.dataframe.Nome_do_Ficheiro)+1)]
        file_paths[0][0] = "FILENAME"
        file_paths[0][1] = "FILE FOUND"
        file_paths[0][2] = "DIRECTORY FOUND"
        for el in range(1,len(self.dataframe.Nome_do_Ficheiro)+1):
            name = self.dataframe.Nome_do_Ficheiro[el-1]

            file_name = os.path.join("{}.easypet".format(name))  # alterar
            filepath, directory = self.find_path(name)
            file_paths[el][0] = file_name
            if filepath is None:
                file_paths[el][1] = "NO"
            else:
                file_paths[el][1] = "YES"
            file_paths[el][2] = directory
            # for dirpath, dirnames, filenames in os.walk(self.main_directory):
            #     for filename in filenames:
            #         if filename == file_name:
            #             filename = os.path.join(dirpath, filename)
            #             file_paths[el][1] = filename
            #             print(filename)
            #             break

                        # print(dirpath)
        dataframe_files =  pd.DataFrame(file_paths)
        dataframe_files.to_excel(os.path.join(os.path.dirname(self.excel_file),
                              'Files_found.xlsx'), sheet_name='files_found')
        print(dataframe_files)

    def find_path(self, name):
        filename = None
        # sub_directory = None
        file_name = os.path.join("{}.easypet".format(name))  # alterar

        print(file_name)
        sub_directories = list()
        for dirpath, dirnames, files in os.walk(self.main_directory):
            # print(dirpath)
            for dirname in dirnames:
                if dirname == name:

                    sub = os.path.join(dirpath, dirname)
                    sub_directories.append(sub)
                    print("Found Directory: {}".format(sub))
                    break
        if len(sub_directories) > 0:
            for sub_directory in sub_directories:
                for dirpath, dirnames, files in os.walk(sub_directory):
                    for f in files:
                        if f == file_name:
                            filename = os.path.join(dirpath, f)
                            print("File Found: {}\n ----------------".format(filename))
                            break
        else:
            print("Directory not Found")
        if filename is None:
            print("File Not Found \n -------------" )
        return filename, sub_directories

    def start(self):
        for el in range(len(self.dataframe.Nome_do_Ficheiro)):
            name = self.dataframe.Nome_do_Ficheiro[el]

            if self.dataframe.Reconstrucao[el] != "FEITO":
                # if not self.files_in_main_location:
                #     if not pd.isna(self.dataframe.Folder[el]):
                #         folder =  "{}_{}".format("-".join(self.dataframe.Data[el]._date_repr.split("-")[0:2]), self.dataframe.Folder[el])
                #     else:
                #         folder = "-".join(self.dataframe.Data[el]._date_repr.split("-")[0:2])
                if not pd.isna(name):
                    if not self.files_in_main_location:
                        # file_name = os.path.join(self.main_directory, folder, name, "{}.easypet".format(name))

                        file_name, subdirectories = self.find_path(name)# alterar
                    else:
                        file_name = os.path.join(self.main_directory, name, "{}.easypet".format(name))
                    if pd.isna(self.dataframe.Inicio_Intervalo_temporal[el]) or pd.isna(self.dataframe.Fim_Intervalo_temporal[el]):
                        remove_turns = None
                        name_multiple_conditions = "Full"
                    else:
                        if pd.isna(self.dataframe.Cut_per_time[el]) or (self.dataframe.Cut_per_time[el]==True):
                            cut_pert_time = True
                            init_time = self.dataframe.Inicio_Intervalo_temporal[el] * 60
                            end_time = self.dataframe.Fim_Intervalo_temporal[el] * 60
                        else:
                            cut_pert_time = self.dataframe.Cut_per_time[el]
                            init_time = self.dataframe.Inicio_Intervalo_temporal[el]
                            end_time = self.dataframe.Fim_Intervalo_temporal[el]
                        remove_turns = {
                            "Cut_per_time": cut_pert_time,
                            "Init time": init_time,
                            "End time": end_time,
                            "Whole body": True,
                            "Dynamic": False,
                            "Static": False,
                            "Gated": False}
                        name_multiple_conditions = "{}s_{}s".format(remove_turns["Init time"], remove_turns["End time"])

                    algorithm = self.dataframe.Algorithm[el].upper()
                    projector = str(self.dataframe.Projector[el])
                    energy_window = [self.dataframe.Energy_window_low[el], self.dataframe.Energy_window_high[el]]
                    calculate_quantification_factors = self.dataframe.Calculate_quantification_factors[el]
                    if self.dataframe.Algorithm[el] == "LM-MRP":
                        algorithm_options = [float(self.dataframe.Algorithm_beta), int(self.dataframe.algorithm_kernel_size)]

                    else:
                        algorithm_options = None
                    if self.dataframe.Algorithm[el] == "FBP":
                        number_of_iterations = None
                    else:
                        number_of_iterations = int(self.dataframe.Number_of_iterations[el])


                    type_of_reconstruction = [self.dataframe.Whole_Body[el], False, self.dataframe.Dynamic[el], False]
                    voxel_size = self.dataframe.Pixel_size[el]
                    number_cumulative_turns = self.dataframe.Cumulative_turns[el]
                    if pd.isna(self.dataframe.Cumulative_turns[el]) and self.dataframe.Dynamic[el]==True:
                        number_cumulative_turns = 1
                    elif pd.isna(self.dataframe.Cumulative_turns[el]) and self.dataframe.Dynamic[el]==False:
                        number_cumulative_turns = None
                    # name_multiple_conditions = "{}s_{}s".format(remove_turns["Init time"], remove_turns["End time"])
                    # self.dataframe.Reconstrucao[el] =
                    if pd.isna(self.dataframe.coincidence_window[el]):
                        coincidence_window = None

                    else:
                        coincidence_window = self.dataframe.coincidence_window[el]



                    if file_name is not None:
                        try:
                            if pd.isna((self.dataframe.Path_to_save[el])):
                                path_to_save = "E:\\OneDrive - Universidade de Aveiro\\Desktop"

                            else:
                                path_to_save = self.dataframe.Path_to_save[el]

                            if not os.path.exists(path_to_save):
                                os.makedirs(path_to_save)


                            folder_to_keep_studies = os.path.join(path_to_save,"studies" , os.path.basename(os.path.dirname(os.path.dirname(file_name))))
                            if not os.path.exists(folder_to_keep_studies):
                                os.makedirs(folder_to_keep_studies)

                            if algorithm_options is not None:
                                str_algorithm_options = [str(option) for option in algorithm_options]
                                str_algorithm_options = "b{}k{}".format(str_algorithm_options[0], str_algorithm_options[1])
                                str_algorithm_options = str_algorithm_options.replace(".","_")
                            else:
                                str_algorithm_options = algorithm_options
                            main_folder_name = os.path.join(folder_to_keep_studies,"{}_{}_{}".format(self.dataframe.Algorithm[el],
                                                                                                        str_algorithm_options, projector))
                            main_folder_name = main_folder_name.replace("[","__")
                            if not os.path.exists(main_folder_name):
                                os.makedirs(main_folder_name)

                            subfolder = os.path.join(main_folder_name,"{}mm{}keV{}ps".format(voxel_size,energy_window,coincidence_window))
                            subfolder = subfolder.replace(".", "_")
                            subfolder = subfolder.replace("[","__")
                            subfolder = subfolder.replace("]","__")
                            subfolder = subfolder.replace(","," ")

                            if not os.path.exists(subfolder):
                                os.makedirs(subfolder)
                            # else:
                            #     os.makedirs(os.path.join())

                            folder_console = os.path.join(subfolder, "console_output")
                            if not os.path.exists(folder_console):
                                os.makedirs(folder_console)

                            logger_file_name = os.path.join(folder_console,
                                                   f'console{time.strftime("%d %b %Y %H_%M_%S", time.gmtime())}.txt')

                            sys.stdout = Logger(logger_file_name)

                            tic = time.time()

                            ReconstructOpenFileTest(list_open_studies=file_name,
                                                    multiple_conditions=[True, name_multiple_conditions],
                                                    remove_turns=remove_turns,
                                                    type_of_reconstruction=type_of_reconstruction,
                                                    voxel_size=voxel_size,
                                                    number_cumulative_turn=number_cumulative_turns,
                                                    algorithm=algorithm, algorithm_options=algorithm_options,
                                                    projector_type=projector, energy_window=energy_window,
                                                    number_of_iterations=number_of_iterations,
                                                    calculate_quantification_factors=calculate_quantification_factors,
                                                    coincidence_window=coincidence_window,
                                                    override_path=subfolder)
                            # sys.stdout.flush()
                            # sys.stdout.close()
                            self.dataframe.loc[el, 'Reconstrucao'] = "FEITO"
                        except ValueError as e:
                            print(e)
                            self.dataframe.loc[el, 'Reconstrucao'] = e

                        except FileNotFoundError as e:
                            print("Ficheiro não encontrado:{}".format(e))
                            self.dataframe.loc[el, 'Reconstrucao'] = "Ficheiro não encontrado:{}".format(e)
                    else:
                        self.dataframe.loc[el, 'Reconstrucao'] = "Ficheiro não encontrado:{}".format(file_name)

                    self.dataframe.to_excel(os.path.join(os.path.dirname(self.excel_file),
                                            'Reconstruções_em_falta.xlsx'), sheet_name='reconstruções', index=False)
                    time.sleep(5)


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        try:
            self.log.write(message)
        except ValueError as e:
            print(e)
        self.terminal.flush()

    def flush(self):
        pass

    def close(self):
        self.log.close()

    # def flush(self):
    #     self.log.close()
    #     pass




    # def flush(self):
    #     # this flush method is needed for python 3 compatibility.
    #     # this handles the flush command by doing nothing.
    #     # you might want to specify some extra behavior here.
    #     pass




if __name__ == "__main__":
    file_name_excel = "C:\\Users\\pedro.encarnacao\\OneDrive - Universidade de Aveiro\\ICNAS\\Reconstruções_em_falta.xlsx"
    file_name_excel = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\ICNAS\\Reconstruções_em_falta.xlsx"
    # file_name_excel = "C:\\Users\\pedro.encarnacao\\Universidade de Aveiro\\Fabiana Ribeiro - iCBR\\Reconstruções_em_falta.xlsx"
    m_dir = "C:\\Users\\pedro.encarnacao\\OneDrive - Universidade de Aveiro\\ICNAS\\Acquisitions 2022"
    m_dir = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\ICNAS\\Acquisitions 2022"


    # file_name_excel = "/home/crispim/Documentos/Simulations/Simulations/Easypet/Reconstruções_em_falta.xlsx"
    # m_dir = "/home/crispim/Documentos/Simulations/"

    file_name_excel = "E:\\OneDrive - Universidade de Aveiro\\Desktop\\Reconstruções_em_falta.xlsx"
    m_dir = "E:\\OneDrive - Universidade de Aveiro\\SimulacoesGATE\\EasyPET3D64\\"
    # m_dir = "C:\\Users\\pedro.encarnacao\\Universidade de Aveiro\\Fabiana Ribeiro - iCBR\\iCBR_Acquisitions\\2022\\"
    s = SelectReconstructionConditionsFromExcel(file_name_excel, main_directory=m_dir)
    # s.find_paths()
    s.start()

    link = "https://docs.google.com/spreadsheets/d/1XoZ7pbKGBFrAfdCEXItbD3Q1mXV5JaL6EX1rNsj9qhw/edit?usp=sharing"