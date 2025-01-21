import os
import tkinter as tk
from array import array
from tkinter import filedialog
import re
import matplotlib.pyplot as plt
import numpy as np
import json
from src.Phantoms import SuvReader
from src.ImageReader import RawDataSetter


class MobyPhantomMapGenerator:
    def __init__(self, file=None, total_activity=200, unit="uCi", radiotracer="F18",
                 attenuation_discrimination_level="High"):
        """ A"""
        if file is None:
            return
        self.directory = os.path.dirname(file)
        self.file = file
        self._original_volume = None
        self.total_activity = total_activity

        if unit == "uCi":
            self.total_activity *= 37000
        self._image_size = [256, 256, 750]
        # self._voxel_size = [0.145, 0.145, 0.145]  # mm
        self._voxel_size = [0.125, 0.125, 0.125]  # mm

        self._voxel_volume = self._voxel_size[0] * self._voxel_size[1] * self._voxel_size[2]*0.001
        self.total_density = 1
        self.weight_total_mice = None
        self.volume_total_mice = None
        self._radiotracer = radiotracer
        self.log_file_name = "Moby_average_log"
        self.log_file = None
        self.attenuation_discrimination_level = "High"
        self.initial_data = {}
        self.heart_orientation = {}
        self.heart_translation = {}
        self.linear_attenuation_coeff_l_cm = {}
        self.linear_attenuation_coeff_l_pixel = {}
        self.activity_ratios = {}
        self.phantom_dimensions = {}
        self.volumes_organs = {}
        self.volumes_organs_cal = {}
        self.organs_weights = {}
        self._s_values = None #  np.array([np.arange(1, 75), np.random.randint(0, 100, 74)]).T
        self.activity_map_generated = None
        self.activity_table = np.round(np.array([np.arange(0, 79)-0.001, np.arange(0, 79)+0.001, np.zeros((79))]).T,3)
        self.activity_table[0, 0:3] = [0, 0, 0]
        self.attenuation_table = np.round(np.array([np.arange(0, 79)-0.001, np.arange(0, 79)+0.001, np.zeros(79), np.zeros(79), np.zeros(79), np.zeros(79), np.zeros(79), np.zeros(79)]).T,3).astype(object)
        self.attenuation_table[:,3] = "false"
        self.attenuation_table[0, 0:3] = [0, 0,"Air"]
        self._densitiesFile = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "dataFiles", "GateMaterials.db")

        self._mapMobyFile = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                           "dataFiles", "Moby settings", "mapMobyToGate.dat")

        self._mapMobyToGate = None
        self._numberOfVoxelsPerOrgan = {}

    def original_volume(self):
        return self._original_volume

    def load_original_volume(self):
        """ """
        output_file = open(self.file, 'rb')  # define o ficheiro que queres ler
        a = array('f')  # define quantos bytes le de cada vez (float32)
        size_file = self._image_size[0] * self._image_size[1] * self._image_size[2]
        a.fromfile(output_file, size_file)  # lê o ficheiro binário (fread)
        output_file.close()  # fecha o ficheiro
        volume = np.array(a)  # não precisas
        self._original_volume = volume.reshape((self._image_size[0], self._image_size[1], self._image_size[2]),
                                               order='f')  # transforma em volume
        # self._determine_linear_array()

    def update_image_size(self, image_size=None):
        if image_size is None:
            print("""Input ERROR: image_size = [n_x_pixels, n_y_pixels, n_z_pixels """)
            return
        self._image_size = image_size

    def load_log_file(self):
        print("________LOADING LOG FILE FORM MOBY________")
        active_dict = self.initial_data
        frame = None
        with open(os.path.join(self.directory, self.log_file_name), "r") as log:
            # data = log.read()
            for line in log:

                line = re.sub(' +', ' ', line)
                par = line.split("\n")
                try:
                    args = par[0].split("=")

                    if len(args) == 1:  # (val, info) = val.split(" ")
                        if args[0] == "Main orientation of heart:":
                            # active_dict = active_dict_list[1]
                            active_dict = self.heart_orientation
                            frame = None

                        elif args[0] == "Translation of heart:":
                            # active_dict = active_dict_list[2]
                            active_dict = self.heart_translation
                            frame = None

                        elif args[0] == "Linear Attenuation Coefficients (1/cm):":
                            # active_dict = active_dict_list[3]
                            active_dict = self.linear_attenuation_coeff_l_cm
                            frame = None

                        elif args[0] == "Linear Attenuation Coefficients (1/pixel):":
                            # active_dict = active_dict_list[4]
                            active_dict = self.linear_attenuation_coeff_l_pixel
                            frame = None

                        elif args[0] == "Activity Ratios":
                            # active_dict = active_dict_list[4]
                            active_dict = self.activity_ratios
                            frame = None

                        elif args[0] == "-----------------Phantom Dimensions------------------":
                            # active_dict = active_dict_list[4]
                            active_dict = self.phantom_dimensions
                            frame = None
                        elif args[0] == "-----------------------------------------------------------------":
                            pass
                        elif args[0] == "---------------------------------------- ":
                            pass
                        elif args[0] == "":
                            pass

                        elif args[0].startswith("CREATING FRAME"):
                            (name, value) = args[0].split("#")
                            frame = value.split(" ")[0]
                            active_dict = self.volumes_organs
                        elif args[0] == "ORGAN VOLUMES: ":
                            pass

                        else:
                            active_dict = self.initial_data
                    if len(args) > 1:
                        MobyPhantomMapGenerator._alocate_data_to_dict(active_dict, args, frame=frame)

                except ValueError as e:
                    print(e)
                    continue
                except IndexError:
                    continue
        print("________________________________")
        # added miss elements
        self.activity_ratios["skin_activity"] = {"value": 1, "info": " "}
        self.activity_ratios["gall_bladder_activity"] = {"value": 12, "info": " "}
        self.activity_ratios["humerus_activity"] = {"value": 28, "info": " "}
        self.activity_ratios["radius_activity"] = {"value": 29, "info": " "}
        self.activity_ratios["ulna_activity"] = {"value": 30, "info": " "}
        self.activity_ratios["femur_activity"] = {"value": 31, "info": " "}
        self.activity_ratios["fibula_activity"] = {"value": 32, "info": " "}
        self.activity_ratios["tibia_activity"] = {"value": 33, "info": " "}
        self.activity_ratios["patella_activity"] = {"value": 34, "info": " "}
        self.activity_ratios["bone_activity"] = {"value": 35, "info": " "}
        self.activity_ratios["marrow_activity"] = {"value": 77, "info": " "}
        self.activity_ratios["lesn_activity"] = {"value": 78, "info": " "}


    @staticmethod
    def _alocate_data_to_dict(data, args, frame=None):
        try:
            if len(args) > 1:
                if frame is not None:
                    key = "{}_fr_{}".format(args[0], frame)
                else:
                    key = args[0].replace(" ","")
                data[key] = {"value": " ", "info": " "}
                values_info = args[1].split(" ")
                if values_info[0] == "":
                    val = values_info[1]
                    info = " ".join(values_info[2:])

                else:
                    val = values_info[0]
                    info = values_info[1]
                data[key]["value"] = float(val)
                data[key]["info"] = info

        except ValueError as e:
            print(e)
            pass
        except IndexError:
            pass

    def load_s_values(self):
        """ """
        suv_data = SuvReader(mice_excel_file_path)
        suv_data.read_file()
        self._s_values = suv_data.df_Naf

    @property
    def density(self):
        return self._density

    def setDensity(self, material=None):
        try:
            material = self._mapMobyToGate[material]
            print(material)
            with open(self._densitiesFile) as file:
                # print(self._material)
                data = file.read()
                data = data.split("\n")

                # if len(self._material) > 1:
                #     material = self._material[0]
                mask = [el.startswith(material) for el in data]

            self._density = float(np.array(data)[np.array(mask)][0].split("d=")[1].split(" ")[0])
            print("Density set to {} g /cm3".format(self._density))
        except IndexError:
            print("Material not found. Density set to 1 g /cm3")
            self._density = 1

    def readMapMobyToGate(self):
        with open(self._mapMobyFile) as file:
            data = file.read()
            data = data.split("\n")
            data = [el.split("=") for el in data]
            moby_words = [el[0].replace(" ", "") for el in data]
            gate_words = [data[el][1].split("\t")[0].replace(" ", "") for el in range(0,len(data)-1)]

        self._mapMobyToGate = dict(zip(moby_words, gate_words))

    def calculate_volume_organs(self):
        print("________CALCULATING VOLUMES________")
        total_volume = 0
        total_weight = 0
        for organ in self.activity_ratios:
            volume_temp = self._original_volume.astype(np.int32)
            num_pixels = len(volume_temp[volume_temp == int(self.activity_ratios[organ]["value"])] )
            calculated_volume = num_pixels * self._voxel_size[0] * self._voxel_size[1] * self._voxel_size[2]*0.001
            self.volumes_organs_cal[organ] = [calculated_volume]
            self.organs_weights[organ] = self.calculateOrganWeight(organ, calculated_volume)
            total_volume += calculated_volume
            total_weight += self.organs_weights[organ]
            print("{}: {}: {} ml".format(int(self.activity_ratios[organ]["value"]),organ, calculated_volume))
        print(self.organs_weights)
        print("________________________________")

        return total_volume, total_weight


    def calculateOrganWeight(self, organ, volume_organ):
        self.setDensity(organ)
        weight_organ = self.density * volume_organ
        return weight_organ

    def calculate_volume_total_phantom(self, precise=False):
        if precise:
            self.volume_total_mice, self.weight_total_mice = self.calculate_volume_organs()

        else:
            self.volume_total_mice = len(self._original_volume[self._original_volume != 0])*self._voxel_volume
            self.weight_total_mice = self.total_density * self.volume_total_mice
        print("Volume_phantom: {} ml".format(self.volume_total_mice))
        print("weight_total_mice: {} g".format(self.weight_total_mice))


    def generate_new_activity_map(self):
        """ """
        self.load_s_values()
        self.activity_map_generated = np.copy(self._original_volume)
        self.calculate_volume_total_phantom(precise=False)
        compare = self._original_volume.astype(int)
        total_activity_in_phantom = 0
        volume_org = 0
        for organ in self.activity_ratios:
            print("_________{}:{}_________".format( int(self.activity_ratios[organ]["value"]),organ))
            act_tomap = 0.26 # background organ activity
            # volume_temp = np.copy(self._original_volume)
            # activity_temp_value = self._s_values[self._s_values[:, 0] ==
            #                                      int(self.activity_ratios[organ]["value"]), 1] * \
            #                       self.total_activity / self.volumes_organs_cal[organ]

            for key in self._s_values:
                # print(self._s_values[key]["name_moby"])
                # print(self._s_values[key]["name_moby"] in organ)
                if str(organ).startswith(str(self._s_values[key]["name_moby"])):
                    act_tomap = self._s_values[key]["SUV_mean"]

                    print("Activity to map: {}".format(act_tomap))


            activity_per_ml = act_tomap * (self.total_activity / self.weight_total_mice)
            print("{} MBq/ml".format(
                np.round(activity_per_ml * 1 * 10 ** -6, 4)))
            print("{} Bq/voxel".format(self._voxel_volume * activity_per_ml))

             # remove heart movements (deve causar problemas quando for para por actividade no coração)

            self.activity_map_generated[compare == int(self.activity_ratios[organ]["value"])] \
                = activity_per_ml

            total_activity_in_phantom += np.sum(self.activity_map_generated[compare == int(self.activity_ratios[organ]["value"])])*self._voxel_volume
            #
            # volume_org += len(self.activity_map_generated[compare == int(self.activity_ratios[organ]["value"])])*self._voxel_volume
            self.activity_table[int(self.activity_ratios[organ]["value"]), 2] \
                = act_tomap*self.total_activity/self.weight_total_mice*self._voxel_volume

            self.attenuation_table[int(self.activity_ratios[organ]["value"]),2] = self._mapMobyToGate[organ]
            print("Total activity in phantom: {} uCi".format(total_activity_in_phantom/37000))
            # print("Volume organ: {} ml".format(volume_org))
        r = RawDataSetter(file_name=os.path.join(self.directory, "activity_map_generated.dat"), size_file_m=self.activity_map_generated.shape)
        r.write_files_simple_binary(volume=self.activity_map_generated)
        r = RawDataSetter(file_name=os.path.join(self.directory, "activity_atlas_iny.dat"),
                          size_file_m=self.activity_map_generated.shape)
        r.write_files_simple_binary(volume=compare)
        print("Total activity in phantom: {} uCi".format(total_activity_in_phantom/37000))

    def generate_attenuation_map(self):
        """ """

    def generate_gate_activity_file(self):
        """ """
        with open(os.path.join(self.directory, "activity_range_rat.dat"), "w") as file:
            file.write(str(len(self.activity_table)))
            file.write("\n")
            # file.write(str(self.activity_table))

            # for row in self.activity_table:
            #     row = row.T
            np.savetxt(file, self.activity_table, fmt='%f', delimiter=' ')

    def generate_gate_attenuation_file(self):
        """"""
        with open(os.path.join(self.directory, "range_atten_moby.dat"), "w") as file:
            file.write(str(len(self.attenuation_table)))
            file.write("\n")
            # file.write(str(self.activity_table))

            # for row in self.activity_table:
            #     row = row.T
            np.savetxt(file, self.attenuation_table, fmt='%f %f %s %s %f %f %f %f', delimiter=' ')


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    # file_path = filedialog.askopenfilename()
    file_path = "/home/crispim/Transferências/Moby_average_act_av.dataFiles"
    file_path = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\PhD\\Resultados Organizar\\MOBY\\Moby_average_act_av.bin"
    main_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                            "dataFiles", "suv_data")
    # human_excel_file_name = "SUV_values_backup.xlsx"
    # human_excel_file_path = os.path.join(main_dir, "dataFiles", human_excel_file_name)  # directory of excel file

    mice_excel_file_name = "SUV_values_brain.xlsx"
    mice_excel_file_path = os.path.join(main_dir,  mice_excel_file_name)  # directory of excel file



    activity = MobyPhantomMapGenerator(file=file_path)
    activity.readMapMobyToGate()
    activity.load_log_file()
    activity.load_original_volume()
    activity.generate_gate_attenuation_file()
    # np.savetxt(file, activity.attenuation_table, fmt='%f %f %s', delimiter=' ')
    # activity.calculate_volume_total_phantom(precise=True)
    # activity.calculate_volume_total_phantom(precise=False)
    activity.calculate_volume_organs()
    activity.generate_new_activity_map()



    activity.generate_gate_activity_file()
    activity.generate_gate_attenuation_file()
    # or_volume = activity.original_volume()
    # act_map = activity.activity_map_generated
    # plt.imshow(np.max(act_map, axis=0))
    # plt.show()
