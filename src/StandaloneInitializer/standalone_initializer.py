import os
from src.EasyPETLinkInitializer.Preprocessing import PrepareEasyPETdata
from src.StandaloneInitializer.reconstruction_initializer import ReconstructionInitializer
from src.EasyPETLinkInitializer.Utilities.exceptions import OnlyOneIncompleteTurn


class ReconstructOpenFileTest:
    def __init__(self, list_open_studies=None, correction_angle=0, number_of_iterations=100, voxel_size=0.5,
                 algorithm="LM-MLEM", algorithm_options=None, reconstruction_data_type=".easypet",
                 crystals_geometry=None, parameters_2D_cut=None,
                 type_of_reconstruction=None, threshold_ratio=0.75, energy_window=None,
                 multiple_conditions=None, override_geometric_values=None,
                 selectable_crystals_bool=None, calculate_quantification_factors=False, **kwargs):

        if energy_window is None:
            peak_A = 511
            peak_B = 511
            energy_window = [peak_A - peak_A * 0.5, peak_A + peak_A * 0.5]
            # energy_window = [350, 650]
        if parameters_2D_cut is None:
            parameters_2D_cut = [0, 0, 0, 0]
        # if crystals_geometry is None:
        #     crystals_geometry = [32, 2]
        if algorithm_options is None:
            algorithm_options = [0.05, 3]
        if multiple_conditions is None:
            multiple_conditions = [False, None]
        if type_of_reconstruction is None:
            type_of_reconstruction = [True, False, False, False]

        type_of_reconstruction_tag = ["whole_body", "static_image", "dynamic_image", "gated_image"]

        print(override_geometric_values)
        self.correction_angle = correction_angle
        self.number_of_iterations = number_of_iterations
        self.whole_body = type_of_reconstruction[0]
        self.static_recon = type_of_reconstruction[1]
        self.dynamic_recon = type_of_reconstruction[2]
        self.gated_recon = type_of_reconstruction[3]
        selectable_crystals = ["Left-Left", "Right-Right", "Left-Right", "Right-Left"]
        self.selectable_crystals_bool = selectable_crystals_bool
        # self.selectable_crystals_bool = [True, False, False, False]
        # self.selectable_crystals_bool = [False, True, False, False]
        if self.selectable_crystals_bool is None:
            self.selectable_crystals_bool = [False] * 6
        self.number_of_subsets = 1
        self.number_cumulative_turns = 4
        self.voxel_size = voxel_size
        self.detector_sensibility_correction = True
        self.detector_sensibility_data = "Simulation"
        self.attenuation_correction = True
        self.attenuation_data = "External"
        self.decay_correction = True
        self.positron_range_correction = False
        self.doi_correction = False
        self.scatter_correction = False
        self.random_correction = False
        self.respiratory_movement_correction = False
        self.heart_movement_correction = False
        self.algorithm = algorithm
        self.algorithm_options = algorithm_options
        self.crystals_geometry = crystals_geometry
        # parameters_2D_cut = [1050, 600, 1050, 600]
        self.parameters_2D_cut = parameters_2D_cut
        self.type_of_reconstruction = type_of_reconstruction
        self.reconstruction_data_type = reconstruction_data_type
        # reconstruction_data_type = ".easypet"
        # energy_window = [450, 550]
        self.threshold_ratio = threshold_ratio
        self.energy_window = energy_window
        self.remove_turns = {
            "Cut_per_time": True,
            "Init time": 3600, #3600
            "End time": 5500, #87000
            "Whole body": True,
            "Dynamic": False,
            "Static": False,
            "Gated": False}
        self.remove_turns = None #7708

        # energy_window = [320, 720]
        # energy_window = [200, 380]
        # energy_window = [0, 1024]
        simulation_file = True

        if simulation_file:
            remove_incomplete_turn = False
            save_validation_data = True
            swap_sideAtoB = False
            self.reconstruction_data_type = ".easypet"
        else:
            remove_incomplete_turn = True
            save_validation_data = True
            swap_sideAtoB = True

        # Rewrite entries
        for key, value in kwargs.items():
            setattr(self, key, value)
        # remove_incomplete_turn = False  # simulation
        # self.selectable_crystals_bool= [True, False, False, False]
        # orig_sys = sys.stdout
        # make output console dir

        try:
            try:
                coincidence_window = self.coincidence_window

            except AttributeError:
                coincidence_window = None
            # coincidence_window = 300
            prepareEasypetdata = PrepareEasyPETdata(study_file=list_open_studies,
                                                    reconstruction_data_type=self.reconstruction_data_type,
                                                    top_correction_angle=self.correction_angle,
                                                    parameters_2D_cut=self.parameters_2D_cut,
                                                    energy_window=self.energy_window,
                                                    threshold_ratio=self.threshold_ratio,
                                                    save_spectrum_file=False,
                                                    save_validation_data=save_validation_data,
                                                    swap_sideAtoB=swap_sideAtoB,
                                                    remove_incomplete_turn=remove_incomplete_turn,
                                                    simulation_file=simulation_file,
                                                    only_left_side_crystals=self.selectable_crystals_bool[0],
                                                    only_right_side_crystals=self.selectable_crystals_bool[1],
                                                    right_left_crystals=self.selectable_crystals_bool[2],
                                                    left_right_crystals=self.selectable_crystals_bool[3],
                                                    coincidence_window=coincidence_window
                                                    )
            prepareEasypetdata.dataRemove()

        except OnlyOneIncompleteTurn:
            print("Only one imcplete turn")
            return

        except FileNotFoundError as e:
            print(e)
            raise FileNotFoundError

        # detector_normalization = "Simulation"
        detector_normalization = "On demand parametric calculation"
        algorithm_function = ReconstructionInitializer(Easypetdata=prepareEasypetdata,
                                                       algorithm=self.algorithm,
                                                       detector_normalization_correction=True,
                                                       algorithm_options=self.algorithm_options,
                                                       study_path=os.path.dirname(list_open_studies),
                                                       number_of_iterations=self.number_of_iterations,
                                                       pixelSizeXY=self.voxel_size, pixelSizeXYZ=self.voxel_size,
                                                       type_of_reconstruction=type_of_reconstruction,
                                                       detector_normalization_data=detector_normalization,
                                                       simulation_file=simulation_file,
                                                       decay_correction=self.decay_correction,
                                                       multiple_save_conditions=multiple_conditions,
                                                       override_geometry_file=True,
                                                       override_geometric_values=override_geometric_values,
                                                       remove_turns=self.remove_turns,
                                                       number_cumulative_turns=self.number_cumulative_turns,
                                                       kwargs=kwargs)
        if calculate_quantification_factors:
            algorithm_function.generate_probability_and_quantification_factors()  # fantoma
        else:
            algorithm_function.start()

        # for t in range(len(type_of_reconstruction)):
        #     if type_of_reconstruction[t]:
        #         folder = os.path.join(os.path.dirname(list_open_studies), type_of_reconstruction_tag[t])
        #         dicom_folder = np.load(os.path.join(folder, "series_number.npy"))
        #         attrs = vars(algorithm_function)
        #         # {'kids': 0, 'name': 'Dog', 'color': 'Spotted', 'age': 10, 'legs': 2, 'smell': 'Alot'}
        #         # now dump this in some way or another
        #         try:
        #             file_path = os.path.join(folder, str(int(dicom_folder) - 1), "reconstruction_parameters.txt")
        #             with open(file_path, 'w') as f:
        #                 f.write('\n'.join("%s: %s" % item for item in attrs.items()))
        #         except FileNotFoundError:
        #             pass

        #
        # volume_static = algorithm_function.im
