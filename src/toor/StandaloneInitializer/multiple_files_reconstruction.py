import os
import numpy as np

from StandaloneInitializer import ReconstructOpenFileTest


class MultipleFilesReconstruction:
    def __init__(self, list_open_studies = None, angle_correction=False, multiple_file_reconstruction=False,
                 multiple_folder=False, multiple_override_geometry=False, mrp_variation=True):
        selectable_crystals = ["Left-Left", "Right-Right", "Left-Right", "Right-Left"] # Left-Left, Right-Right, Left-Right, Right-Left, Top-Down, Down-Top
        # selectable_crystals_bool = [False]*len(selectable_crystals) # Left-Left, Right-Right, Left-Right, Right-Left, Top-Down, Down-Top
        if angle_correction:
            angles = np.arange(-1, 1.25, 0.25)
            for angle in angles:
                ReconstructOpenFileTest(list_open_studies=list_open_studies, correction_angle=angle, multiple_conditions=[True, angle],
                                        override_geometric_values=[[1.175-1.5, -1.175-1.5, 1.175-1.5, -1.175-1.5], [0, 0], [0, 0]])
                # [-1.175, 1.175, 1.175, -1.175] simulation
        elif mrp_variation:
            beta = np.array([0.2,0.3,0.5,0.7,0.8,0.9])
            kernel = np.array([3, 5, 7])
            pixel_size =np.array([0.75])

            for b in beta:
                for k in kernel:
                    for p in pixel_size:
                        ReconstructOpenFileTest(list_open_studies=list_open_studies, correction_angle=0,
                                                multiple_conditions=[True, "b_{}_k_{}_p{}".format(b,k,p)],
                                                algorithm_options=[b, k], voxel_size=p,

                                                )

        elif multiple_override_geometry:
            displ = np.arange(-2, 0, 0.1)
            displacement_x = np.array([np.repeat(displ, len(displ)), np.tile(displ, len(displ))]).T
            displacement_x = np.array([displ,displ]).T
            displacement_y = 0
            displacement_z = 0
            # for sl in range(len(selectable_crystals)):
            #     selectable_crystals_bool = [False] * len(selectable_crystals)
            #     selectable_crystals_bool[sl] = True
            for i in range(len(displacement_x)):
                print(displacement_x[i])
                override = [[-1.175+displacement_x[i,0],1.175+displacement_x[i,0],-1.175+displacement_x[i,1],1.175+displacement_x[i,1]],[displacement_y,displacement_y],
                            [displacement_z, displacement_z]]
                # ReconstructOpenFileTest(list_open_studies=list_open_studies, correction_angle=0,
                #                         multiple_conditions=[True, "{}_{}".format(displacement_x[i], selectable_crystals[sl])],
                #                         override_geometric_values=override,
                #                         selectable_crystals_bool =selectable_crystals_bool)
                ReconstructOpenFileTest(list_open_studies=list_open_studies, correction_angle=0,
                                        multiple_conditions=[True, "{}".format(displacement_x[i])],
                                        override_geometric_values=override
                                        )

        elif multiple_file_reconstruction:
            if multiple_folder:
                path_folder = os.path.dirname(os.path.dirname(list_open_studies))
                folders = [f.path for f in os.scandir(path_folder) if os.path.isdir(f)]
            else:
                folders = [list_open_studies]
            print(folders)
            for folder in folders[3:]:
                # print(folder)
                list_open_studies = os.path.join(folder,os.path.basename(folder)+".easypet")
                print(list_open_studies)
                try:
                    ReconstructOpenFileTest(list_open_studies=list_open_studies, correction_angle=0,
                                            multiple_conditions=[False, 0],
                                            override_geometric_values=[[-1.175, 1.175, -1.175, 1.175], [0, 0], [0, 0]])
                except ValueError as e:
                    print(e)

                # subfolders = [f.path for f in os.scandir(folder) if os.path.isdir(f)]
                # # subfolders=subfolders[:]
                # for subfolder in subfolders:
                #     file = subfolder.split('/')
                #     file = file[-1]
                #     # fileNamet = subfolder + "/" + file + '.easypet'
                #     list_open_studies = os.path.join(subfolder, file, '.easypet')
                #     print(subfolder)
                #     print(list_open_studies)


