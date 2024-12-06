import os
from src.EasyPETLinkInitializer.EasyPETDataReader import binary_data


class Metadados:
    def __init__(self, list_open_studies):
        path_folder = os.path.dirname(os.path.dirname(list_open_studies))
        folders = [f.path for f in os.scandir(path_folder) if os.path.isdir(f)]
        for folder in folders:
            file = os.path.join(folder, os.path.basename(folder) + ".easypet")
            [self.listMode, self.Version_binary, self.header, self.dates, self.otherinfo, self.acquisitionInfo,
             self.stringdata, self.systemConfigurations_info, self.energyfactor_info,
             self.peakMatrix_info] = binary_data().open(file)

            for key, value in self.acquisitionInfo.items():
                # name, age = value
                print("{}: {} ".format(key, value))
            print("-------------------")


if __name__ == "__main__":
    import os
    import tkinter as tk
    from tkinter import filedialog

    from src.EasyPETLinkInitializer.EasyPETDataReader import binary_data

    root = tk.Tk()
    root.withdraw()
    # matplotlib.rcParams['font.family'] = "Gill Sans MT"
    file_path = filedialog.askopenfilename()
    Metadados(file_path)