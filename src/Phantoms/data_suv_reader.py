import os
import pandas as pd


class SuvReader:
    def __init__(self, excel_file):
        self.excel_file = excel_file
        self.df_study1_M = None
        self.df_study1_F = None
        self.df_study2_M = None
        self.df_study2_F = None
        self.df_study3 = None
        self.df_Tg2576 = None
        self.df_Wild_Type = None
        self.df_Naf = None

    def read_file(self):
        file_name = os.path.basename(self.excel_file)
        # file_name = self.excel_file.split('\\')[-1]
        if file_name == "SUV_values_backup.xlsx":

            self.read_human_file()
        elif file_name == "SUV_values_brain.xlsx":
            self.read_mice_file()

    def read_human_file(self):
        xls = pd.ExcelFile(self.excel_file)  # Read excel with pandas
        self.df_study1_M = xls.parse(xls.sheet_names[0], header=1)  # turns sheet into a data frame
        self.df_study1_F = xls.parse(xls.sheet_names[1], header=1)
        self.df_study2_M = xls.parse(xls.sheet_names[2], header=1)
        self.df_study2_F = xls.parse(xls.sheet_names[3], header=1)
        self.df_study3 = xls.parse(xls.sheet_names[4], header=1)

        self.df_study1_M = self.df_study1_M.set_index('organs').to_dict(orient="index")
        self.df_study1_F = self.df_study1_F.set_index('organs').to_dict(orient="index")
        self.df_study2_M = self.df_study2_M.set_index('organs').to_dict(orient="index")
        self.df_study2_F = self.df_study2_F.set_index('organs').to_dict(orient="index")
        self.df_study3 = self.df_study3.set_index('organs').to_dict(orient="index")

        for j in [self.df_study1_M, self.df_study1_F, self.df_study2_M, self.df_study2_F, self.df_study3]:
            for i in j.keys():
                if j[i]["SUV_mean"] == "None":
                    j[i]["SUV_mean"] = 0
                    j[i]["SUV_min"] = 0
                    j[i]["SUV_max"] = 0

    def read_mice_file(self):
        xls = pd.ExcelFile(self.excel_file)  # Read excel with pandas
        self.df_Tg2576 = xls.parse(xls.sheet_names[0], header=1)  # turns sheet into a data frame
        self.df_Wild_Type = xls.parse(xls.sheet_names[1], header=1)
        self.df_Naf = xls.parse(xls.sheet_names[2], header=1)
        self.df_Tg2576 = self.df_Tg2576.set_index('organs').to_dict(orient="index")
        self.df_Wild_Type = self.df_Wild_Type.set_index('organs').to_dict(orient="index")
        self.df_Naf = self.df_Naf.set_index('organs').to_dict(orient="index")
        for j in [self.df_Tg2576, self.df_Wild_Type, self.df_Naf]:
            for i in j.keys():
                if j[i]["SUV_mean"] == "None":
                    j[i]["SUV_mean"] = 0
