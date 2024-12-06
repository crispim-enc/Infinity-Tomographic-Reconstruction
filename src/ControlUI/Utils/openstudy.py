import os
from ctypes import Union

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QVariant
from PyQt5.QtGui import QColor

from PyQt5 import QtWidgets
from src.EasyPETLinkInitializer.EasyPETDataReader import binary_data
from src.ControlUI.Visualization import PopulateMainWindowVTK
from src.ImageReader.DICOM.filesreader import DicomVTKReader


class OpenNewCaseStudy:
    def __init__(self, ):
        self.list_open_studies = []
        self.path_file = os.path.dirname(os.path.abspath(__file__))

        # file_name = os.path.join(path_file, 'demo_acqs_files', 'Easypet Scan NAF_FDG', 'static_image', 'im.npy')

    def updateList(self):
        file_name = QFileDialog.getOpenFileNames(None, 'Choose data file', self.path_file,
                                                 "EasyPET file (*.easypet)")
        if len(file_name[0]) == 0:
            return
        file_name = file_name[0][0]
        # self.list_open_studies.append((file_name))
        self.list_open_studies.append(file_name)
        self.selected_study = self.list_open_studies[0]
        # self.selected_study = file_name
        # self._current_whole_body_image_path = os.path.join(os.path.dirname(file_name), "static_image")
        # self._current_static_image_path = os.path.join(os.path.dirname(file_name), "static_image")
        # self._dynamic_image_path = os.path.join(os.path.dirname(file_name), "static_image")
        # self._gated_image_path = os.path.join(os.path.dirname(file_name), "static_image")
        # [listMode, Version_binary, header, dates, otherinfo] = binary_data().open(file_name)
        [listMode, Version_binary, header, dates, otherinfo, acquisitionInfo, stringdata,
         systemConfigurations_info, energyfactor_info, peakMatrix_info] = binary_data().open(file_name)
        acquisition_name = os.path.basename(os.path.normpath(file_name))
        acquisition_name = os.path.splitext(acquisition_name)[0]

        file_path = os.path.dirname(os.path.normpath(file_name))
        file_name_original = os.path.join(file_path, '{} Original data.easypetoriginal'.format(acquisition_name))

        # self._tm = DictionaryTableModel(acquisitionInfo)
        # self.tableView_info.setModel(self._tm)
        # for row in range(0, len(acquisitionInfo)):
        #     self.tableView_info.
        # EnergyWindow._energy_histogram(self, listMode)
        # EnergyWindow._add_2d_histogram(self, listMode_original=listMode_original)
        volume_static = None

        dynamic_volumes = [volume_static] * 10
        dicomreader = DicomVTKReader().DICOMreader
        # volume = DicomVTKReader().vtkImage
        self.populateMainWindowVTK.addNewDataToCurrentView(dicomReader=dicomreader)

        # PopulateDynamicViewWindowVTK._create_dynamic_view(self, list_of_volumes=dynamic_volumes)


class DictionaryTableModel(QAbstractTableModel):
    def __init__(self, data=None):
        super(DictionaryTableModel, self).__init__()
        self._data = data if data is not None else []
        self._hdr = self.gen_hdr_data() if data else []

        self._base_color = {'NewConnection': 'blue', }
        # self.data()

    def gen_hdr_data(self):
        # self._hdr = sorted(list(set().union(*(d.keys() for d in self.data))))
        self._hdr = list(self._data.keys())
        # self._hdr = "list(self._data.keys())"
        # self._hdr = ["1","2"]
        return self._hdr

    def data(self, index: QModelIndex, role: int):
        if role == Qt.DisplayRole:
            try:
                value = self._data[self._hdr[index.row()]]
                print(value)
            except KeyError:
                print("key error")
                value = None
            return str(value) if value else ""


    def rowCount(self, index):
        # The length of the outer list.
        return len(self._hdr)

    def columnCount(self, index):
        # The length of our headers.
        return 1

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data["Start date time"])

            if orientation == Qt.Vertical:
                print(str(self._hdr[section]))
                return str(self._hdr[section])

class CustomTableModel(QAbstractTableModel):
    """A model to interface a Qt view with pandas dataframe """

    def __init__(self, dataframe: pd.DataFrame, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self._dataframe = dataframe

    def rowCount(self, parent=QModelIndex()) -> int:
        """ Override method from QAbstractTableModel

        Return row count of the pandas DataFrame
        """
        if parent == QModelIndex():
            return len(self._dataframe)

        return 0

    def columnCount(self, parent=QModelIndex()) -> int:
        """Override method from QAbstractTableModel

        Return column count of the pandas DataFrame
        """
        if parent == QModelIndex():
            return len(self._dataframe.columns)
        return 0

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return data cell from the pandas DataFrame
        """
        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            return str(self._dataframe.iloc[index.row(), index.column()])

        return None

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: Qt.ItemDataRole
    ):
        """Override method from QAbstractTableModel

        Return dataframe index as vertical header data and columns as horizontal header data.
        """
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._dataframe.columns[section])

            if orientation == Qt.Vertical:
                return str(self._dataframe.index[section])

        return None
