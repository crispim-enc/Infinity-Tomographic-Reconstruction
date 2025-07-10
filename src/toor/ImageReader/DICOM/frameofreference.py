# *******************************************************
# * FILE: frameofreference.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

from pydicom.dataset import FileDataset


class FrameOfReference(FileDataset):
    def __init__(self, filename_or_obj, dataset, ds):
        super().__init__(filename_or_obj, dataset)
        self.ds = ds
        self.ds.PositionReferenceIndicator = None

    def override(self, acquisitionInfo):
        self.ds.FrameOfReferenceUID = acquisitionInfo.Frameofreferenceuid


