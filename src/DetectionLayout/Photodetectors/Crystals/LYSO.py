from src.DetectionLayout.Photodetectors.Crystals import GenericCrystal


class LYSOCrystal(GenericCrystal):
    """
    Class that represents a LYSO crystal. It contains the information about the crystal geometry and the detectors that compose it.
    Methods:


    """
    def __init__(self, crystal_id=1):
        super(LYSOCrystal, self).__init__()
        self._density = 7.4
        self._crystalID = crystal_id
        # self._crystalSizeX = 2  # mm
        # self._crystalSizeY = 2  # mm
        # self._crystalSizeZ = 20
        # self._centroid = [0, 0, 0]
        # self._vertices = np.array([[0, 0, 0],
        #                            [0, 0, 0],
        #                            [0, 0, 0],
        #                            [0, 0, 0],
        #                            [0, 0, 0],
        #                            [0, 0, 0],
        #                            [0, 0, 0],
        #                            [0, 0, 0]])
        #
        # self._volume = self._crystalSizeX * self._crystalSizeY * self._crystalSizeZ * 1e-3
        # self._mass = self._density * self._volume

