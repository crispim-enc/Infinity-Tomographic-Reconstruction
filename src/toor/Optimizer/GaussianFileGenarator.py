import numpy as np


class GaussianParameters:
    def __init__(self, voxelSize=None, shiftInvariant=True, deviceObject=None, radioIsotope=None):
        if voxelSize is None:
            print("Voxel size not provided. Using default value of [1, 1, 1]. Could lead to incorrect results.")
            voxelSize = [1, 1, 1]
        self.shiftInvariant = shiftInvariant
        self.deviceObject = deviceObject
        self.radioIsotope = radioIsotope
        self.filename = None
        self.sigma_y = 0.42
        self.sigma_z = 0.42
        self._sigma_y_square = np.float32(self.sigma_y**2/voxelSize[1])
        self._sigma_z_square = np.float32(self.sigma_z**2/voxelSize[2])
        self._acceptableYDistance = None
        self._acceptableZDistance = None
        self._gaussian_y_fix_term = None
        self._gaussian_z_fix_term = None
        self._invert2timesigma_z_square = None
        self._invert2timesigma_y_square = None
        print("Voxel size: ", voxelSize)
        print("Sigma y square: ", self._sigma_y_square)
        print("Sigma z square: ", self._sigma_z_square)

    def setShiftInvariantParameters(self):
        self._acceptableYDistance = np.float32(11.09205 * self._sigma_y_square)
        self._acceptableZDistance = np.float32(11.09205 * self._sigma_z_square)
        self._gaussian_y_fix_term = np.float32(1/(np.sqrt(2*np.pi*self._sigma_y_square)))
        self._gaussian_z_fix_term = np.float32(1/(np.sqrt(2*np.pi*self._sigma_z_square)))
        self._invert2timesigma_z_square = np.float32(1 / (2 * self._sigma_z_square))
        self._invert2timesigma_y_square = np.float32(1 / (2 * self._sigma_y_square))
        print("Acceptable Y distance: ", self._acceptableYDistance)
        print("Acceptable Z distance: ", self._acceptableZDistance)
        print("Gaussian y fix term: ", self._gaussian_y_fix_term)
        print("Gaussian z fix term: ", self._gaussian_z_fix_term)
        print("Invert 2 times sigma z square: ", self._invert2timesigma_z_square)
        print("Invert 2 times sigma y square: ", self._invert2timesigma_y_square)

    def setShiftVariantParameters(self, FWHM_tangential, FWHM_axial):
        sigma_y = FWHM_tangential / 2.355
        sigma_z = FWHM_axial / 2.355
        self._sigma_y_square = np.ascontiguousarray(sigma_y**2, dtype=np.float32)
        self._sigma_z_square = np.ascontiguousarray(sigma_z**2, dtype=np.float32)
        self._acceptableYDistance = np.ascontiguousarray(11.09205 * self._sigma_y_square, dtype=np.float32)
        self._acceptableZDistance = np.ascontiguousarray(11.09205 * self._sigma_z_square, dtype=np.float32)
        self._gaussian_y_fix_term = np.ascontiguousarray(1/(np.sqrt(2*np.pi*self._sigma_y_square)), dtype=np.float32)
        self._gaussian_z_fix_term = np.ascontiguousarray(1/(np.sqrt(2*np.pi*self._sigma_z_square)), dtype=np.float32)
        self._invert2timesigma_z_square = np.ascontiguousarray(1 / (2 * self._sigma_z_square), dtype=np.float32)
        self._invert2timesigma_y_square = np.ascontiguousarray(1 / (2 * self._sigma_y_square), dtype=np.float32)

    @property
    def sigma_y_square(self):
        return self._sigma_y_square

    @property
    def sigma_z_square(self):
        return self._sigma_z_square

    @property
    def acceptableYDistance(self):
        return self._acceptableYDistance

    @property
    def acceptableZDistance(self):
        return self._acceptableZDistance

    @property
    def gaussian_y_fix_term(self):
        return self._gaussian_y_fix_term

    @property
    def gaussian_z_fix_term(self):
        return self._gaussian_z_fix_term

    @property
    def invert2timesigma_z_square(self):
        return self._invert2timesigma_z_square

    @property
    def invert2timesigma_y_square(self):
        return self._invert2timesigma_y_square

