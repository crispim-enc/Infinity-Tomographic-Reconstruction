import numpy as np
import skimage
from skimage.transform import iradon, iradon_sart
from skimage.filters import gaussian
from EasyPETLinkInitializer.Preprocessing import Sinogram


class AnalyticalReconstruction:
    def __init__(self, initial_points=None, end_points=None,
                 pixel_size=0.2, type_filter="ramp", regularization=None, rebinning="reb"):

        self.interpolation = "cubic"
        self.clip = None
        self.projection_shifts = None
        self.type_filter = type_filter
        self.pixel_size = pixel_size
        self.regularization = regularization
        self.rebinning = rebinning
        self.im = None

        self.sinoClass = Sinogram(initialPoints=initial_points,
                                  endPoints=end_points)
        self.sinoClass.calculate_s_phi()
        self.sinoClass.updateLimits()
        self.michelogram = None
        self.relaxation = 0.15

    def michelogramRegularization(self):
        if self.regularization == "gaussian":
            michelogram_filtered = gaussian(self.michelogram, sigma=1.5, output=None, mode='nearest', cval=0,
                                            multichannel=None, preserve_range=False, truncate=4.0)
            return michelogram_filtered

    def _prepareData(self, timecut=None):
        phi = self.sinoClass.phi
        # bins_x = len(np.unique(phi))
        bins_x = 200
        bins_y = int(np.ceil(2 * self.sinoClass.max_s / self.pixel_size))
        self.sinoClass.calculateMichelogram(f2f_or_reb=self.rebinning, bins_x=bins_x, bins_y=bins_y, timecut=timecut)
        self.michelogram = self.sinoClass.michelogram
        if self.regularization is not None:
            self.michelogramRegularization()
        self.im = np.zeros(
            (int(self.michelogram[0].shape[1]), int(self.michelogram[0].shape[1]), self.michelogram[0].shape[2]))

    def FBP2D(self, timecut=None):
        self._prepareData(timecut=timecut)
        theta = self.michelogram[1][:-1]
        for k in range(self.michelogram[0].shape[2]):
            if skimage.__version__ < "0.2":
                self.im[:, :, k] = iradon(self.michelogram[0][:, :, k].T, theta=theta, circle=True,
                                          filter=self.type_filter,
                                          interpolation=self.interpolation,
                                          output_size=int(self.michelogram[0].shape[1]))
            else:
                self.im[:, :, k] = iradon(self.michelogram[0][:, :, k].T, theta=theta, circle=True,
                                          filter_name=self.type_filter,
                                          interpolation=self.interpolation, output_size=int(self.michelogram[0].shape[1]))

    def SART(self):
        self._prepareData()
        theta = self.michelogram[1][:-1]
        for k in range(self.michelogram[0].shape[2]):
            self.im[:, :, k] = iradon_sart(self.michelogram[0][:, :, k].T, theta=theta, image=None,
                                           projection_shifts=self.projection_shifts, clip=self.clip, relaxation=self.relaxation)
