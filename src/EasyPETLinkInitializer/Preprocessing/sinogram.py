import numpy as np
import matplotlib.pyplot as plt


class Sinogram:
    def __init__(self, listMode=None, parametric=None, range_s=None, range_phi=None, range_z=None):
        if listMode is None or parametric is None:
            return

        if range_s is None:
            range_s = [-30, 30]

        if range_phi is None:
            range_phi = [0, 360]

        if range_z is None:
            range_z = [0, 64]

        self.listMode = listMode
        self.parametric = parametric
        self.s = None
        self.phi = None
        self.max_s = range_s[1]
        self.min_s = range_s[0]
        self.max_phi = range_phi[1]
        self.min_phi = range_phi[0]
        self.z_min = range_z[0]
        self.z_max = range_z[0]
        self._michelogram = False
        self._projected_sinogram = None

    def calculate_s_phi(self):
        xi = self.parametric.xi
        yi = self.parametric.yi
        xf = self.parametric.xf
        yf = self.parametric.yf

        p1 = np.column_stack((xi, yi))
        p2 = np.column_stack((xf, yf))
        p3 = np.copy(p1) * 0
        p4 = (p1 + p2) / 2

        # phi = phi%360
        v1 = p1 - p3
        # v1 =  v1 / np.sqrt(v1[:, 0] ** 2 + v1[:, 1] ** 2)
        v2 = p4 - p3
        # v2 = v2 / np.sqrt(v2[:, 0] ** 2 + v2[:, 1] ** 2)
        n1 = (np.sqrt(v1[:, 0] ** 2 + v1[:, 1] ** 2))

        abcissa = (xf - xi)
        declive = np.zeros(abcissa.shape)
        declive[abcissa != 0] = (yf - yi)[abcissa != 0] / abcissa[abcissa != 0]

        phi = np.degrees(np.arctan(declive))
        phi[np.sign(xi - xf) == -1] += 180
        # phi[np.sign(xi - xf) == -1] *= -1
        #
        # s = (np.cross(v1, v2) / n1)
        s = np.sqrt(v2[:, 0] ** 2 + v2[:, 1] ** 2)
        # s = np.cross(v1, v2)
        # norm_s = np.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2)
        # s = s / norm_s[:, None]
        self.s = s
        self.phi = phi
        self.s = np.round(self.s, 2)
        self.phi = np.round(self.phi, 2)

    def updateLimits(self):
        s_max = np.abs(self.s).max()

        self.min_s = -s_max
        self.max_s = s_max

        self.max_phi = self.phi.max()
        self.min_phi = self.phi.min()
        min_z = np.zeros(2)
        min_z[0] = np.round(self.parametric.zi, 4).min()
        min_z[1] = np.round(self.parametric.zf, 4).min()

        max_z = np.zeros(2)
        max_z[0] = np.round(self.parametric.zi, 4).max()
        max_z[1] = np.round(self.parametric.zf, 4).max()
        self.z_min = min_z.min()
        self.z_max = max_z.max()

    def projected_sinogram(self, bins_x=100, bins_y=200, rebining_x=100, rebining_y=100):
        if bins_x is None:
            bins_x = int(len(np.unique(self.phi)) / rebining_x)

        if bins_y is None:
            bins_y = int(len(np.unique(self.s)) / rebining_y)

        self._projected_sinogram = plt.hist2d(self.phi, self.s, bins=[bins_x, bins_y],
                                              range=[[self.min_phi, self.max_phi],
                                                     [self.min_s, self.max_s]])
        return self._projected_sinogram

    def calculateMichelogram(self, f2f_or_reb="reb", bins_x=None,bins_y=None, timecut=None):
        if f2f_or_reb == "f2f":
            self.front2Front(bins_x=bins_x, bins_y=bins_y, timecut=timecut)
        elif f2f_or_reb == "reb":
            self.rebinningSSBR(bins_x=bins_x, bins_y=bins_y, timecut=timecut)
        else:
            raise KeyError

    def front2Front(self, bins_x=100, bins_y=200, timecut=None):
        zf = np.round(self.parametric.zf, 4)
        zi = np.round(self.parametric.zi, 4)
        zi_unique_values = np.unique(zi)
        number_of_sino = len(zi_unique_values)

        if timecut is not None:
            s = self.s[timecut[0]:timecut[1]]
            phi = self.phi[timecut[0]:timecut[1]]
            zi = zi[timecut[0]:timecut[1]]
            zf = zf[timecut[0]:timecut[1]]

        else:
            s = self.s
            phi = self.phi

        print("Number_of_events before:{}".format(len(zi)))
        mask = (zi == zf)
        s = s[mask]
        phi = phi[mask]
        zi = zi[mask]
        print("Number_of_events after:{}".format(len(zi)))
        temp_michelogram = np.histogramdd((phi, s, zi), bins=[bins_x, bins_y, number_of_sino],
                                          range=[[self.min_phi, self.max_phi],
                                                 [self.min_s, self.max_s], [self.z_min, self.z_max]])
        self._michelogram = [temp_michelogram[0], temp_michelogram[1][0], temp_michelogram[1][1],
                             temp_michelogram[1][2]]
        self._michelogram = tuple(self._michelogram)

    def rebinningSSBR(self, bins_x=100, bins_y=200, rebining_x=100, rebining_y=100, timecut=None):

        if bins_x is None:
            bins_x = int(len(np.unique(self.phi)) / rebining_x)

        if bins_y is None:
            bins_y = int(len(np.unique(self.s)) / rebining_y)
        zf = np.round(self.parametric.zf, 4)
        zi = np.round(self.parametric.zi, 4)


        if timecut is not None:
            s = self.s[timecut[0]:timecut[1]]
            phi = self.phi[timecut[0]:timecut[1]]
            zi = zi[timecut[0]:timecut[1]]
            zf = zf[timecut[0]:timecut[1]]

        else:
            s = self.s
            phi = self.phi

        mask = np.abs(zf - zi) <= 4
        s = s[mask]
        phi = phi[mask]
        zi = zi[mask]
        zf = zf[mask]

        zi = np.round((zf + zi) / 2, 5)
        zi_unique_values = np.unique(zi)
        number_of_sino = len(zi_unique_values)

        temp_michelogram = np.histogramdd((phi, s, zi), bins=[bins_x, bins_y, number_of_sino],
                                          range=[[self.min_phi, self.max_phi],
                                                 [self.min_s, self.max_s], [self.z_min, self.z_max]])
        self._michelogram = [temp_michelogram[0], temp_michelogram[1][0], temp_michelogram[1][1], temp_michelogram[1][2]]
        self._michelogram = tuple(self._michelogram)

    @property
    def michelogram(self):
        return self._michelogram


if __name__ == "__main__":
    pass
