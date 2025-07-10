# *******************************************************
# * FILE: NECPhantom.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

import numpy as np


class NECPhantom:
    def __init__(self):
        self._type_phantom = None
        self._material = "polyethylene"
        self._density = 0.96
        self._length = 70
        self._diameter = 25
        self._cylindricalholeDiameter = 3.2
        self._distanceHoleToCenter = 10
        self.imagePhantom = None

    def mouseLike(self):
        self._material = "polyethylene"
        self._density = 0.96
        self._length = 70
        self._diameter = 25
        self._cylindricalholeDiameter = 3.2
        self._type_phantom = "mouseLike"
        self._distanceHoleToCenter = 10

    def ratLike(self):
        self._length = 150
        self._diameter = 50
        self._cylindricalholeDiameter = 3.2
        self._type_phantom = "ratLike"
        self._distanceHoleToCenter = 17.5

    def voxelizedPhantom(self, image_size=None, voxelSize=None):
        """ """
        if image_size is not None:
            max_x = image_size[0] / 2
            min_x = -image_size[0] / 2
            max_y = image_size[1] / 2
            min_y = -image_size[1] / 2
            max_z = image_size[2] / 2
            min_z = -image_size[2] / 2
        else:
            max_x = self._diameter / 2
            min_x = -self._diameter / 2
            max_y = self._diameter / 2
            min_y = -self._diameter / 2
            max_z = self._length / 2
            min_z = -self._length / 2

        if voxelSize is None:
            voxelSize = [0.2, 0.2, 0.2]

        self.limits_max = np.array([max_x, max_y, max_z])
        self.limits_min = np.array([min_x, min_y, min_z])
        self.voxelSize = voxelSize
        x = np.arange(self.limits_min[0], self.limits_max[0], voxelSize[0])
        y = np.arange(self.limits_min[1], self.limits_max[1], voxelSize[1])
        z = np.arange(self.limits_min[2], self.limits_max[2], voxelSize[2])
        xx, yy = np.meshgrid(x, y)

        xxx = np.ascontiguousarray(np.zeros((x.shape[0], y.shape[0], z.shape[0])))
        yyy = np.ascontiguousarray(np.zeros((x.shape[0], y.shape[0], z.shape[0])))
        zzz = np.ascontiguousarray(np.zeros((x.shape[0], y.shape[0], z.shape[0])))
        # yyy = np.zeros((xx.shape[0], xx.shape[1], z.shape[0]))
        # zzz = np.zeros((xx.shape[0], xx.shape[1],z.shape[0]))
        xxx[:] = x[..., None, None]
        yyy[:] = y[None, ..., None]
        zzz[:] = z[None, None, ...]

        # plt.imshow(np.max(xxx,2)-xx)
        # plt.figure()
        # # plt.imshow(np.max(zzz,2))
        # plt.show()
        image = np.zeros(xxx.shape)
        # image = np.zeros(xx.shape)

        i = 0
        for ob in self.objects:
            print(ob.density)
            image[((xxx - ob.center[0]) ** 2 + (yyy - ob.center[1]) ** 2 <= ob.rMax ** 2) &
                  ((xxx - ob.center[0]) ** 2 + (yyy - ob.center[1]) ** 2 >= ob.rMin ** 2) &
                  (zzz >= (ob.center[2] - ob.height / 2)) & (zzz <= (ob.center[2] + ob.height / 2))] = ob.density
            # image[((xx - ob.center[0]) ** 2 + (yy - ob.center[1]) ** 2 < ob.rMax ** 2)] = i
            i += 1
        self.imagePhantom = image

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    nec = NECPhantom()
    nec.mouseLike()
    nec.voxelizedPhantom()