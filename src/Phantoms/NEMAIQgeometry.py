
#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#
"""
Created by Pedro Encarnação: Universidade de Aveiro 2023
based on NEMA Specification NU 4-2008: Image Quality Phantom
Geron Bindseil, Western University, 2012.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
# from src.Geometry import GeometryDesigner
from src.Phantoms import CylindricalStructure


class NEMAIQ2008NU:
    def __init__(self, centerPhantom=None, alpha=0, beta=0, sigma=0):
        # Generate the documentation
        """
        NEMA IQ NU 4-2008 Phantom
        :param centerPhantom: Center of the phantom
        :param alpha: Rotation around the z axis
        :param beta: Rotation around the y axis
        :param sigma: Rotation around the x axis

        """
        if centerPhantom is None:
            centerPhantom = [0, 0, 0]

        self.alphaRotation = alpha # Rotation around the z axis
        self.betaRotation = beta # Rotation around the y axis
        self.sigmaRotation = sigma # Rotation around the x axis
        self.xTranslation = 0 # Translation in the x axis
        self.yTranslation = 0 # Translation in the y axis
        self.zTranslation = 0 # Translation in the z axis
        self.centerPhantom = centerPhantom
        # Total phantom
        self._globalPhantom = CylindricalStructure()
        self._globalPhantom.setMaterial("Vacuum")
        self._globalPhantom.setRMin(0)
        self._globalPhantom.setRMax(16.75)
        self._globalPhantom.setHeight(80)
        self._globalPhantom.setCenter(np.array(centerPhantom, dtype=np.float32))

        # Body hollow part (top) that is attached to the fixed top cover
        self._bodyHollow = CylindricalStructure()
        self._bodyHollow.setMaterial("Plexiglass")
        self._bodyHollow.setRMin(15)
        self._bodyHollow.setRMax(16.75)
        self._bodyHollow.setHeight(30)
        self._bodyHollow.setCenter(np.array([0, 0, 15]))

        # Water contained in the chamber
        self._bodyHollowWater = CylindricalStructure()
        self._bodyHollowWater.setMaterial("Water")
        self._bodyHollowWater.setRMin(0)
        self._bodyHollowWater.setRMax(15)
        self._bodyHollowWater.setHeight(30)
        self._bodyHollowWater.setCenter(np.array([0, 0, 15]))

        # Now the twin cold chambers (chamber 1 has Air, chamber 2 has Water)
        # Fixed top cover chamber #1 sitting in the water
        self._airChamberStructure = CylindricalStructure()
        self._airChamberStructure.setMaterial("Plexiglass")
        self._airChamberStructure.setRMin(0)
        self._airChamberStructure.setRMax(5)
        self._airChamberStructure.setHeight(15)
        self._airChamberStructure.setCenter(np.array([7.5, 0, 7.5]) + self._bodyHollowWater.center)

        self._airChamberFilling = CylindricalStructure()
        self._airChamberFilling.setMaterial("AirBodyInterface")
        self._airChamberFilling.setRMin(0)
        self._airChamberFilling.setRMax(4)
        self._airChamberFilling.setHeight(14)
        self._airChamberFilling.setCenter(np.array([7.5, 0, 7.5]) + self._bodyHollowWater.center)

        self._waterChamberStructure = CylindricalStructure()
        self._waterChamberStructure.setMaterial("Plexiglass")
        self._waterChamberStructure.setRMin(0)
        self._waterChamberStructure.setRMax(5)
        self._waterChamberStructure.setHeight(15)
        self._waterChamberStructure.setCenter(np.array([-7.5, 0, 7.5]) + self._bodyHollowWater.center)

        self._waterChamberFilling = CylindricalStructure()
        # self._waterChamberFilling.setMaterial("Water")
        self._waterChamberFilling.setMaterial("AirBodyInterface")
        self._waterChamberFilling.setRMin(0)
        self._waterChamberFilling.setRMax(4)
        self._waterChamberFilling.setHeight(14)
        self._waterChamberFilling.setCenter(np.array([-7.5, 0, 7.5]) + self._bodyHollowWater.center)
        # Fixed top cover
        self._fixedTopCover = CylindricalStructure()
        self._fixedTopCover.setMaterial("Plexiglass")
        self._fixedTopCover.setRMin(0)
        self._fixedTopCover.setRMax(16.75)
        self._fixedTopCover.setHeight(5)
        self._fixedTopCover.setCenter(np.array([0, 0, 32.5]))

        # Shaft 1
        self._shaft1 = CylindricalStructure()
        self._shaft1.setMaterial("A150_Tissue_Plastic")
        self._shaft1.setRMin(0)
        self._shaft1.setRMax(1.75)
        self._shaft1.setHeight(5)
        self._shaft1.setCenter(np.array([7.5, 0, 0]) + self._fixedTopCover.center)

        # Shaft 2
        self._shaft2 = CylindricalStructure()
        self._shaft2.setMaterial("A150_Tissue_Plastic")
        self._shaft2.setRMin(0)
        self._shaft2.setRMax(1.75)
        self._shaft2.setHeight(5)
        self._shaft2.setCenter(np.array([-7.5, 0, 0]) + self._fixedTopCover.center)

        # Shaft 3
        self._shaft3 = CylindricalStructure()
        self._shaft3.setMaterial("A150_Tissue_Plastic")
        self._shaft3.setRMin(0)
        self._shaft3.setRMax(1.75)
        self._shaft3.setHeight(5)
        self._shaft3.setCenter(np.array([0, 7.5, 0]) + self._fixedTopCover.center)

        # O-Ring
        self._oRing1 = CylindricalStructure()
        self._oRing1.setMaterial("ABS")
        self._oRing1.setRMin(2)
        self._oRing1.setRMax(3.65)
        self._oRing1.setHeight(1.5)
        self._oRing1.setCenter(np.array([7.5, 0, 35.75]))

        # O-Ring
        self._oRing2 = CylindricalStructure()
        self._oRing2.setMaterial("ABS")
        self._oRing2.setRMin(2)
        self._oRing2.setRMax(3.65)
        self._oRing2.setHeight(1.5)
        self._oRing2.setCenter(np.array([-7.5, 0, 35.75]))

        # O-Ring
        self._oRing3 = CylindricalStructure()
        self._oRing3.setMaterial("ABS")
        self._oRing3.setRMin(2)
        self._oRing3.setRMax(3.65)
        self._oRing3.setHeight(1.5)
        self._oRing3.setCenter(np.array([0, 7.5, 35.75]))

        # screw head
        self._screwHead1 = CylindricalStructure()
        self._screwHead1.setMaterial("A150_Tissue_Plastic")
        self._screwHead1.setRMin(0)
        self._screwHead1.setRMax(4.2)
        self._screwHead1.setHeight(2.3)
        self._screwHead1.setCenter(np.array([7.5, 0, 37.65]))

        # screw head
        self._screwHead2 = CylindricalStructure()
        self._screwHead2.setMaterial("A150_Tissue_Plastic")
        self._screwHead2.setRMin(0)
        self._screwHead2.setRMax(4.2)
        self._screwHead2.setHeight(2.3)
        self._screwHead2.setCenter(np.array([-7.5, 0, 37.65]))

        self._screwHead3 = CylindricalStructure()
        self._screwHead3.setMaterial("A150_Tissue_Plastic")
        self._screwHead3.setRMin(0)
        self._screwHead3.setRMax(4.2)
        self._screwHead3.setHeight(2.3)
        self._screwHead3.setCenter(np.array([0, 7.5, 37.65]))

        # Now do the bottom half which is solid plexiglass with holes of various sizes and a filling cap.
        self._bottomBody = CylindricalStructure()
        self._bottomBody.setMaterial("AirBodyInterface")
        self._bottomBody.setRMin(0)
        self._bottomBody.setRMax(15)
        self._bottomBody.setHeight(28)
        self._bottomBody.setCenter(np.array([0, 0, -14]))

        self._bottomBodyStructure = CylindricalStructure()
        self._bottomBodyStructure.setMaterial("Plexiglass")
        self._bottomBodyStructure.setRMin(15)
        self._bottomBodyStructure.setRMax(16.75)
        self._bottomBodyStructure.setHeight(28)
        self._bottomBodyStructure.setCenter(np.array([0, 0, -14]))

        self._rod1mm = CylindricalStructure()
        self._rod1mm.setMaterial("Water")
        self._rod1mm.setRMin(0)
        self._rod1mm.setRMax(0.5)
        self._rod1mm.setHeight(20)
        self._rod1mm.setCenter(np.array([-2.16, -6.66, 4] + self._bottomBody.center, dtype=np.float32))

        self._rod2mm = CylindricalStructure()
        self._rod2mm.setMaterial("Water")
        self._rod2mm.setRMin(0)
        self._rod2mm.setRMax(1)
        self._rod2mm.setHeight(20)
        self._rod2mm.setCenter(np.array([5.66, -4.11, 4] + self._bottomBody.center, dtype=np.float32))

        self._rod3mm = CylindricalStructure()
        self._rod3mm.setMaterial("Water")
        self._rod3mm.setRMin(0)
        self._rod3mm.setRMax(1.5)
        self._rod3mm.setHeight(20)
        self._rod3mm.setCenter(np.array([5.66, 4.11, 4] + self._bottomBody.center, dtype=np.float32))

        self._rod4mm = CylindricalStructure()
        self._rod4mm.setMaterial("Water")
        self._rod4mm.setRMin(0)
        self._rod4mm.setRMax(2)
        self._rod4mm.setHeight(20)
        self._rod4mm.setCenter(np.array([-2.16, 6.66, 4] + self._bottomBody.center, dtype=np.float32))

        self._rod5mm = CylindricalStructure()
        self._rod5mm.setMaterial("Water")
        self._rod5mm.setRMin(0)
        self._rod5mm.setRMax(2.5)
        self._rod5mm.setHeight(20)
        self._rod5mm.setCenter(np.array([-7, 0, 4] + self._bottomBody.center, dtype=np.float32))

        # Wide hole at the end of the bottom lid.
        self._wideHoleLid = CylindricalStructure()
        self._wideHoleLid.setMaterial("Water")
        self._wideHoleLid.setRMin(0)
        self._wideHoleLid.setRMax(10)
        self._wideHoleLid.setHeight(3)
        self._wideHoleLid.setCenter(np.array([0, 0, -7.5] + self._bottomBody.center, dtype=np.float32))

        # Wide hole at the end of the bottom lid.
        self._wideORing = CylindricalStructure()
        self._wideORing.setMaterial("ABS")
        self._wideORing.setRMin(11.175)
        self._wideORing.setRMax(12.825)
        self._wideORing.setHeight(1)
        self._wideORing.setCenter(np.array([0, 0, -9] + self._bottomBody.center, dtype=np.float32))

        self._smallShaftScrews = [CylindricalStructure() for i in range(6)]
        _rminShaftScrews = [0 for i in range(6)]
        _rmaxShaftScrews = [1.45 for i in range(6)]
        _heightShaftScrews = [9 for i in range(6)]
        _materialShaftScrews = ["A150_Tissue_Plastic" for i in range(6)]
        _centerShaftScrews = [[14, 0, -9.5], [7, 12.12, -9.5], [-7, 12.12, -9.5],
                              [-14, 0, -9.5], [-7, -12.12, -9.5], [7, -12.12, -9.5]
                              ]

        self._airSmallShaftScrews = [CylindricalStructure() for i in range(6)]
        _rminAirShaftScrews = [0 for i in range(6)]
        _rmaxAirShaftScrews = [1.45 for i in range(6)]
        _heightAirShaftScrews = [4.5 for i in range(6)]
        _materialAirShaftScrews = ["AirBodyInterface" for i in range(6)]
        _centerAirShaftScrews = [[14, 0, -2.75], [7, 12.12, -2.75], [-7, 12.12, -2.75],
                                 [-14, 0, -2.75], [-7, -12.12, -2.75], [7, -12.12, -2.75]
                                 ]

        self._smallScrews = [CylindricalStructure() for i in range(6)]
        _rminScrews = [0 for i in range(6)]
        _rmaxScrews = [2.25 for i in range(6)]
        _heightScrews = [2.3 for i in range(6)]
        _materialScrews = ["A150_Tissue_Plastic" for i in range(6)]
        _centerScrews = [[14, 0, -29.15], [7, 12.12, -29.15], [-7, 12.12, -29.15],
                         [-14, 0, -29.15], [-7, -12.12, -29.15], [7, -12.12, -29.15]]
        for i in range(len(self._smallShaftScrews)):
            self._smallShaftScrews[i].setMaterial(_materialShaftScrews[i])
            self._smallShaftScrews[i].setRMin(_rminShaftScrews[i])
            self._smallShaftScrews[i].setRMax(_rmaxShaftScrews[i])
            self._smallShaftScrews[i].setHeight(_heightShaftScrews[i])
            self._smallShaftScrews[i].setCenter(np.array(_centerShaftScrews[i]) + self._bottomBody.center)

            self._airSmallShaftScrews[i].setMaterial(_materialAirShaftScrews[i])
            self._airSmallShaftScrews[i].setRMin(_rminAirShaftScrews[i])
            self._airSmallShaftScrews[i].setRMax(_rmaxAirShaftScrews[i])
            self._airSmallShaftScrews[i].setHeight(_heightAirShaftScrews[i])
            self._airSmallShaftScrews[i].setCenter(np.array(_centerAirShaftScrews[i]) + self._bottomBody.center)

            self._smallScrews[i].setMaterial(_materialScrews[i])
            self._smallScrews[i].setRMin(_rminScrews[i])
            self._smallScrews[i].setRMax(_rmaxScrews[i])
            self._smallScrews[i].setHeight(_heightScrews[i])
            self._smallScrews[i].setCenter(np.array(_centerScrews[i]))

        # central shaft
        self._shaft4 = CylindricalStructure()
        self._shaft4.setMaterial("ABS")
        self._shaft4.setRMin(0)
        self._shaft4.setRMax(1.75)
        self._shaft4.setHeight(5)
        self._shaft4.setCenter(np.array([0, 0, -11.5], dtype=np.float32) + self._bottomBody.center)

        # O-Ring
        self._oRing4 = CylindricalStructure()
        self._oRing4.setMaterial("ABS")
        self._oRing4.setRMin(2)
        self._oRing4.setRMax(3.65)
        self._oRing4.setHeight(1.5)
        self._oRing4.setCenter(np.array([0, 0, -28.75]))

        # O-Ring
        self._screwHead4 = CylindricalStructure()
        self._screwHead4.setMaterial("A150_Tissue_Plastic")
        self._screwHead4.setRMin(0)
        self._screwHead4.setRMax(4.2)
        self._screwHead4.setHeight(2.3)
        self._screwHead4.setCenter(np.array([0, 0, -30.65]))

        self._screwHeadShaft4 = CylindricalStructure()
        self._screwHeadShaft4.setMaterial("A150_Tissue_Plastic")
        self._screwHeadShaft4.setRMin(0)
        self._screwHeadShaft4.setRMax(1.75)
        self._screwHeadShaft4.setHeight(1.5)
        self._screwHeadShaft4.setCenter(np.array([0, 0, -28.75]))

        self.objects = [self._globalPhantom, self._bodyHollow, self._bodyHollowWater,
                        self._airChamberStructure, self._airChamberFilling, self._waterChamberStructure,
                        self._waterChamberFilling, self._fixedTopCover, self._shaft1, self._shaft2, self._shaft3,
                        self._oRing1, self._oRing2, self._oRing3, self._screwHead1, self._screwHead2,
                        self._screwHead3, self._wideHoleLid, self._wideORing,
                        self._bottomBody, self._bottomBodyStructure, self._rod1mm, self._rod2mm, self._rod3mm, self._rod4mm,
                        self._rod5mm, ] \
                       + self._smallShaftScrews + self._smallScrews + self._airSmallShaftScrews + [self._shaft4, self._oRing4,self._screwHead4, self._screwHeadShaft4]

        for ob in self.objects:
            ob.setCenter(ob.center + centerPhantom)

        self.voxelizedPhantom = None
        self.voxelSize = None
        # self.rotateAndTranslate(alpha=180)
        # self.rotateAndTranslate(beta=90)

    def rotateAndTranslate(self, alpha=0, beta=0, sigma=0, x=0, y=0, z=0, angunit="deg"):
        self.alphaRotation = alpha
        self.betaRotation = beta
        self.sigmaRotation = sigma
        self.xTranslation = x
        self.yTranslation = y
        self.zTranslation = z
        if angunit == "deg":
            alpha = np.deg2rad(alpha)
            beta = np.deg2rad(beta)
            sigma = np.deg2rad(sigma)
        A = np.array([[np.cos(sigma)*np.cos(beta),
                       -np.sin(sigma)*np.cos(alpha)+np.cos(sigma)*np.sin(beta)*np.sin(alpha),
                       np.sin(sigma)*np.sin(alpha)+np.cos(sigma)*np.sin(beta)*np.cos(alpha),
                       x],

                      [np.sin(sigma) * np.cos(beta),
                       np.cos(sigma) * np.cos(alpha) + np.sin(sigma) * np.sin(beta) * np.sin(alpha),
                       -np.cos(sigma) * np.sin(alpha) + np.sin(sigma) * np.sin(beta) * np.cos(alpha),
                       y],

                      [np.sin(beta),
                       np.cos(beta)*np.sin(alpha),
                       np.cos(beta)*np.cos(alpha),
                       z],

                      [0,
                       0,
                       0,
                       1]], dtype=np.float32)

        B = np.ones(4)

        for ob in self.objects:
            B[0:3] = ob.center
            newCenter = np.array([A[0, 0] * B[0] + A[0, 1] * B[1] + A[0, 2] * B[2]+A[0, 3] * B[3],
                           A[1, 0] * B[0] + A[1, 1] * B[1] + A[1, 2] * B[2] + A[1, 3] * B[3],
                           A[2, 0] * B[0] + A[2, 1] * B[1] + A[2, 2] * B[2] + A[2, 3] * B[3]], dtype=np.float32)

            ob.setCenter(newCenter)

        # B[0:3] = self.limits_min
        # self.limits_min  = np.array([A[0, 0] * B[0] + A[0, 1] * B[1] + A[0, 2] * B[2] + A[0, 3] * B[3],
        #                       A[1, 0] * B[0] + A[1, 1] * B[1] + A[1, 2] * B[2] + A[1, 3] * B[3],
        #                       A[2, 0] * B[0] + A[2, 1] * B[1] + A[2, 2] * B[2] + A[2, 3] * B[3]], dtype=np.float32)
        #
        #
        # B[0:3] = self.limits_max
        # self.limits_max = np.array([A[0, 0] * B[0] + A[0, 1] * B[1] + A[0, 2] * B[2] + A[0, 3] * B[3],
        #                             A[1, 0] * B[0] + A[1, 1] * B[1] + A[1, 2] * B[2] + A[1, 3] * B[3],
        #                             A[2, 0] * B[0] + A[2, 1] * B[1] + A[2, 2] * B[2] + A[2, 3] * B[3]],
        #                            dtype=np.float32)


    def voxelizedAtlasPhantom(self, image_size=None, voxelSize=None):
        """ """
        if image_size is not None:
            max_x = image_size[0] / 2
            min_x = -image_size[0] / 2
            max_y = image_size[1] / 2
            min_y = -image_size[1] / 2
            max_z = image_size[2] / 2
            min_z = -image_size[2] / 2
        else:
            max_x = 20
            min_x = -20
            max_y = 20
            min_y = -20
            max_z = 40
            min_z = -40

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
            # print(ob.density)
            image[((xxx - ob.center[0]) ** 2 + (yyy - ob.center[1]) ** 2 <= ob.rMax ** 2) &
                  ((xxx - ob.center[0]) ** 2 + (yyy - ob.center[1]) ** 2 >= ob.rMin ** 2) &
                  (zzz >= (ob.center[2] - ob.height / 2)) & (zzz <= (ob.center[2] + ob.height / 2))] = ob.density
            # image[((xx - ob.center[0]) ** 2 + (yy - ob.center[1]) ** 2 < ob.rMax ** 2)] = i
            i += 1

        self.voxelizedPhantom = image
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(np.max(image, axis=2), extent=[-max_x, max_x, -max_y, max_y], cmap="gray")
        ax2.imshow(np.max(image, axis=1).T, extent=[-max_x, max_x, -max_z, max_z], cmap="gray")
        ax3.imshow(np.mean(image, axis=0).T, extent=[-max_y, max_y, -max_z, max_z], cmap="gray")

        # gd = GeometryDesigner(volume=image)
        # gd._draw_image_reconstructed()
        # # # plt.imshow(image, extent=[-30, 30, -30, 30])
        # plt.show()

    def voxelizedDensityPhantom(self):
        """"""




if __name__ == "__main__":

    nema = NEMAIQ2008NU()
    nema.voxelizedAtlasPhantom(voxelSize=[0.2, 0.2, 0.2])
