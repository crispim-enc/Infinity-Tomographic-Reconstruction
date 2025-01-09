#  Copyright (c) 2025. # *******************************************************
#  * FILE: $FILENAME
#  * AUTHOR: Pedro Encarnação
#  * DATE: $CURRENT_DATE
#  * LICENSE: CC BY-NC-SA 4.0
#  *******************************************************

import numpy as np
import matplotlib.pyplot as plt
from src.Phantoms import CylindricalStructure


class UltraMicroDerenzo:
    def __init__(self):
        """

        """
        self.partWithRods = CylindricalStructure()
        self.partWithRods.setMaterial("PMMA")
        self.partWithRods.setRMax(30)
        self.partWithRods.setRMin(0)
        self.partWithRods.setHeight(3)
        self.partWithRods.setCenter([0, 0, 0])


        self.firstHoleFirstQuandrant = CylindricalStructure()
        # self.firstHoleFirstQuandrant.setCenter([3.226279, 1.862693, 0])
        self.firstHoleFirstQuandrant.setCenter([-4.2, 8.4, 0])
        # self.firstHoleFirstQuandrant.setCenter([0, 0, 0])
        self.firstHoleFirstQuandrant.setRMax(0.7)
        self.firstQuandrant = Repeater(self.firstHoleFirstQuandrant)
        self.firstQuandrant.setRotation(-30)
        self.firstQuandrant.defineTriangularMatrix()
        self.firstQuandrant.defineQuadrant()
    def voxelizedPhantom(self):
        pass


class Repeater:
    def __init__(self, initialHole):
        self._repeater = "quadrant"
        self._initialHole = initialHole
        self._numberHoles_x = 4
        self._numberHoles_y = None
        self._maxHeight = 15
        self._copySpacing = 2

        self._rotation = 0
        self._distance = None
        self._totalHoles = None
        self._centersQuadrant = []
        self._distanceToCenter = None

    def defineTriangularMatrix(self):
        self._numberHoles_y = int(self._maxHeight // (2 * self._initialHole.rMax + self._copySpacing))
        self._totalHoles = self._numberHoles_x * self._numberHoles_y - self._numberHoles_y
        print("Total holes: {}".format(self._totalHoles))
        print("Number of holes in x: {}".format(self._numberHoles_x))
        print("Number of holes in y: {}".format(self._numberHoles_y))

    def defineQuadrant(self):
        """

        """
        # array_initial = np.array([self._initialHole.center[0], self._initialHole.center[1], self._initialHole.center[2]])
        # array_initial[0:2] = Repeater._rotatePoints(self._rotation, self._initialHole.center[0:2])
        # self._initialHole.setCenter(array_initial)
        adaptation_range = self._numberHoles_y
        for i in range(self._numberHoles_x):
            for j in range(adaptation_range):
                # print("i: {}, j: {}".format(i, j))
                x = i*4*self._initialHole.rMax + self._initialHole.center[0] + j*self._initialHole.rMax*2
                y = -4*j*self._initialHole.rMax + self._initialHole.center[1]
                self._centersQuadrant.append([x, y, self._initialHole.center[2]])

            adaptation_range -= 1
        self._centersQuadrant = np.array(self._centersQuadrant)

        # RP = np.array([self.distanceToCenter * np.cos(self._rotation), self.distanceToCenter * np.sin(self._rotation), np.zeros(len(self._centersQuadrant))])
        self._centersQuadrant[:,0:2] = (Repeater._rotatePoints(self._rotation, self.centersQuadrant[:,0:2])).T
        # self._centersQuadrant = self._centersQuadrant.T

    @staticmethod
    def _rotatePoints(angle, initial_point):
        rotation_matrix = np.array([[np.cos(angle, dtype=np.float32), -np.sin(angle, dtype=np.float32)],
                                    [np.sin(angle, dtype=np.float32), np.cos(angle, dtype=np.float32)]],
                                   dtype=np.float32)

        return np.array([np.cos(angle, dtype=np.float32) * initial_point[:,0] - np.sin(angle, dtype=np.float32) * initial_point[:, 1],
                         np.sin(angle, dtype=np.float32) * initial_point[:,0] + np.cos(angle, dtype=np.float32) * initial_point[:, 1]],
                        dtype=np.float32)

    @property
    def distanceToCenter(self):
        self._distanceToCenter = np.ones(len(self._centersQuadrant))*10
        # self._distanceToCenter = np.sqrt(self._centersQuadrant[:,0]**2 + self._centersQuadrant[:,1]**2)
        return self._distanceToCenter

    def setRotation(self, rotation):
        self._rotation = np.deg2rad(rotation)

    @property
    def centersQuadrant(self):
        return np.array(self._centersQuadrant)


if __name__ == "__main__":
    microDerenzo = UltraMicroDerenzo()
    # plt.plot(microDerenzo.firstQuandrant.centersQuadrant[:,0], microDerenzo.firstQuandrant.centersQuadrant[:,1], "o")
    # plt.show()

    hole = np.array([1, 2, 3, 4, 5, 6])
    d = np.array([1.4, 1.2, 1, 0.9, 0.8, 0.7])# individual hole diameter in mm
    n = np.array([10, 15, 21, 28, 36, 45]) # number of holes
    w = 22.0 # width of the PMMA part with holes
    h = np.cos(np.pi/6) * (2*(n-1)) *d
    i = 0
    for theta in range(0,360,60):
        distanctoToVertex= w/2 - h[i]
        theta = np.deg2rad(theta)
        rotationMatrix = [[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]]
        rotrans = np.dot(rotationMatrix, [distanctoToVertex, 0])
        i += 1
        print("hole {} at {}".format(i, rotrans))

    volume_per_quadrante = np.pi * (d/2)**2 * 3 *n

    print("Volume per quadrante: {}".format(volume_per_quadrante*0.001))
    print("Total volume: {}".format(np.sum(volume_per_quadrante)*0.001))
    activtivit = 18.5/(np.sum(volume_per_quadrante)*0.001)
    print("Activity: {}".format(activtivit))

