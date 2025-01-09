import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from src.ImageReader import RawDataSetter


class HandInputSegmentation:
    def __init__(self, filename):
        image_reader = RawDataSetter(filename)
        image_reader.read_files()
        self.volume = image_reader.volume

    @staticmethod
    def tellme(s):
        print(s)
        plt.title(s, fontsize=16)
        plt.draw()

    @staticmethod
    def f(x, y, pts):
        z = np.zeros_like(x)
        for p in pts:
            z = z + 1 / (np.sqrt((x - p[0]) ** 2 + (y - p[1]) ** 2))
        return 1 / z


if __name__ =="__main__":
    # fig, (ax_a, ax_c, ax_s) = plt.subplots(1)
    fig= plt.figure()

    HandInputSegmentation.tellme('You will define a triangle, click to begin')
    image = np.random.randint(0,100,(41,41,68))


    while True:
        pts = []

        HandInputSegmentation.tellme('Select 3 corners with mouse')

        # plt.subplot(1, 3, 1)
        plt.imshow(np.mean(image, axis=2))
        pts = np.asarray(plt.ginput(-1, timeout=-1))
        # plt.subplot(1, 3, 2)
        # plt.imshow(np.mean(image, axis=1))
        # pts_coronal = np.asarray(plt.ginput(-1, timeout=-1))
        # plt.subplot(1, 3, 3)
        # plt.imshow(np.mean(image, axis=2))
        # pts_sagittal = np.asarray(plt.ginput(-1, timeout=-1))
        # plt.waitforbuttonpress()



        # ph = plt.fill(pts[:, 0], pts[:, 1], 'r', lw=2)

        HandInputSegmentation.tellme('Happy? Key click for yes, mouse click for no')
        X, Y = np.meshgrid(np.linspace(0, image.shape[0], image.shape[0]), np.linspace(0,image.shape[1], image.shape[1]))
        # Z = HandInputSegmentation.f(image, image, pts)
        # plt.figure()
        # CS = plt.contour(X, Y, Z, 20)
        grid = matplotlib.path.Path(pts)
        points = np.zeros((X.shape[0]*X.shape[1],2))
        X = X.reshape(X.shape[0] * X.shape[1])
        Y = Y.reshape(Y.shape[0] * Y.shape[1])
        points[:,0] = X
        points[:,1] = Y
        mask = grid.contains_points(points)

        mask = mask.reshape(image.shape[0], image.shape[1])
        mask_3D  = np.tile(mask, (1,1,image.shape[2]))
        plt.imshow(np.mean(mask_3D*image))

        if plt.waitforbuttonpress():
            break



        # Get rid of fill
        # for p in ph:
        #     p.remove()


    # Define a nice function of distance from individual pts



