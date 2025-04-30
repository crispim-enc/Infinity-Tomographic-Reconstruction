import numpy as np


class PoissonMapGenerator:
    def __init__(self, activity=100, activity_units="uCi", acquisition_duration=1, image=np.zeros((83, 83, 128)),
                 reading_data=None):
        self.activity = activity
        if activity_units == "uCi":
            self.activity = self.activity * 37000
        elif activity_units == "mCi":
            self.activity = self.activity * 37000000
        self.acqusition_duration = acquisition_duration
        self.image = image
        self.number_of_voxels = self.image.shape[0] * self.image.shape[1] * self.image.shape[2]
        # top = np.unique(reading_data[:, 5])
        # bot = np.unique(reading_data[:, 4])
        self.poisson_map_cpu()

    def poisson_map_cpu(self):
        mapp = np.zeros(self.number_of_voxels)
        for i in range(self.acqusition_duration):
            mapp += np.random.poisson(np.ceil(self.activity / self.number_of_voxels), self.number_of_voxels)

        self.mapp_im = np.reshape(mapp, self.image.shape)


