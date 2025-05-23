import numpy as np


class AlgorithmConvergency:
    def __init__(self, max_iterations=100, tolerance=0.0001):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def __call__(self, image, previous_image):
        if np.sum(np.abs(image - previous_image)) < self.tolerance:
            return True
        if self.max_iterations == 0:
            return True
        self.max_iterations -= 1
        return False