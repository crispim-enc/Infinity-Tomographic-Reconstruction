import skimage


class ImageMetrics(object):
    def __init__(self, ground_truth=None, image_test=None):
        if ground_truth is None:
            return
        if image_test is None:
            return

        self.ground_truth = ground_truth
        self.image_test = image_test

    def mean_square_error(self):
        """
        The mean-squared error (MSE) is computed by comparing the pixel-by-pixel difference between the input image and the reference image.
        The MSE represents the mean of the squared differences between the two images.
        """
        return skimage.metrics.mean_squared_error(self.ground_truth, self.image_test)

    def peak_signal_noise_ratio(self):
        """ """
        return skimage.metrics.peak_signal_noise_ratio(self.ground_truth, self.image_test)

    def structural_similarity(self):
        """ """
        return skimage.metrics.structural_similarity(self.ground_truth, self.image_test)

    def normalized_root_mean_square_error(self):
        """ """
        return skimage.metrics.normalized_root_mse(self.ground_truth, self.image_test)

    def mean_absolute_error(self):
        """ """
        return skimage.metrics.mean_absolute_error(self.ground_truth, self.image_test)

    def mean_absolute_percentage_error(self):
        """ """
        return skimage.metrics.mean_absolute_percentage_error(self.ground_truth, self.image_test)

    def compare_images(self):
        """ """
        return skimage.metrics.compare_images(self.ground_truth, self.image_test)

    def compare_ssim(self):
        """ """
        return skimage.metrics.compare_ssim(self.ground_truth, self.image_test)
