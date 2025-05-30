import cv2
import numpy as np

import morphsnakes as ms

from eyetracker.morphsnakes_custom import ellipsoid_level_set


# def preprocess_eye_image(image, brightness_modifier):
#     image = image[::10, ::10]
#     brightener = ImageEnhance.Brightness(Image.fromarray(image))
#     return np.array(brightener.enhance(float(brightness_modifier)))


def binarise_with_morphosnakes(img, semi_axis, morph_iteration_number):
    init_ls = ellipsoid_level_set(img.shape, semi_axis=semi_axis)
    m = ms.morphological_chan_vese(img, iterations=morph_iteration_number,
                                   init_level_set=init_ls,
                                   smoothing=1, lambda1=1, lambda2=2)
    return m


def calc_theta_centroid(img, morphosnakes=True):
    mom = cv2.moments(img, binaryImage=morphosnakes)
    theta = 0.5 * np.arctan2(2 * mom["nu11"], (mom["nu20"] - mom["nu02"]))
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    try:
        centroid = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
    except ZeroDivisionError:
        centroid = np.mean(img, axis=0)
        centroid = tuple(centroid.squeeze())
    return theta, centroid
