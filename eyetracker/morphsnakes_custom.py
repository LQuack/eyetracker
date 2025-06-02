"""
Modified circle_level_set function from morphsnakes package (https://pypi.org/project/morphsnakes/)
"""

import numpy as np


def ellipsoid_level_set(image_shape, center=None, semi_axis=None):
    """Create an ellipsoid level set with binary values.

    Parameters
    ----------
    image_shape : tuple of positive integers
        Shape of the image
    center : tuple of integers, optional
        Coordinates of the center of the ellipsoid.
        If not given, it defaults to the center of the image.
    semi_axis : tuple of floats, optional
        Lengths of the semi-axis of the ellispoid.
        If not given, it defaults to the half of the image dimensions.

    Returns
    -------
    out : array with shape `image_shape`
        Binary level set of the ellipsoid with the given `center`
        and `semi_axis`.

    See also
    --------
    circle_level_set
    """

    if center is None:
        center = tuple(i // 2 for i in image_shape)

    if semi_axis is None:
        semi_axis = tuple(i / 2 for i in image_shape)

    if len(center) != len(image_shape):
        raise ValueError("`center` and `image_shape` must have the same length.")

    if len(semi_axis) != len(image_shape):
        raise ValueError("`semi_axis` and `image_shape` must have the same length.")

    if len(image_shape) == 2:
        xc, yc = center
        rx, ry = semi_axis
        phi = 1 - np.fromfunction(
            lambda x, y: ((x - xc) / rx) ** 2 +
                         ((y - yc) / ry) ** 2,
            image_shape, dtype=float)
    elif len(image_shape) == 3:
        xc, yc, zc = center
        rx, ry, rz = semi_axis
        phi = 1 - np.fromfunction(
            lambda x, y, z: ((x - xc) / rx) ** 2 +
                            ((y - yc) / ry) ** 2 +
                            ((z - zc) / rz) ** 2,
            image_shape, dtype=float)
    else:
        raise ValueError("`image_shape` must be a 2- or 3-tuple.")

    res = np.int8(phi > 0)
    return res
