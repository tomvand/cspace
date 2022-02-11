# Alternative implementation, see if this is faster...

from functools import lru_cache
from typing import Tuple

import numpy as np


@lru_cache(maxsize=None)
def expand_axis(x, d, width,
                focal_length, baseline, radius
                ):
    cx = (width - 1) / 2
    u = x - cx
    try:
        assert (d > 0)
        zw = focal_length * baseline / d  # Eq 1 (minus sign left out)
        xw = u / focal_length * zw  # Eq 2
        assert (xw != 0)
        alpha = np.arctan(zw / xw)  # Eq 3  (swapped xw, zw?)
        arg = radius / np.sqrt(zw ** 2 + xw ** 2)  # Eq 4 arcsin argument
        assert (-1.0 < arg < 1.0)
        alpha1 = np.arcsin(arg)  # Eq 4
        r1x = zw / np.tan(alpha + alpha1)  # Eq 5
        r2x = zw / np.tan(alpha - alpha1)  # Eq 6
        x1 = np.clip(cx + focal_length * r1x / zw, 0, width - 1)
        x2 = np.clip(cx + focal_length * r2x / zw, 0, width - 1)
    except AssertionError:
        x1, x2 = x, x
    return int(x1), int(x2)


@lru_cache(maxsize=None)
def expand_disparity(d,
                     focal_length, baseline, radius):
    if d == 0:
        dnew = 0
    else:
        zw = focal_length * baseline / d
        znew = zw - radius
        znew = np.clip(znew, 0.1, None)
        dnew = int(focal_length * baseline / znew)
    return dnew


class CSpaceCached(object):
    def __init__(self,
                 image_shape: Tuple[int, int],
                 disparities: int,
                 baseline: float,
                 focal_length: float,
                 radius: float
                 ):
        self.image_shape = image_shape
        self.disparities = disparities
        self.baseline = baseline
        self.focal_length = focal_length
        self.radius = radius
        self._generate_luts()

    def _generate_luts(self):
        # Call cached function for all expected inputs at least once
        w = max(self.image_shape)
        for d in range(self.disparities):
            expand_disparity(d, self.focal_length, self.baseline, self.radius)
            for x in range(w):
                expand_axis(x, d, w, self.focal_length, self.baseline, self.radius)

    def filter(self, image):
        image_temp = image.copy()
        image_out = image.copy()
        # Row-wise expansion
        for y in range(self.image_shape[0]):
            for x in range(self.image_shape[1]):
                d = int(image[y][x])
                x1, x2, = expand_axis(x, d, self.image_shape[1], self.focal_length, self.baseline, self.radius)
                for x_write in range(x1, x2 + 1):
                    image_temp[y][x_write] = max(image_temp[y][x_write], d)
        # Column-wise expansion
        for x in range(self.image_shape[1]):
            for y in range(self.image_shape[0]):
                d = image_temp[y][x]
                y1, y2 = expand_axis(y, d, self.image_shape[0], self.focal_length, self.baseline, self.radius)
                dnew = expand_disparity(d, self.focal_length, self.baseline, self.radius)
                for y_write in range(y1, y2 + 1):
                    image_out[y_write][x] = max(image_out[y_write][x], dnew)
        return image_out
