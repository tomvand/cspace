import numpy as np
from typing import Tuple


class CSpace(object):
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
        self.lut_vertical = self._generate_lut_1d(0)
        self.lut_horizontal = self._generate_lut_1d(1)
        # Note: horizontal and vertical luts could probably be combined with some index shuffling...
        self.lut_disparity = self._generate_disparity_lut()

    def _generate_lut_1d(self, axis):
        lut = []  # index using lut[pixel][disparity] -> (r1x, r2x)
        for i in range(self.image_shape[axis]):
            lut.append([])
            cx = (self.image_shape[axis] - 1) / 2
            u = i - cx
            for d in range(self.disparities):
                try:
                    assert(d > 0)
                    zw = self.focal_length * self.baseline / d  # Eq 1 (minus sign left out)
                    xw = u / self.focal_length * zw  # Eq 2
                    assert(xw != 0)
                    alpha = np.arctan(zw / xw)  # Eq 3  (swapped xw, zw?)
                    arg = self.radius / np.sqrt(zw ** 2 + xw ** 2)  # Eq 4 arcsin argument
                    assert(-1.0 < arg < 1.0)
                    alpha1 = np.arcsin(arg)  # Eq 4
                    r1x = zw / np.tan(alpha + alpha1)  # Eq 5
                    r2x = zw / np.tan(alpha - alpha1)  # Eq 6
                    x1 = np.clip(cx + self.focal_length * r1x / zw, 0, self.image_shape[axis] - 1)
                    x2 = np.clip(cx + self.focal_length * r2x / zw, 0, self.image_shape[axis] - 1)
                except AssertionError:
                    x1, x2 = i, i
                lut[i].append((int(x1), int(x2)))
        return lut

    def _generate_disparity_lut(self):
        lut = []
        for d in range(self.disparities):
            if d == 0:
                dnew = 0
            else:
                zw = self.focal_length * self.baseline / d
                znew = zw - self.radius
                znew = np.clip(znew, 0.1, None)  # Near clip to prevent divide-by-zero errors
                dnew = int(self.focal_length * self.baseline / znew)
            lut.append(dnew)
        return lut

    def filter(self, image):
        image_temp = image.copy()
        image_out = image.copy()
        # Row-wise expansion
        for y in range(self.image_shape[0]):
            for x in range(self.image_shape[1]):
                d = image[y][x]
                x1, x2 = self.lut_horizontal[x][d]
                for x_write in range(x1, x2 + 1):
                    image_temp[y][x_write] = max(image_temp[y][x_write], d)
        # Column-wise expansion
        for x in range(self.image_shape[1]):
            for y in range(self.image_shape[0]):
                d = image_temp[y][x]
                y1, y2 = self.lut_vertical[y][d]
                dnew = self.lut_disparity[d]
                for y_write in range(y1, y2 + 1):
                    image_out[y_write][x] = max(image_out[y_write][x], dnew)
        return image_out
