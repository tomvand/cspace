# Alternative implementation, see if this is faster...

from functools import lru_cache
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
        pass