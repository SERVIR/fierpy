import numpy as np
from collections.abc import Iterable


class Domain:
    """Class for managing the spatial domain of a model run.
    """

    def __init__(self, bbox: Iterable, resolution: float = 0.25,start_time: str = "1981-01-01",end_time: str = "2020-12-31"):
        if np.any((np.array(bbox) % resolution) != 0):
            self.bbox = _round_out(bbox,resolution)
        else:
            self.bbox = bbox

        self.resoluton = resolution
        self.start_time = start_time
        self.end_time = end_time

        minx, miny, maxx, maxy = self.bbox
        y_coords = np.arange(miny, maxy + self.resolution, self.resolution)[::-1]
        x_coords = np.arange(minx, maxx + self.resolution, self.resolution)

        self.xx, self.yy = np.meshgrid(x_coords,y_coords)

        return


def _round_out(bb: Iterable, res: float):
    """Function to round bounding box to nearest resolution

    args:
        bb (iterable): list of bounding box coordinates in the order of [W,S,E,N]
        res: (float): resolution of pixels to round bounding box to
    """
    minx = bb[0] - (bb[0] % res)
    miny = bb[1] - (bb[1] % res)
    maxx = bb[2] + (res - (bb[2] % res))
    maxy = bb[3] + (res - (bb[3] % res))
    return minx, miny, maxx, maxy
