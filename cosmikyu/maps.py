import numpy as np
from pixell import enmap


def get_geometry(ra, dec, width, height, res=None, shape=None):
    if shape is None:
        nx = int(np.ceil(width / res))
        ny = int(np.ceil(height / res))
        shape = (ny, nx)
    else:
        pass
    pos = [[-1. * height / 2. + dec, -1. * width / 2. + ra], [height / 2. + dec, width / 2. + ra]]
    shape, wcs = enmap.geometry(pos=pos, shape=shape)
    return shape, wcs, pos
