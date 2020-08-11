import healpy as hp
import numpy as np
import os
from past.utils import old_div
from pixell import enmap
import torch
from torch.utils.data import Dataset



def rand_geometry(width, height, res=None, shape=None, seed=None):
    np.random.seed(seed)
    dec = np.random.uniform(-1*np.pi/2, np.pi/2)/9.
    ra = np.random.uniform(-1*np.pi, np.pi)
    return get_geometry(ra, dec, width, height, res=res, shape=shape)
    

def get_geometry(ra, dec, width, height, res=None, shape=None):
    if shape is None:
        nx = int(np.ceil(width/res))
        ny = int(np.ceil(height/res))
        shape = (ny, nx)
    else:pass
    pos = [[-1.*height/2.+dec, -1.*width/2.+ra], [height/2.+dec, width/2.+ra]]
    shape, wcs = enmap.geometry(pos=pos, shape=shape)
    return (shape, wcs, pos)
    

