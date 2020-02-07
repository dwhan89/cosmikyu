from pixell import enmap

def get_pos(dec, ra, patch_width, res, shape=None):
    '''
    :param dec: The declination of the center of a patch in radian
    :param ra: The right ascension of the center of a patch in radian
    :param patch_width: The height and the width of a patch in radian
    :param res: The resolution of a patch
    :param shape: optional (Ny, Nx)
    :return: (shape, wcs)
    '''
    y0, x0 = (dec - patch_width / 2, ra - patch_width / 2)
    y1, x1 = (y0 + patch_width, x0 + patch_width)
    return enmap.geometry([[y0,x0],[y1,x1]], res=res, shape=shape)