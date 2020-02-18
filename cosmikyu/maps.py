import healpy as hp
import numpy as np
import os
from past.utils import old_div
from pixell import enmap

default_tcmb = 2.726
H_CGS = 6.62608e-27
K_CGS = 1.3806488e-16
C_light = 2.99792e+10

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


def fnu(nu,tcmb=default_tcmb):
    """
    nu in GHz
    tcmb in Kelvin
    """
    nu = np.asarray(nu)
    mu = H_CGS*(1e9*nu)/(K_CGS*tcmb)
    ans = mu/np.tanh(old_div(mu,2.0)) - 4.0
    return ans


class SehgalSky2010(object):
    # to handle orignal Sehgal et al sims
    def __init__(self,path=None, data_type="healpix"):
        if path is None: path = os.environ['SEHGAL_SKY']
        assert(data_type in ['healpix', 'enmap', 'alm'])
        self.data_type = data_type
        self.path        = path
        self.frequencies = [str(s).zfill(3) for s in [30,90,148,219,277,350]]
        if self.data_type == 'healpix' :
            self.rfunc = hp.read_map
            self.nside = 8192
        elif self.data_type == 'enmap':
            self.rfunc = enmap.read_map
            self.shape,self.wcs = enmap.read_map_geometry(self.get_lensed_cmb(True))
        elif self.data_type == 'alm':
            self.rfunc = lambda x : hp.read_alm(x, hdu=1)

    def __format_fname(self, fname_temp, freq=None):
        if freq is not None:
            freq = str(freq).zfill(3)
            fname = os.path.join(self.path, fname_temp.format(freq, self.data_type))
        else:
            fname = os.path.join(self.path, fname_temp.format(self.data_type))
        return fname

    def get_total_cmb(self,freq,filename_only=False):
        filename = self.__format_fname('{}_skymap_{}.fits', freq=freq)
        return filename if filename_only else self.rfunc(filename)

    def get_lensed_cmb(self,freq, filename_only=False):
        filename = self.__format_fname('{}_lensedcmb_{}.fits', freq=freq)
        return filename if filename_only else self.rfunc(filename)

    def get_ksz(self, freq, filename_only=False):
        filename = self.__format_fname('{}_ksz_{}.fits', freq=freq)
        return filename if filename_only else self.rfunc(filename)

    def get_kappa(self,filename_only=False):
        raise NotImplemented("soon")

    def get_compton_y(self, tcmb=default_tcmb, scale=0.75):
        return self.get_tsz(148, scale)/(fnu(148)*tcmb * 1e6)

    def get_cib(self,freq,filename_only=False, scale=0.75):
        filename = self.__format_fname('{}_ir_pts_{}.fits', freq=freq)
        return filename if filename_only else self.rfunc(filename)*scale

    def get_radio(self,freq,filename_only=False):
        filename = self.__format_fname('{}_rad_pts_{}.fits', freq=freq)
        return filename if filename_only else self.rfunc(filename)

    def get_galactic_dust(self,freq,filename_only=False):
        filename = self.__format_fname('{}_dust_{}.fits', freq=freq)
        return filename if filename_only else self.rfunc(filename)

    def get_tsz(self,freq,filename_only=False, scale=0.75):
        filename = self.__format_fname('{}_tsz_{}.fits', freq=freq)
        return filename if filename_only else self.rfunc(filename)*scale
