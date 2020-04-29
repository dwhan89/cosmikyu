import healpy as hp
import numpy as np
import os
from past.utils import old_div
from pixell import enmap
import torch
from torch.utils.data import Dataset


default_tcmb = 2.726
H_CGS = 6.62608e-27
K_CGS = 1.3806488e-16
C_light = 2.99792e+10

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
    

def get_template(shape, wcs):
    return enmap.zeros(shape, wcs)

def fnu(nu,tcmb=default_tcmb):
    """
    nu in GHz
    tcmb in Kelvin
    """
    nu = np.asarray(nu)
    mu = H_CGS*(1e9*nu)/(K_CGS*tcmb)
    ans = mu/np.tanh(old_div(mu,2.0)) - 4.0
    return ans

class StampedSky(object):
    def __init__(self, path, res_arcmin, shape):
        self.path = path
        self.res_arcmin = res_arcmin
        self.res_arcmin_str = "%0.1f" % res_arcmin
        self.shape = shape
        self.size = np.product(shape)
        self.frequencies = [str(s).zfill(3) for s in [30,90,148,219,277,350]]
        self.rfunc = enmap.read_fits 

    def __format_fname(self, fname_temp, sim_idx, freq=None):
        sim_idx_str = str(sim_idx).zfill(6)
        if freq is not None:
            freq = str(freq).zfill(3)
            fname = os.path.join(self.path, fname_temp.format(freq, sim_idx_str, self.res_arcmin_str, self.shape[0], self.shape[1]))
        else:
            fname = os.path.join(self.path, fname_temp.format(sim_idx_str, self.res_arcmin_str, self.size, self.shape[0], self.shape[1]))
        return fname


    def get_total_cmb(self, freq_idx, sim_idx, filename_only=False):
        raise NotImplemented()
    
    def get_lensed_cmb(self, freq_idx, sim_idx, filename_only=False):
        raise NotImplemented()

    def get_kappa(self,filename_only=False):
        raise NotImplemented()
   
    def get_galactic_dust(self,freq,filename_only=False):
        raise NotImplemented()

    def get_ksz(self, freq, sim_idx, filename_only=False):
        filename = self.__format_fname('{}_ksz_{}_{}arcmin_{}x{}.fits', sim_idx=sim_idx, freq=freq)
        return filename if filename_only else self.rfunc(filename)

    def get_compton_y(self, sim_idx, tcmb=default_tcmb, scale=0.75):
        return self.get_tsz(148, sim_idx, scale=scale)/(fnu(148)*tcmb * 1e6)

    def get_cib(self, freq, sim_idx, filename_only=False, scale=0.75):
        filename = self.__format_fname('{}_ir_pts_{}_{}arcmin_{}x{}.fits', sim_idx=sim_idx, freq=freq)
        return filename if filename_only else self.rfunc(filename)*scale

    def get_radio(self, freq, sim_idx, filename_only=False):
        filename = self.__format_fname('{}_rad_pts_{}_{}arcmin_{}x{}.fits', sim_idx=sim_idx, freq=freq)
        return filename if filename_only else self.rfunc(filename)

    def get_tsz(self, freq, sim_idx, filename_only=False, scale=0.75):
        filename = self.__format_fname('{}_tsz_{}_{}arcmin_{}x{}.fits', sim_idx=sim_idx, freq=freq)
        return filename if filename_only else self.rfunc(filename)*scale

class StampedSkyDataSet(Dataset):
    def __init__(self, root_dir, num_sim, res_arcmin, fg_type, freq, shape, transform=None):
        self.stamped_sky = StampedSky(root_dir, res_arcmin, shape)
        self.freq = freq
        self.fg_rfuncs = {
            'ir_pts': self.stamped_sky.get_cib,
            'ksz': self.stamped_sky.get_ksz,
            'rad_pts': self.stamped_sky.get_radio,
            'tsz': self.stamped_sky.get_tsz   
        }
        self.fg_type = fg_type
        self.rfunc = self.fg_rfuncs[self.fg_type]    
        self.transform = transform
        self.num_sim = num_sim
        
    
    def __len__(self):
        return self.num_sim
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        fg_map = self.rfunc(self.freq, idx, False)
        sample = {'data': np.array(fg_map), 'wcs': fg_map.wcs.copy()}
        if self.transform:
            sample = self.transform(sample)
         
        return sample

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
            self.rfunc = lambda x : hp.read_alm(x, hdu=1).astype("complex128")
        self.fg_loaders = {
            'ir_pts': self.get_cib,
            'ksz': self.get_ksz,
            'rad_pts': self.get_radio,
            'tsz': self.get_tsz
            }


    def __format_fname(self, fname_temp, freq=None, euler=None):
        post_fix = "" #if euler is None else "_%03d_%03d_%03d" %(euler[0], euler[1], euler[2])
        if freq is not None:
            freq = str(freq).zfill(3)
            fname = os.path.join(self.path, fname_temp.format(freq, self.data_type, post_fix))
        else:
            fname = os.path.join(self.path, fname_temp.format(self.data_type, post_fix))
        return fname

    def get_total_cmb(self,freq,filename_only=False):
        filename = self.__format_fname('{}_skymap_{}{}.fits', freq=freq)
        return filename if filename_only else self.rfunc(filename)

    def get_lensed_cmb(self,freq, filename_only=False):
        filename = self.__format_fname('{}_lensedcmb_{}{}.fits', freq=freq)
        return filename if filename_only else self.rfunc(filename)

    def get_ksz(self, freq, filename_only=False, euler=(0,0,0)):
        filename = self.__format_fname('{}_ksz_{}{}.fits', freq=freq, euler=euler)
        return filename if filename_only else self.rfunc(filename)

    def get_kappa(self,filename_only=False):
        raise NotImplemented("soon")

    def get_compton_y(self, tcmb=default_tcmb, scale=0.75, euler=(0,0,0)):
        return self.get_tsz(148, scale=scale, euler=euler)/(fnu(148)*tcmb * 1e6)

    def get_cib(self,freq,filename_only=False, scale=0.75, euler=(0,0,0)): 
        filename = self.__format_fname('{}_ir_pts_{}{}.fits', freq=freq, euler=euler)
        return filename if filename_only else self.rfunc(filename)*scale

    def get_radio(self,freq,filename_only=False, euler=(0,0,0)):
        filename = self.__format_fname('{}_rad_pts_{}{}.fits', freq=freq, euler=euler)
        return filename if filename_only else self.rfunc(filename)

    def get_galactic_dust(self,freq,filename_only=False): 
        filename = self.__format_fname('{}_dust_{}{}.fits', freq=freq)
        return filename if filename_only else self.rfunc(filename)

    def get_tsz(self,freq,filename_only=False, scale=0.75, euler=(0,0,0)):
        filename = self.__format_fname('{}_tsz_{}{}.fits', freq=freq, euler=euler)
        return filename if filename_only else self.rfunc(filename)*scale
