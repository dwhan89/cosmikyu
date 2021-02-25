import os

import healpy as hp
import numpy as np
import scipy.interpolate
import torch
from orphics import maps as omaps
from pixell import enmap, utils, curvedsky, powspec
import pylops, cupy as cp
from . import transforms, nn as cnn, model, stats
from multiprocessing import Pool
import gc
import pandas as pd

default_tcmb = 2.726
H_CGS = 6.62608e-27
K_CGS = 1.3806488e-16
C_light = 2.99792e+10

def fnu(nu,tcmb=default_tcmb):
    """
    nu in GHz
    tcmb in Kelvin
    """
    nu = np.asarray(nu)
    mu = H_CGS*(1e9*nu)/(K_CGS*tcmb)
    ans = mu/np.tanh(old_div(mu,2.0)) - 4.0
    return ans

def jysr2thermo(nu, tcmb=default_tcmb):
    nu = np.asarray(nu)*1e9
    mu = H_CGS*(nu)/(K_CGS*tcmb)
    conv_fact = 2*(K_CGS*tcmb)**3/(H_CGS**2*C_light**2)*mu**4/(4*(np.sinh(mu/2.))**2)
    conv_fact *= 1e23
    return 1/conv_fact*tcmb*1e6


def thermo2jysr2(nu, tcmb=default_tcmb):
    return 1/jysr2thermo(nu, tcmb)



class Sehgal10Reprojected(object):
    def __init__(self, input_dir, shape, wcs):
        self.compts = ["kappa", "ksz", "tsz", "ir_pts", "rad_pts"]
        self.highflux_cats = ["rad_pts", "ir_pts"]
        self.geometry = (shape, wcs)
        self.lmax = 10000

        rot_angles1 = [0, 15, 30, 45, 60, 75]
        rot_angles2 = [0, 20, 40, 60, 80]
        self.arrowed_angles = {}
        for rot_angle1 in rot_angles1:
            for rot_angle2 in rot_angles2:
                if rot_angle1 != 0 and rot_angle2 == 0: continue
                self.arrowed_angles[(rot_angle1, rot_angle2)] = None

        self.input_dir = input_dir
        self.processed_dir = os.path.join(input_dir, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)

    def get_compts_idxes(self, trimmed=False):
        return [compt_idx if not trimmed else compt_idx.split("_")[0] for compt_idx in self.compts]

    def get_fits_path(self, dirpath, rot_angle1, rot_angle2, compt_idx, fits_type="alm"):
        freq_idx = "" if compt_idx == "kappa" else "148_"
        file_name = "%s%s_%s_%s_%s_000.fits" % (freq_idx, compt_idx, fits_type , "%0.3d" % rot_angle1, "%0.3d" % rot_angle2)
        return os.path.join(dirpath, file_name)

    def get_highflux_cat_path(self, compt_idx):
        if compt_idx not in self.highflux_cats: return ""
        freq_idx = "" if compt_idx == "kappa" else "148_"
        file_name = "%s%s_highflux_cat.npy" % (freq_idx, compt_idx)
        return os.path.join(self.input_dir, file_name)

    def get_maps(self, rot_angle1, rot_angle2, compts=None, use_sht=True, ret_alm=True, transfer=None, load_processed=False, save_processed=False, flux_cut=None):
        if compts is None: compts = self.compts
        shape, wcs = self.geometry
        nshape = (len(compts),) + shape[-2:]
        ret = enmap.zeros(nshape, wcs)

        if load_processed and not ret_alm: 
            for i, compt_idx in enumerate(compts):
                input_file = self.get_fits_path(self.processed_dir, rot_angle1, rot_angle2, compt_idx)
                print("loading", input_file)
                temp = enmap.read_map(input_file)
                ret[i,...] = enmap.extract(temp, shape, wcs).copy()
                del temp
            return ret
        else:
            for i, compt_idx in enumerate(compts):
                input_file = self.get_fits_path(self.input_dir, rot_angle1, rot_angle2, compt_idx)
                print("loading", input_file)
                alm = np.complex128(hp.read_alm(input_file, hdu=(1)))
                ret[i, ...] = curvedsky.alm2map(alm, enmap.zeros(nshape[1:], wcs))
                del alm
                if compt_idx in self.highflux_cats:
                    print("adding high flux cats")

                    hiflux_cat = np.load(self.get_highflux_cat_path(compt_idx))
                    hiflux_cat[:, :2] = self.__pix2hp(hiflux_cat[:, :2])

                    mat_rot, _, _ = hp.rotator.get_rotation_matrix(
                        (rot_angle1 * utils.degree * -1, rot_angle2 * utils.degree, 0))
                    uvec = hp.ang2vec(hiflux_cat[:, 0], hiflux_cat[:, 1])
                    rot_vec = np.inner(mat_rot, uvec).T
                    temppos = hp.vec2ang(rot_vec)
                    rot_pos = np.zeros(hiflux_cat[:, :2].shape)
                    rot_pos[:, 0] = temppos[0]
                    rot_pos[:, 1] = temppos[1]
                    rot_pos = self.__hp2pix(rot_pos)
                    del temppos
                    rot_pix = np.round(enmap.sky2pix(nshape[-2:], wcs, rot_pos.T).T).astype(np.int)
                    loc = np.where((rot_pix[:, 0] >= 0) & (rot_pix[:, 0] < nshape[-2]) & (rot_pix[:, 1] >= 0.) & (
                            rot_pix[:, 1] < nshape[-1]))
                    hiflux_cat = hiflux_cat[loc[0], 2]
                    rot_pix = rot_pix[loc[0], :]

                    hiflux_map = enmap.zeros(nshape[-2:], wcs)
                    hiflux_map[rot_pix[:, 0], rot_pix[:, 1]] = hiflux_cat
                    if flux_cut is not None:
                        tmin = flux_cut*1e-3*jysr2thermo(148)
                        loc = np.where(hiflux_map>tmin)
                        hiflux_map[loc]=0
                    hiflux_map = hiflux_map / enmap.pixsizemap(shape, wcs)
                    ret[i, ...] = ret[i, ...] + hiflux_map
                    del hiflux_map

        alms = None
        if transfer is not None:
            l, f = transfer
            interp_func = scipy.interpolate.interp1d(l, f, bounds_error=False, fill_value=0.)
            if use_sht:
                l_intp = np.arange(self.lmax + 1)
                f_int = interp_func(l_intp)
                alms = curvedsky.map2alm(ret, lmax=self.lmax, spin=0)
                for i in range(len(compts)):
                    alms[i] = hp.almxfl(alms[i], f_int)
                ret = curvedsky.alm2map(alms, ret, spin=0)
            else:
                ftmap = enmap.fft(ret)
                f_int = interp_func(enmap.modlmap(shape, wcs).ravel())
                ftmap = ftmap * np.reshape(f_int, (shape[-2:]))
                ret = enmap.ifft(ftmap).real;
                del ftmap

        if save_processed:
            raise NotImplemented()

        if ret_alm and alms is None:
            alms = curvedsky.map2alm(ret, lmax=self.lmax, spin=0)
        return ret if not ret_alm else (ret, alms)

    def get_specs(self, use_sht=True, overwrite=False, ret_dl=True, flux_cut=None):

        file_name = "148GHz_sepcs.npz" if flux_cut is None else "148GHz_sepcs_f{}.npz".format(flux_cut)
        file_path = os.path.join(self.input_dir, file_name)
        if os.path.exists(file_path) and not overwrite:
            specs = np.load(file_path)
        else:
            specs = {}
            _, alms = self.get_maps(0, 0, compts=None, use_sht=True, ret_alm=True, flux_cut=flux_cut)

            for i, key1 in enumerate(self.compts):
                for j, key2 in enumerate(self.compts):
                    key1 = key1.split("_")[0]
                    key2 = key2.split("_")[0]
                    key = [key1, key2]
                    key.sort()
                    key = "dls_" + "x".join(key)
                    if key in specs: continue

                    cl = hp.alm2cl(alms[i], alms[j])
                    l = np.arange(len(cl))
                    l_fact = l * (l + 1) / (2 * np.pi)
                    dls = l_fact * cl

                    specs[key] = dls
            specs["l"] = l
            np.savez(file_path, **specs)

        if not ret_dl:
            specs = dict(specs)
            l = specs["l"]
            for key in specs:
                if key == "l": continue
                l_fact = l * (l + 1) / (2 * np.pi)
                specs[key] = np.nan_to_num(specs[key] / l_fact)

        return dict(specs)

    def get_correlation(self, use_sht=True, overwrite=False):
        specs = self.get_specs(use_sht, overwrite)
        corr = {}
        for i, key1 in enumerate(self.compts):
            for j, key2 in enumerate(self.compts):
                key1 = key1.split("_")[0]
                key2 = key2.split("_")[0]
                keys = [key1, key2]
                keys.sort()
                key = "rho_" + "x".join(keys)
                if key in corr: continue

                dl1 = specs["dls_" + "x".join([key1, key1])]
                dl2 = specs["dls_" + "x".join([key2, key2])]
                dlx = specs["dls_" + "x".join([keys[0], keys[1]])]
                rho = dlx / np.sqrt(dl1 * dl2)
                corr[key] = np.nan_to_num(rho)
        corr["l"] = specs["l"].copy()

        return corr

    def __pix2hp(self, pos):
        ret = np.zeros(pos.shape)
        ret[:, 0] = pos[:, 0] + np.pi / 2.
        ret[:, 1] = pos[:, 1] + np.pi
        return ret

    def __hp2pix(self, pos):
        ret = np.zeros(pos.shape)
        ret[:, 0] = pos[:, 0] - np.pi / 2.
        ret[:, 1] = pos[:, 1] - np.pi
        return ret

class Sehgal10ReprojectedFromCat(Sehgal10Reprojected):
    def __init__(self, input_dir, shape, wcs):
        super().__init__(input_dir, shape, wcs)

    def get_maps(self, rot_angle1, rot_angle2, compts=None, use_sht=True, ret_alm=True, transfer=None, load_processed=False, save_processed=False, flux_cut=None):
        if compts is None: compts = self.compts
        shape, wcs = self.geometry
        nshape = (len(compts),) + shape[-2:]
        ret = enmap.zeros(nshape, wcs)

        if load_processed and not ret_alm: 
            for i, compt_idx in enumerate(compts):
                input_file = self.get_fits_path(self.processed_dir, rot_angle1, rot_angle2, compt_idx)
                print("loading", input_file)
                temp = enmap.read_map(input_file)
                ret[i,...] = enmap.extract(temp, shape, wcs).copy()
                del temp
            return ret
        else:
            for i, compt_idx in enumerate(compts):
                if "pts" not in compt_idx:
                    input_file = self.get_fits_path(self.input_dir, rot_angle1, rot_angle2, compt_idx)
                    print("loading", input_file)

                    alm = np.complex128(hp.read_alm(input_file, hdu=(1)))
                    ret[i, ...] = curvedsky.alm2map(alm, enmap.zeros(nshape[1:], wcs))
                else:
                    input_file = self.get_fits_path(self.input_dir, rot_angle1, rot_angle2, compt_idx, fits_type="enmap")
                    print("loading", input_file)
                    temp = enmap.read_map(input_file)
                    ret[i,...] = enmap.extract(temp, shape, wcs).copy()
                    del temp
        alms = None
        if transfer is not None:
            l, f = transfer
            interp_func = scipy.interpolate.interp1d(l, f, bounds_error=False, fill_value=0.)
            if use_sht:
                l_intp = np.arange(self.lmax + 1)
                f_int = interp_func(l_intp)
                alms = curvedsky.map2alm(ret, lmax=self.lmax, spin=0)
                for i in range(len(compts)):
                    alms[i] = hp.almxfl(alms[i], f_int)
                ret = curvedsky.alm2map(alms, ret, spin=0)
            else:
                ftmap = enmap.fft(ret)
                f_int = interp_func(enmap.modlmap(shape, wcs).ravel())
                ftmap = ftmap * np.reshape(f_int, (shape[-2:]))
                ret = enmap.ifft(ftmap).real;
                del ftmap

        if save_processed:
            raise NotImplemented()

        if flux_cut is not None:
            flux_map = flux_cut/enmap.pixsizemap(shape, wcs)
            flux_map *= 1e-3*jysr2thermo(148)
            for i, compt_idx in enumerate(compts):
                if "pts" not in compt_idx: continue
                loc = np.where(ret[i]>flux_map)
                ret[i][loc]=0.
            del flux_map
        
        if ret_alm and alms is None:
            alms = curvedsky.map2alm(ret, lmax=self.lmax, spin=0)
        return ret if not ret_alm else (ret, alms)



class SehgalNetworkFullSky(object):
    def __init__(self,  cuda, ngpu, nbatch, norm_info_file, pixgan_state_file, tuner_state_file,
                 clkk_spec_file, transfer_file, taper_width, nprocess=1, xgrid_file=None, xgrid_inv_file=None, weight_file = None, cache_dir=None):
        ## fixed full sky geometry
        self.shape = (21600, 43200) 
        _ , self.wcs = enmap.fullsky_geometry(res=0.5*utils.arcmin)
        self.template = enmap.zeros(self.shape, self.wcs)
        self.stamp_shape = (5, 128, 128)
        self.nbatch = nbatch 
        self.taper_width = taper_width

        Ny, Nx = self.shape
        ny,nx = self.stamp_shape[-2:]
        num_ybatch = int(np.ceil((Ny-self.taper_width)/(ny-self.taper_width)))
        num_xbatch = int(np.ceil((Nx-self.taper_width)/(nx-self.taper_width)))
        #self.num_batch = num_xbatch*num_ybatch 
        self.num_batch = (num_ybatch, num_xbatch)

        Ny_pad, Nx_pad = num_ybatch*ny, num_xbatch*nx
        self.shape_padded = (Ny_pad, Nx_pad)

        self.lmax = 10000
        self.cache_dir = cache_dir
        if self.cache_dir is None:
            self.cache_dir = os.path.join(os.getcwd(), "cache")
        
        self.nprocess = nprocess
        self.cuda = cuda
        self.ngpu = 0 if not self.cuda else ngpu
        if torch.cuda.is_available() and not cuda:
            print("[WARNING] You have a CUDA device. You probably want to run with CUDA enabled")
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.device = torch.device("cuda" if cuda else "cpu")

        self.norm_info_file = norm_info_file 
        self.normalizer = transforms.SehgalDataNormalizerScaledLogZShrink(self.norm_info_file, channel_idxes=["kappa"])
        self.unnormalizer = transforms.SehgalDataUnnormalizerScaledLogZShrink(self.norm_info_file)

        ## network specific infos
        STanh = cnn.ScaledTanh(15., 2./15.)
        nconv_fc = 64
        kernal_size = 4
        stride = 2
        padding = 1
        output_padding = 0
        dropout_rate = 0

        ## prepare input clkk
        self.clkk_spec = np.load(clkk_spec_file)

        ## transfer
        if transfer_file:
            self.transfer_func = np.load(transfer_file)
        LF = cnn.LinearFeature(4, 4)
        nconv_layer_gen = 4
        nthresh_layer_gen = 3
        self.pixgan_generator = model.UNET_Generator(self.stamp_shape, nconv_layer=nconv_layer_gen, nconv_fc=nconv_fc,
                                                     ngpu=ngpu,
                                                     kernal_size=kernal_size, stride=stride, padding=padding,
                                                     output_padding=output_padding, normalize=True,
                                                     activation=[LF, STanh], nin_channel=1, nout_channel=4,
                                                     nthresh_layer=nthresh_layer_gen, dropout_rate=dropout_rate).to(
            device=self.device)
        print(f"Loading {pixgan_state_file}")
        self.pixgan_generator.load_state_dict(torch.load(pixgan_state_file, map_location=self.device))

        ## tuner layer
        LF = cnn.LinearFeature(5, 5, bias=True)
        nconv_layer_gen = 5
        nthresh_layer_gen = 0
        self.forse_generator = model.VAEGAN_Generator(self.stamp_shape, nconv_layer=nconv_layer_gen, nconv_fc=nconv_fc,
                                                      ngpu=ngpu,
                                                      kernal_size=kernal_size, stride=stride, padding=padding,
                                                      output_padding=output_padding, normalize=True,
                                                      activation=[LF, STanh], nin_channel=5, nout_channel=5,
                                                      nthresh_layer=nthresh_layer_gen, dropout_rate=dropout_rate).to(
            device=self.device)
        print(f"Loading {tuner_state_file}")
        self.forse_generator.load_state_dict(torch.load(tuner_state_file, map_location=self.device))

        self.pixgan_generator.eval()
        self.forse_generator.eval()

        ## load the xgrid later
        self.xgrid_file = xgrid_file
        self.xgrid_inv_file = xgrid_inv_file
        self.xgrid = None
        self.xgrid_inv = None
        self.xscales = None
        ## load weight later
        self.weight_file = weight_file
        self.weight = None

        self.taper = None

    def _get_xgrid(self):
        if self.xgrid is None:
            if self.xgrid_file is not None:
                self.xgrid = np.load(self.xgrid_file)
            else:
                self._generate_grid_info()
        return self.xgrid
    
    def _get_xgrid_inv(self):
        if self.xgrid_inv is None:
            if self.xgrid_inv_file is not None:
                self.xgrid_inv = np.load(self.xgrid_inv_file)
            else:
                self._generate_xgrid_info()
        return self.xgrid_inv
  
    def _get_xscales(self):
        if self.xscales is None:
            pixshapemap = enmap.pixshapemap(self.shape[-2:], self.wcs)
            dxs = pixshapemap[1,:,0]/utils.arcmin
            self.xscales = 0.5/dxs

        return self.xscales
    def _generate_xgrid_info(self):
        shape = self.shape[-2:]
        wcs = self.wcs 
        xgrid = np.zeros(self.shape_padded)  
        xscales = self._get_xscales()
        #loc = np.where(xscales<=1.05)
        num_ybatch = self.num_batch[0]
        num_xbatch = self.num_batch[1]
        taper_width = self.taper_width
        stamp_shape = self.stamp_shape[-2:]
        xwidths = (np.nan_to_num(xscales)*stamp_shape[1])/2
        print("generating x grid")
        for yidx in range(num_ybatch):
            if yidx % 10 == 0:
                print(f"{(yidx)/num_ybatch*100:.2f} perc completed")
            ysidx = yidx*(stamp_shape[0]-taper_width)
            yeidx = ysidx+stamp_shape[0]
            yoffset = yidx*taper_width

            for ycidx in range(ysidx, yeidx):
                yocidx = ycidx+yoffset
                for xidx in range(num_xbatch):
                    xsidx = xidx*(stamp_shape[1]-taper_width)
                    xmidx = xsidx+stamp_shape[1]//2
                    xsidx = int(xmidx - xwidths[ycidx%shape[0]])
                    xeidx = int(xmidx + xwidths[ycidx%shape[0]])
                    

                    xgrid_vald = np.linspace(xsidx, xeidx, stamp_shape[1]) #% shape[1]

                    xosidx = xidx*(stamp_shape[1])
                    xoeidx = xosidx+stamp_shape[1]
                    xgrid[yocidx,xosidx:xoeidx] = xgrid_vald
                    #xgrid_inv[yocidx,xosidx:xoeidx] =  np.argsort(xgrid[yocidx,xosidx:xoeidx]% shape[1]) 

        self.xgrid = xgrid.astype(np.float32)
        
 
        posmap = enmap.posmap(shape, wcs) 
        dec = posmap[0].flatten()
        ra = posmap[1].flatten()
        dec += np.pi/2
        dec %= np.pi
        ra += np.pi
        ra %= (2*np.pi)
        xgrid_inv = hp.pixelfunc.ang2pix(8192, dec, ra); del posmap, dec, ra
        xgrid_inv = xgrid_inv.reshape(shape)
    
        self.xgrid_inv = xgrid_inv.astype(np.int32)

    def _get_taper(self):
        if self.taper is None:
            ny, nx = self.stamp_shape[-2:]
            taper = omaps.cosine_window(ny, nx, self.taper_width, self.taper_width)
            minval = np.min(taper[taper != 0])
            taper[taper == 0] = minval
            self.taper = taper
        return self.taper.copy()
    def _get_weight(self, overwrite=False):
        if self.weight is None:
            if self.weight_file is not None and not overwrite:
                self.weight = np.load(self.weight_file)
            else:
                Ny, Nx = self.shape
                ny, nx = self.stamp_shape[-2:]
                taper_width = self.taper_width
                nbatchy, nbatchx = self.num_batch
                batch_idxes = np.array_split(np.arange(nbatchy), min(nbatchy,self.nprocess)) 
                stamp = self._get_taper()
                xgrid = self._get_xgrid()
                
                self.weight = np.zeros((2, Ny, Nx), dtype=np.float32)
                for i, method in enumerate(["interp", "nearest"]):
                    global _generate_weight_core
                    def _generate_weight_core(batch_idxes, method=method):
                        retysidx = batch_idxes[0]*(ny-taper_width)
                        retyeidx = min(batch_idxes[-1]*(ny-taper_width)+ny,Ny)
                        ret = np.zeros((retyeidx-retysidx, Nx))
                        for yidx in batch_idxes:
                            ysidx = yidx*(ny-taper_width)
                            yeidx = min(ysidx+ny,Ny)
                            yoffset = taper_width*yidx
                            yosidx = ysidx*ny
                            yoeidx = yosidx+ny
                            for xidx in range(nbatchx):
                                xosidx = xidx*nx
                                xoeidx = xosidx+nx
                                for j, ycidx in enumerate(np.arange(ysidx, yeidx)):
                                    if ycidx >= Ny:
                                        continue
                                    yrcidx = ycidx-retysidx 
                                    yocidx = ycidx+yoffset  
                                    xvals = xgrid[yocidx,xosidx:xoeidx] 
                                    xmin = int(np.ceil(xvals[0]))
                                    xmax = int(np.floor(xvals[-1]))
                                    xin = np.arange(xmin,xmax+1)
                                    xin = np.arange(xmin,xmax+1)[:Nx] 
                                    
                                    if method == "nearest":
                                        #xin = np.round(xvals).astype(np.int)
                                        #yvals = np.ones(len(xvals))
                                        yvals = stamp[j,:].copy()
                                        fit = scipy.interpolate.interp1d(xvals, yvals, assume_sorted=True, kind="nearest")
                                        #ret[yrcidx,xin%Nx] += yvals
                                    elif method == "interp": 
                                        yvals = stamp[j,:].copy() 
                                        fit = scipy.interpolate.interp1d(xvals, yvals, assume_sorted=True)
                                    ret[yrcidx,xin%Nx] += fit(xin); del fit
                        return ((retysidx,retyeidx), ret)
                    with Pool(len(batch_idxes)) as p:
                        storage = p.map(_generate_weight_core, batch_idxes)
                    del _generate_weight_core
                    for idxes, ring in storage:
                        self.weight[i, idxes[0]:idxes[1],:] += ring
                    del storage

                loc = np.where(self.weight[1] == 0)
                self.weight[1][loc] = np.inf; del loc
        return self.weight
                        

    def _generate_input_kappa(self, seed=None, conv_beam=False):
        np.random.seed(seed)
        clkk = self.clkk_spec[:,1]
        alm = curvedsky.rand_alm(clkk)
        return curvedsky.alm2map(alm, self.template.copy())[np.newaxis, ...]

    def generate_samples(self, seed=None, verbose=True, input_kappa=None, transfer=True,  post_processes=[], use_cache=True, flux_cut=7, polfix=True):
        if input_kappa is None:
            if verbose: print("making input gaussian kappa")
            gkmap = self._generate_input_kappa(seed=seed)
        else:
            if input_kappa.ndim == 2: input_kappa = input_kappa[np.newaxis,...]
            gkmap = input_kappa
        if use_cache: 
            os.makedirs(self.cache_dir, exist_ok=True)

        nch, Ny, Nx = gkmap.shape
        Ny_pad, Nx_pad = self.shape_padded
        wcs = gkmap.wcs
        ny, nx = self.stamp_shape[-2:]
        taper_width = self.taper_width
        nbatchy, nbatchx = self.num_batch 
        xgrid = self._get_xgrid()
        taper = self._get_taper() 
       	batch_idxes = np.array_split(np.arange(nbatchy), min(nbatchy,self.nprocess))	
        
        xin = np.arange(Nx+1)
        if verbose: print("start sampling")
        global _get_sampled
        def _get_sampled(batch_idxes):
            retysidx = batch_idxes[0]*(ny)
            retyeidx = (batch_idxes[-1]+1)*(ny)
            ret = np.zeros((retyeidx-retysidx, Nx_pad))
            for i, yidx in enumerate(batch_idxes):
                ysidx = yidx*(ny-taper_width)
                yeidx = min(ysidx+ny, Ny)
                yoffset = yidx*taper_width
		
                for ycidx in np.arange(ysidx, yeidx):
                    if ycidx >= Ny:
                        continue
                    yocidx = ycidx+yoffset
                    yrcidx = yocidx-retysidx
                    yvals = np.append(gkmap[0,ycidx,:],gkmap[0,ycidx,0])
                    fit =  scipy.interpolate.CubicSpline(xin,yvals, bc_type="periodic")
                    xval = xgrid[yocidx,:]% Nx

                    ret[yrcidx,:] = fit(xval)
            return ret
        with Pool(len(batch_idxes)) as p:
            sampled = p.map(_get_sampled, batch_idxes)
        sampled = np.vstack(sampled)
        sampled = sampled[np.newaxis,...]
        if verbose: print("end sampling")
        del _get_sampled, xin

        if polfix:
            if verbose: print("pol fix")
            sampled[:,:1*ny,:] = np.flip(sampled[:,1*ny:2*ny,:],1)
            sampled[:,-1*ny:,:] = np.flip(sampled[:,-2*ny:-1*ny,:],1)

        del gkmap; gkmap = sampled
        gkmap = self.normalizer(gkmap)
        def process_ml(input_imgs, batch_maker):
            input_imgs = batch_maker(input_imgs)
            nsample = input_imgs.shape[0]
            output_imgs = np.zeros((nsample, 5, ny, nx))
            ctr = 0
            nitr = int(np.ceil(input_imgs.shape[0]/ self.nbatch))
            for batch in np.array_split(np.arange(input_imgs.shape[0]), nitr):
                input_tensor = torch.autograd.Variable(self.Tensor(input_imgs[batch].copy()))
                ret = self.pixgan_generator(input_tensor).detach()
                ret = torch.cat((input_tensor, ret), 1)
                ret = self.forse_generator(ret).detach()
                output_imgs[batch] = ret.data.to(device="cpu").numpy()
                if verbose and ctr%20 == 0:
                    print(f"batch {ctr}/{nitr} completed")
                ctr += 1
            return output_imgs

        def post_process(output_imgs,  unbatch_maker):
            output_imgs = unbatch_maker(output_imgs)
            output_imgs = output_imgs[0,...]
            return output_imgs

        if verbose: print("make the primary images")
        batch_maker = transforms.Batch((nch, ny, nx))
        unbatch_maker = transforms.UnBatch((nch, Ny_pad, Nx_pad))

        processed = process_ml(gkmap, batch_maker); del gkmap
        processed = post_process(processed, unbatch_maker)

        for post_process in post_processes:
            processed = post_process(processed)

        processed = self.unnormalizer(processed)

        loc = np.where(processed[3:5] < 0)
        processed[3:5][loc] = 0.; del loc
        pixarea = (0.5*utils.arcmin)**2
        thermo2mjy = 1/jysr2thermo(148)*pixarea*1e3
        #processed[3] *= thermo2mjy
        #processed[3] *= 1.18
        #processed[4] *= 1.1
        #loc = np.where(processed[3]>2)
        #processed[3][loc] = processed[3][loc]**0.75; del loc 
        #loc = np.where(processed[3] > 7)
        #processed[3][loc] = 0.; del loc
        #processed[3] /= thermo2mjy
        '''
        for i in range(3,5):
            loc = np.where(processed[i] < 0)
            processed[i][loc] = 0.; del loc
        '''
        reprojected = np.zeros((5,Ny,Nx), dtype=np.float32)
        for compt_idx in range(0,5):
            if verbose: print(f"reprojecting images {compt_idx}")
            method = "interp" if compt_idx < 4 else "nearest"
            global _get_reprojected
            def _get_reprojected(batch_idxes, method=method):
                retysidx = batch_idxes[0]*(ny-taper_width)
                retyeidx = min(batch_idxes[-1]*(ny-taper_width)+ny,Ny)
                ret = np.zeros((retyeidx-retysidx, Nx))

                for yidx in batch_idxes:
                    ysidx = yidx*(ny-taper_width)
                    yeidx = min(ysidx+ny,Ny)
                    yoffset = taper_width*yidx
                    for xidx in range(nbatchx):
                        xosidx = xidx*nx
                        xoeidx = xosidx+nx
                        for j, ycidx in enumerate(np.arange(ysidx, yeidx)):
                            if ycidx >= Ny:
                                continue
                            yrcidx = ycidx-retysidx
                            yocidx = ycidx+yoffset
                            yvals = processed[compt_idx,yocidx,xosidx:xoeidx]
                            xvals = xgrid[yocidx,xosidx:xoeidx]

                            
                            xmin = int(np.ceil(xvals[0]))
                            xmax = int(np.floor(xvals[-1]))
                            xin = np.arange(xmin,xmax+1)
                            xin = np.arange(xmin,xmax+1)[:Nx]
                            spread_fact = ny/max(xmax-xmin,ny) if compt_idx > 5 else 1.
                            if method == "nearest":
                                xin = np.round(xvals).astype(np.int)
                                yvals = yvals*taper[j,:]*spread_fact
                                #yvals = yvals*spread_fact
                                #ret[yrcidx,xin%Nx] += yvals
                                fit = scipy.interpolate.interp1d(xvals, yvals, assume_sorted=True, kind="nearest")
                            elif method == "interp":
                                yvals = yvals*taper[j,:]
                                fit = scipy.interpolate.interp1d(xvals, yvals, assume_sorted=True)
                            else:
                                assert(False)
                            ret[yrcidx,xin%Nx] += fit(xin)
                return ((retysidx,retyeidx), ret)

            with Pool(len(batch_idxes)) as p:
                storage = p.map(_get_reprojected, batch_idxes)
            for idxes, ring in storage:
                reprojected[compt_idx, idxes[0]:idxes[1],:] += ring
            del storage, _get_reprojected
        ## weight correction for diffused

        reprojected[:4] = reprojected[:4]/self._get_weight()[0] 
        reprojected[4] = reprojected[4]/self._get_weight()[1]
        return reprojected

        ## process point sources
        jysr2uk = jysr2thermo(148)
        decs, _ = enmap.posaxes((Ny,Nx), wcs)
        ptbatchy = nbatchy+1

        ptbatch_idxes = np.array_split(np.arange(ptbatchy), min(ptbatchy,self.nprocess))
        mlsims_pts = processed[3:,...].copy(); del processed
        mlsims_catalogs = {}
        for i, compt_idx in enumerate(["ir", "rad"]):
            if verbose:
                print(f"making pt source cats {compt_idx}")
            gc.collect()
            global _make_catalog
            def _make_catalog(batch_idxes, compt_idx = i):
                yosidx = batch_idxes[0]*int(ny)-int(0.5*ny)
                yoeidx = (batch_idxes[-1]+1)*int(ny)-int(0.5*ny)
                yosidx = max(yosidx,0)
                
                
                loc = np.where(mlsims_pts[compt_idx,yosidx:yoeidx] >= 0)
                cat = np.zeros((len(loc[0]),3))
                yoidxes = (loc[0].copy()+yosidx)
                a = yoidxes // ny 
                b = yoidxes % ny 

                cat[:,0] = a*(ny-taper_width)+b; del a,b
                cat[:,1] = xgrid[yosidx:yoeidx][loc].copy() 
                cat[:,1] %= Nx
                cat[:,2] = mlsims_pts[compt_idx,yosidx:yoeidx][loc].copy()
                cat[:,2] *= (1/jysr2uk*1e3*(0.5*utils.arcmin)**2)
               
                del loc
                if cat[-1,0] >= Ny:
                    rloc = np.where(cat[:,0]<Ny)
                    cat = cat[rloc[0],:]; del rloc                

                cat = pd.DataFrame(cat, columns=['dec','ra', 'I'])
                cat = cat.sort_values(['dec', 'ra'], ascending=[True, True])
                cat.reset_index(drop=True, inplace=True)
                pddec_idxes = cat.groupby("dec").indices
                
                merged = []
                for i, dec_idx in enumerate(pddec_idxes):
                    dec = decs[int(dec_idx)]
                    scale = (0.5*utils.arcmin)/max(np.cos(dec), (0.5*utils.arcmin)/(2*np.pi))
                    nsegment = int(np.round((np.ceil(2*np.pi/scale))))
                    cat_sub = cat.iloc[pddec_idxes[dec_idx]].to_numpy()
                    
                    RB = stats.FastBINNER(0,Nx,nsegment)
                    _, ra_bincount = RB.bin(cat_sub[:,1])
                    ra_bincount = np.int64(ra_bincount)
                    
                    delta_ra = Nx//nsegment
                    ras_resd = Nx%nsegment
                    shift_idxes = sorted(np.random.choice(nsegment, size=ras_resd, replace=False))
                    shifts = np.zeros(nsegment, dtype=np.int)
                    shifts[shift_idxes] = +1
                    ras = np.arange(nsegment)*delta_ra
                    ras += np.random.randint(delta_ra, size=nsegment)
                    ras += np.cumsum(shifts)
                    
                    del delta_ra, ras_resd, shifts, RB

                    ctr = 0
                    reduced = []
                    for j in range(nsegment):
                        ccount = ra_bincount[j]
                        if ccount == 0:
                            continue

                        row = np.zeros(3)
                        row[1] = ras[j]
                        if ccount == 1:
                            row[2] = cat_sub[ctr,2]
                        else:
                            row[2] = np.random.choice(cat_sub[ctr:ctr+ccount,2], 1)  
                        if row[2] != 0: 
                            reduced.append(row)
                        ctr += ccount
                    cat_sub = np.array(reduced).reshape((len(reduced),3))
                    cat_sub[:,0] = dec_idx
                    del ra_bincount, reduced, ras
                    
                    merged.append(cat_sub)
                del cat
                merged = np.vstack(merged)
                
                ## flux correction
                merged[:,2] *= 1.1

                if compt_idx == 0:
                    loc =  np.where(merged[:,2] > 1)
                    merged[loc[0],2] = merged[loc[0],2]**0.55
                    del loc
                
                if flux_cut is not None:
                    loc = np.where(merged[:,2]<=flux_cut)
                    merged = merged[loc[0],...]
                    del loc

                return np.float32(merged)


            with Pool(len(batch_idxes)) as p:
                storage = p.map(_make_catalog, ptbatch_idxes)
            del _make_catalog
            mlsims_catalogs[compt_idx] = np.vstack(storage)
            
        del decs, ptbatch_idxes,
        ## catalog to map 
        gc.collect()
        pixsizemap = enmap.pixsizemap((Ny, Nx),wcs)
        conv = ((1e-3*jysr2thermo(148))/pixsizemap); del pixsizemap
        for i, compt_idx in enumerate(["ir", "rad"]):
            cat = mlsims_catalogs[compt_idx] 
            reprojected[([i+3],np.int64(cat[:,0]), np.int64(cat[:,1]))]= cat[:,2]
            reprojected[i+3] = reprojected[i+3]*conv

        #del mlsims_catalogs

        return reprojected#, mlsims_catalogs


        alms = None

        

        def fft_taper(lcrit, taper_width=10000, order=5):
            ell = np.arange(lcrit+taper_width+1)
            f = np.ones(len(ell))
            f[lcrit:lcrit+taper_width+1] = np.cos(np.linspace(0,np.pi/2,taper_width+1))**order
            f[-1] = 0
            return ell, f
        
        fft_taper_width = 20000#int(modlmap.max())-self.lmax
        l, f_taper = fft_taper(self.lmax, taper_width=fft_taper_width, order=10)
        f_transf = f_taper 
        if deconv_beam: 
            if verbose: print("deconvolving beam")
            l, f_beam = self._get_beam(ell=l)
            f_beam[-1*fft_taper_width:] = f_beam[-1*fft_taper_width-1]
            f_beam = 1/f_beam
            f_transf = f_transf*f_beam
        f_transfs = [None]*5
        for i in range(5):
            f_transfs[i] = f_transf.copy()

        if transfer:
            if verbose: print("applying trasfer function")
            for i in range(5):
                f_transfs[i][:self.lmax+1] = f_transfs[i][:self.lmax+1]*self.transfer_func[:self.lmax+1,i+1]

        def _load_combined_tf_funcs(ell, use_cache, cache_dir):
            lmax = int(np.ceil(np.max(ell)))
            target_file = os.path.join(cache_dir, f"tf_beam_{str(deconv_beam)}_transfer_{str(transfer)}_lmax_{lmax}_sht_{str(use_sht)}.npy")
            if os.path.exists(target_file) and use_cache:
                storage = np.load(target_file)
                print(f"loading {target_file}")
            else:
                interp_funcs = _gen_interp_funcs()
                storage = np.zeros((len(ell), 6))
                storage[:,0] = ell.copy()
                for i in range(5):
                    storage[:,i+1] = interp_funcs[i](ell)
                if use_cache:
                    np.save(target_file, storage)
                    print(f"saving {target_file}")
            return storage
            



        def _gen_interp_funcs():
            interp_funcs = [None]*5
            for i in range(5):
                interp_funcs[i] =  scipy.interpolate.interp1d(l, f_transfs[i], bounds_error=False, fill_value=(f_transfs[i][0],f_transfs[i][-1]))
            return interp_funcs

        torch.cuda.empty_cache() 

        
        
        if deconv_beam or transfer:
            taper = omaps.cosine_window(Ny, Nx, 20, 20)
            minval = np.min(taper[taper!= 0])
            taper[taper==0] = minval
            if use_sht:
                raise NotImplemented()
                l_intp = np.arange(self.lmax + 1)
                f_int = interp_func(l_intp)
                alms = curvedsky.map2alm(output_imgs, lmax=self.lmax, spin=0)
                for i in range(4):
                    alms[i] = hp.almxfl(alms[i], f_int)
                output_imgs = curvedsky.alm2map(alms, ret, spin=0)
                del alms
            else:
                maxidx = 3
                ftmap = enmap.fft(output_imgs[:maxidx,...])
                modlmap = enmap.modlmap((Ny, Nx) , wcs).ravel()
                combined_tfs = _load_combined_tf_funcs(modlmap, use_cache, self.cache_dir)
                for i in range(maxidx):
                    ftmap[i] = ftmap[i] * np.reshape(combined_tfs[:, i+1], (Ny, Nx))
                del modlmap
                output_imgs[:maxidx,...] = enmap.ifft(ftmap).real;
                del ftmap
        if transfer:
            conv = enmap.pixsizemap(self.shape, self.wcs)/jysr2thermo(148)*1e3

            output_imgs[3] *= 1.1

            target = output_imgs[3]*conv
            lowflux = target.copy()
            cut = 1
            loc = np.where(lowflux>cut)
            lowflux[loc] = 0.
            highflux = target-lowflux
            highflux = highflux**0.61
            #loc = np.where(highflux<cut)
            #highflux[loc] = 0
            target = lowflux+highflux
            output_imgs[3] = target/conv
            del lowflux, highflux
            output_imgs[4] *= 1.6284

        if flux_cut is not None:
            flux_map = flux_cut/enmap.pixsizemap(self.shape, self.wcs)
            flux_map *= 1e-3*jysr2thermo(148)
            for i in range(3,5):
                loc = np.where(output_imgs[i]>flux_map)
                output_imgs[i][loc]=0.
            del flux_cut
     
        for i in range(3,5):
            loc = np.where(output_imgs[i]<0)
            output_imgs[i][loc]=0.

        return output_imgs  if not ret_corr else (output_imgs, corr)
