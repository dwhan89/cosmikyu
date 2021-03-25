import gc
import os
from multiprocessing import Pool

import healpy as hp
import numpy as np
import scipy.interpolate
import torch
from orphics import maps as omaps
from past.utils import old_div
from pixell import enmap, utils, curvedsky, powspec

from . import transforms, nn as cnn, model
from .utils import car2hp_coords, hp2car_coords, load_data

DEFAULT_TCMB = 2.726
H_CGS = 6.62608e-27
K_CGS = 1.3806488e-16
C_light = 2.99792e+10


class _SeedTracker(object):
    def __init__(self):
        self.CMB = 0
        self.FG = 1
        self.KAPPA = 2
        self.freq_dict = {30, 90, 148, 219, 277, 350}

    def get_cmb_seed(self, sim_idx):
        return self.CMB, sim_idx

    def get_fg_seed(self, sim_idx, freq_idx):
        assert (freq_idx in self.freq_dict)
        return self.FG, sim_idx, freq_idx

    def get_kappa_seed(self, sim_idx):
        return self.KAPPA, sim_idx


seed_tracker = _SeedTracker()


def fnu(nu, tcmb=DEFAULT_TCMB):
    """
    nu in GHz
    tcmb in Kelvin
    """
    nu = np.asarray(nu)
    mu = H_CGS * (1e9 * nu) / (K_CGS * tcmb)
    ans = mu / np.tanh(old_div(mu, 2.0)) - 4.0
    return ans


def jysr2thermo(nu, tcmb=DEFAULT_TCMB):
    nu = np.asarray(nu) * 1e9
    mu = H_CGS * (nu) / (K_CGS * tcmb)
    conv_fact = 2 * (K_CGS * tcmb) ** 3 / (H_CGS ** 2 * C_light ** 2) * mu ** 4 / (4 * (np.sinh(mu / 2.)) ** 2)
    conv_fact *= 1e23
    return 1 / conv_fact * tcmb * 1e6


def thermo2jysr2(nu, tcmb=DEFAULT_TCMB):
    return 1 / jysr2thermo(nu, tcmb)


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
        file_name = "%s%s_%s_%s_%s_000.fits" % (
            freq_idx, compt_idx, fits_type, "%0.3d" % rot_angle1, "%0.3d" % rot_angle2)
        return os.path.join(dirpath, file_name)

    def get_highflux_cat_path(self, compt_idx):
        if compt_idx not in self.highflux_cats: return ""
        freq_idx = "" if compt_idx == "kappa" else "148_"
        file_name = "%s%s_highflux_cat.npy" % (freq_idx, compt_idx)
        return os.path.join(self.input_dir, file_name)

    def get_maps(self, rot_angle1, rot_angle2, compts=None, use_sht=True, ret_alm=True, transfer=None,
                 load_processed=False, save_processed=False, flux_cut=None):
        if compts is None: compts = self.compts
        shape, wcs = self.geometry
        nshape = (len(compts),) + shape[-2:]
        ret = enmap.zeros(nshape, wcs)

        if load_processed and not ret_alm:
            for i, compt_idx in enumerate(compts):
                input_file = self.get_fits_path(self.processed_dir, rot_angle1, rot_angle2, compt_idx)
                print("loading", input_file)
                temp = enmap.read_map(input_file)
                ret[i, ...] = enmap.extract(temp, shape, wcs).copy()
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
                    hiflux_cat[:, :2] = car2hp_coords(hiflux_cat[:, :2])

                    mat_rot, _, _ = hp.rotator.get_rotation_matrix(
                        (rot_angle1 * utils.degree * -1, rot_angle2 * utils.degree, 0))
                    uvec = hp.ang2vec(hiflux_cat[:, 0], hiflux_cat[:, 1])
                    rot_vec = np.inner(mat_rot, uvec).T
                    temppos = hp.vec2ang(rot_vec)
                    rot_pos = np.zeros(hiflux_cat[:, :2].shape)
                    rot_pos[:, 0] = temppos[0]
                    rot_pos[:, 1] = temppos[1]
                    rot_pos = hp2car_coords(rot_pos)
                    del temppos
                    rot_pix = np.round(enmap.sky2pix(nshape[-2:], wcs, rot_pos.T).T).astype(np.int)
                    loc = np.where((rot_pix[:, 0] >= 0) & (rot_pix[:, 0] < nshape[-2]) & (rot_pix[:, 1] >= 0.) & (
                            rot_pix[:, 1] < nshape[-1]))
                    hiflux_cat = hiflux_cat[loc[0], 2]
                    rot_pix = rot_pix[loc[0], :]

                    hiflux_map = enmap.zeros(nshape[-2:], wcs)
                    hiflux_map[rot_pix[:, 0], rot_pix[:, 1]] = hiflux_cat
                    if flux_cut is not None:
                        tmin = flux_cut * 1e-3 * jysr2thermo(148)
                        loc = np.where(hiflux_map > tmin)
                        hiflux_map[loc] = 0
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


class Sehgal10ReprojectedFromCat(Sehgal10Reprojected):
    def __init__(self, input_dir, shape, wcs):
        super().__init__(input_dir, shape, wcs)

    def get_maps(self, rot_angle1, rot_angle2, compts=None, use_sht=True, ret_alm=True, transfer=None,
                 load_processed=False, save_processed=False, flux_cut=None):
        if compts is None: compts = self.compts
        shape, wcs = self.geometry
        nshape = (len(compts),) + shape[-2:]
        ret = enmap.zeros(nshape, wcs)

        if load_processed and not ret_alm:
            for i, compt_idx in enumerate(compts):
                input_file = self.get_fits_path(self.processed_dir, rot_angle1, rot_angle2, compt_idx)
                print("loading", input_file)
                temp = enmap.read_map(input_file)
                ret[i, ...] = enmap.extract(temp, shape, wcs).copy()
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
                    input_file = self.get_fits_path(self.input_dir, rot_angle1, rot_angle2, compt_idx,
                                                    fits_type="enmap")
                    print("loading", input_file)
                    temp = enmap.read_map(input_file)
                    ret[i, ...] = enmap.extract(temp, shape, wcs).copy()
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
            flux_map = flux_cut / enmap.pixsizemap(shape, wcs)
            flux_map *= 1e-3 * jysr2thermo(148)
            for i, compt_idx in enumerate(compts):
                if "pts" not in compt_idx: continue
                loc = np.where(ret[i] > flux_map)
                ret[i][loc] = 0.
            del flux_map

        if ret_alm and alms is None:
            alms = curvedsky.map2alm(ret, lmax=self.lmax, spin=0)
        return ret if not ret_alm else (ret, alms)


class SehgalNetworkFullSky(object):
    def __init__(self, cuda, ngpu, nbatch, norm_info_file, pixgan_state_file, tuner_state_file,
                 clkk_spec_file, cmb_spec_file, transfer_1dspec_file, transfer_2dspec_file, taper_width, nprocess=1, xgrid_file=None,
                 weight_file=None, output_dir=None):
        ## fixed full sky geometry
        self.shape = (21600, 43200)
        _, self.wcs = enmap.fullsky_geometry(res=0.5 * utils.arcmin)
        self.template = enmap.zeros(self.shape, self.wcs)
        self.stamp_shape = (5, 128, 128)
        self.nbatch = nbatch
        self.taper_width = taper_width
        self.fg_compts = ["kappa", "ksz", "tsz", "ir_pts", "rad_pts"]

        Ny, Nx = self.shape
        ny, nx = self.stamp_shape[-2:]
        num_ybatch = int(np.ceil((Ny - self.taper_width) / (ny - self.taper_width)))
        num_xbatch = int(np.ceil((Nx - self.taper_width) / (nx - self.taper_width)))
        self.num_batch = (num_ybatch, num_xbatch)

        Ny_pad, Nx_pad = num_ybatch * ny, num_xbatch * nx
        self.shape_padded = (Ny_pad, Nx_pad)

        self.lmax = 10000
        self.output_dir = output_dir
        if self.output_dir is None:
            self.output_dir = os.path.join(os.getcwd(), "output")

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
        STanh = cnn.ScaledTanh(15., 2. / 15.)
        nconv_fc = 64
        kernal_size = 4
        stride = 2
        padding = 1
        output_padding = 0
        dropout_rate = 0

        ## prepare input specs
        self.clkk_spec = np.load(clkk_spec_file)
        self.cmb_spec = powspec.read_camb_scalar(cmb_spec_file)

        ## transfer
        self.transf_1dspec = np.load(transfer_1dspec_file)
        self.transf_2dspec = load_data(transfer_2dspec_file)

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
        self.xgrid = None
        self.xscales = None
        ## load weight later
        self.weight_file = weight_file
        self.weight = None

        self.taper = None
        self.jysr2thermo = None

    def _get_xgrid(self):
        if self.xgrid is None:
            if self.xgrid_file is not None:
                self.xgrid = np.load(self.xgrid_file)
            else:
                self._generate_grid_info()
        return self.xgrid

    def _get_xscales(self):
        if self.xscales is None:
            pixshapemap = enmap.pixshapemap(self.shape[-2:], self.wcs)
            dxs = pixshapemap[1, :, 0] / utils.arcmin
            self.xscales = 0.5 / dxs

        return self.xscales

    def _generate_xgrid_info(self):
        shape = self.shape[-2:]
        wcs = self.wcs
        xgrid = np.zeros(self.shape_padded)
        xscales = self._get_xscales()
        num_ybatch = self.num_batch[0]
        num_xbatch = self.num_batch[1]
        taper_width = self.taper_width
        stamp_shape = self.stamp_shape[-2:]
        xwidths = (np.nan_to_num(xscales) * stamp_shape[1]) / 2
        print("generating x grid")
        for yidx in range(num_ybatch):
            if yidx % 10 == 0:
                print(f"{(yidx) / num_ybatch * 100:.2f} perc completed")
            ysidx = yidx * (stamp_shape[0] - taper_width)
            yeidx = ysidx + stamp_shape[0]
            yoffset = yidx * taper_width

            for ycidx in range(ysidx, yeidx):
                yocidx = ycidx + yoffset
                for xidx in range(num_xbatch):
                    xmidx = xsidx + stamp_shape[1] // 2
                    xsidx = int(xmidx - xwidths[ycidx % shape[0]])
                    xeidx = int(xmidx + xwidths[ycidx % shape[0]])

                    xgrid_vald = np.linspace(xsidx, xeidx, stamp_shape[1])  # % shape[1]

                    xosidx = xidx * (stamp_shape[1])
                    xoeidx = xosidx + stamp_shape[1]
                    xgrid[yocidx, xosidx:xoeidx] = xgrid_vald

        self.xgrid = xgrid.astype(np.float32)

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
                batch_idxes = np.array_split(np.arange(nbatchy), min(nbatchy, self.nprocess))
                stamp = self._get_taper()
                xgrid = self._get_xgrid()

                self.weight = np.zeros((2, Ny, Nx), dtype=np.float32)
                for i, method in enumerate(["interp", "nearest"]):
                    global _generate_weight_core

                    def _generate_weight_core(batch_idxes, method=method):
                        retysidx = batch_idxes[0] * (ny - taper_width)
                        retyeidx = min(batch_idxes[-1] * (ny - taper_width) + ny, Ny)
                        ret = np.zeros((retyeidx - retysidx, Nx))
                        for yidx in batch_idxes:
                            ysidx = yidx * (ny - taper_width)
                            yeidx = min(ysidx + ny, Ny)
                            yoffset = taper_width * yidx
                            yosidx = ysidx * ny
                            for xidx in range(nbatchx):
                                xosidx = xidx * nx
                                xoeidx = xosidx + nx
                                for j, ycidx in enumerate(np.arange(ysidx, yeidx)):
                                    if ycidx >= Ny:
                                        continue
                                    yrcidx = ycidx - retysidx
                                    yocidx = ycidx + yoffset
                                    xvals = xgrid[yocidx, xosidx:xoeidx]
                                    xmin = int(np.ceil(xvals[0]))
                                    xmax = int(np.floor(xvals[-1]))
                                    xin = np.arange(xmin, xmax + 1)[:Nx]
                                    yvals = stamp[j, :].copy()
                                    if method == "nearest":
                                        fit = scipy.interpolate.interp1d(xvals, yvals, assume_sorted=True,
                                                                         kind="nearest")
                                    elif method == "interp":
                                        fit = scipy.interpolate.interp1d(xvals, yvals, assume_sorted=True)
                                    ret[yrcidx, xin % Nx] += fit(xin);
                                    del fit
                        return (retysidx, retyeidx), ret

                    with Pool(len(batch_idxes)) as p:
                        storage = p.map(_generate_weight_core, batch_idxes)
                    del _generate_weight_core
                    for idxes, ring in storage:
                        self.weight[i, idxes[0]:idxes[1], :] += ring
                    del storage

                loc = np.where(self.weight[1] == 0)
                self.weight[1][loc] = np.inf
                del loc
        return self.weight

    def _generate_gaussian_kappa(self, seed=None):
        if seed is not None:
            np.random.seed(seed_tracker.get_kappa_seed(seed))
        clkk = self.clkk_spec[:, 1]
        alm = curvedsky.rand_alm(clkk)
        return curvedsky.alm2map(alm, self.template.copy())[np.newaxis, ...]

    def generate_unlensed_cmb(self, seed=None):
        if seed is not None:
            np.random.seed(seed_tracker.get_cmb_seed(seed))


    def _get_jysr2thermo(self, mode="car"):
        assert (mode == "car")
        if self.jysr2thermo is None:
            pixsizemap = enmap.pixsizemap(self.shape, self.wcs)
            self.jysr2thermo = (1e-3 * jysr2thermo(148) / pixsizemap);
            del pixsizemap
            self.jysr2thermo = self.jysr2thermo.astype(np.float32)
        return self.jysr2thermo

    def get_output_file_name(self, compt_idx, sim_idx, freq=None, polidx=None):
        if compt_idx in ["tsz", "rad_pts", "ir_pts"]:
            assert(freq in seed_tracker.freq_dict)
            output_file = os.path.join(self.output_dir, f"{compt_idx}_{freq:03d}ghz_{sim_idx:05d}.fits")
        elif compt_idx in ["kappa", "ksz"]:
            output_file = os.path.join(self.output_dir, f"{compt_idx}_{sim_idx:05d}.fits")
        elif compt_idx in ["lensed_cmb", "combined"]:
            assert(freq in seed_tracker.freq_dict)
            assert(polidx in ["T,Q,U"])
            output_file = os.path.join(self.output_dir, f"{compt_idx}_{polidx}_{freq:03d}ghz_{sim_idx:05d}.fits")
        else:
            raise NotImplemented()

        return output_file

    def get_all_foregrounds(self, seed=None, freq=148, verbose=True, input_kappa=None, post_processes=[],
                            save_output=True, flux_cut=7, polfix=True, dtype=np.float32):
        if save_output:
            os.makedirs(self.output_dir, exist_ok=True)
        try:
            assert(seed is not None)
            if verbose:
                print(f"trying to load saved foregrounds. sim idx: {seed}, freq: {freq}GHz")
            fgmaps = enmap.empty((5,)+self.shape, self.wcs, dtype=dtype)
            for i, compt_idx in enumerate(self.fg_compts):
                fname = self.get_output_file_name(compt_idx, seed, freq=freq)
                fgmaps[i] = enmap.read_fits(self.get_output_file_name(fname))
        except:

            if freq == 148:
                fgmaps = self._generate_foreground_148GHz(seed=seed, verbose=verbose, input_kappa=input_kappa,
                                                      post_processes=post_processes, save_output=save_output,
                                                      flux_cut=flux_cut, polfix=polfix)
            else:
                raise NotImplemented()
                fgmaps = self.get_all_foregrounds(seed=seed, freq=freq, verbose=verbose, input_kappa=input_kappa,
                                                  post_processes=post_processes, save_output=save_output, flux_cut=flux_cut,
                                                  polfix=polfix, dtype=dtype)

        return fgmaps


    def _generate_foreground_148GHz(self, seed=None, verbose=True, input_kappa=None, post_processes=[],
                            flux_cut=7, polfix=True):
        if input_kappa is None:
            if verbose: print("making input gaussian kappa")
            gaussian_kappa = self._generate_gaussian_kappa(seed=seed)
        else:
            if input_kappa.ndim == 2: input_kappa = input_kappa[np.newaxis, ...]
            gaussian_kappa = input_kappa

        Ny, Nx = self.shape
        Ny_pad, Nx_pad = self.shape_padded
        ny, nx = self.stamp_shape[-2:]
        taper_width = self.taper_width
        nbatchy, nbatchx = self.num_batch
        xgrid = self._get_xgrid()
        taper = self._get_taper()
        batch_idxes = np.array_split(np.arange(nbatchy), min(nbatchy, self.nprocess))

        xin = np.arange(Nx + 1)
        if verbose: print("start sampling")
        global _get_sampled

        def _get_sampled(batch_idxes):
            retysidx = batch_idxes[0] * (ny)
            retyeidx = (batch_idxes[-1] + 1) * (ny)
            ret = np.zeros((retyeidx - retysidx, Nx_pad))
            for i, yidx in enumerate(batch_idxes):
                ysidx = yidx * (ny - taper_width)
                yeidx = min(ysidx + ny, Ny)
                yoffset = yidx * taper_width

                for ycidx in np.arange(ysidx, yeidx):
                    if ycidx >= Ny:
                        continue
                    yocidx = ycidx + yoffset
                    yrcidx = yocidx - retysidx
                    yvals = np.append(gaussian_kappa[0, ycidx, :], gaussian_kappa[0, ycidx, 0])
                    fit = scipy.interpolate.CubicSpline(xin, yvals, bc_type="periodic")
                    xval = xgrid[yocidx, :] % Nx

                    ret[yrcidx, :] = fit(xval)
            return ret

        with Pool(len(batch_idxes)) as p:
            gaussian_kappa = p.map(_get_sampled, batch_idxes)
        del _get_sampled, xin
        gc.collect()

        gaussian_kappa = np.vstack(gaussian_kappa)
        gaussian_kappa = gaussian_kappa[np.newaxis, ...]
        if verbose: print("end sampling")

        if polfix:
            if verbose: print("pol fix")
            gaussian_kappa[:, :1 * ny, :] = np.flip(gaussian_kappa[:, 1 * ny:2 * ny, :], 1)
            gaussian_kappa[:, -1 * ny:, :] = np.flip(gaussian_kappa[:, -2 * ny:-1 * ny, :], 1)

        gaussian_kappa = self.normalizer(gaussian_kappa)

        def process_ml(input_imgs, batch_maker):
            input_imgs = batch_maker(input_imgs)
            nsample = input_imgs.shape[0]
            output_imgs = np.zeros((nsample, 5, ny, nx))
            ctr = 0
            nitr = int(np.ceil(input_imgs.shape[0] / self.nbatch))
            for batch in np.array_split(np.arange(input_imgs.shape[0]), nitr):
                input_tensor = torch.autograd.Variable(self.Tensor(input_imgs[batch].copy()))
                ret = self.pixgan_generator(input_tensor).detach()
                ret = torch.cat((input_tensor, ret), 1)
                ret = self.forse_generator(ret).detach()
                output_imgs[batch] = ret.data.to(device="cpu").numpy()
                if verbose and ctr % 20 == 0:
                    print(f"batch {ctr}/{nitr} completed")
                ctr += 1
            return output_imgs

        def post_process(output_imgs, unbatch_maker):
            output_imgs = unbatch_maker(output_imgs)
            output_imgs = output_imgs[0, ...]
            return output_imgs

        if verbose: print("make the primary images")
        batch_maker = transforms.Batch((1, ny, nx))
        unbatch_maker = transforms.UnBatch((1, Ny_pad, Nx_pad))

        processed = process_ml(gaussian_kappa, batch_maker);
        del gaussian_kappa
        processed = post_process(processed, unbatch_maker)
        del batch_maker, unbatch_maker
        torch.cuda.empty_cache()
        gc.collect()

        for post_process in post_processes:
            processed = post_process(processed)
        processed = self.unnormalizer(processed)

        loc = np.where(processed[3:5] < 0)
        processed[3:5][loc] = 0.
        reprojected = np.zeros((5, Ny, Nx), dtype=np.float32)
        for compt_idx in range(0, 5):
            if verbose: print(f"reprojecting images {compt_idx}")
            method = "interp" if compt_idx < 3 else "nearest"
            global _get_reprojected

            def _get_reprojected(batch_idxes, method=method):
                retysidx = batch_idxes[0] * (ny - taper_width)
                retyeidx = min(batch_idxes[-1] * (ny - taper_width) + ny, Ny)
                ret = np.zeros((retyeidx - retysidx, Nx))

                for yidx in batch_idxes:
                    ysidx = yidx * (ny - taper_width)
                    yeidx = min(ysidx + ny, Ny)
                    yoffset = taper_width * yidx
                    for xidx in range(nbatchx):
                        xosidx = xidx * nx
                        xoeidx = xosidx + nx
                        for j, ycidx in enumerate(np.arange(ysidx, yeidx)):
                            if ycidx >= Ny:
                                continue
                            yrcidx = ycidx - retysidx
                            yocidx = ycidx + yoffset
                            yvals = processed[compt_idx, yocidx, xosidx:xoeidx]
                            xvals = xgrid[yocidx, xosidx:xoeidx]

                            xmin = int(np.ceil(xvals[0]))
                            xmax = int(np.floor(xvals[-1]))
                            xin = np.arange(xmin, xmax + 1)[:Nx]
                            yvals = yvals * taper[j, :]
                            if method == "nearest":
                                fit = scipy.interpolate.interp1d(xvals, yvals, assume_sorted=True, kind="nearest")
                            elif method == "interp":
                                fit = scipy.interpolate.interp1d(xvals, yvals, assume_sorted=True)
                            else:
                                assert (False)
                            ret[yrcidx, xin % Nx] += fit(xin)
                return ((retysidx, retyeidx), ret)

            with Pool(len(batch_idxes)) as p:
                storage = p.map(_get_reprojected, batch_idxes)
            for idxes, ring in storage:
                reprojected[compt_idx, idxes[0]:idxes[1], :] += ring
            del storage, _get_reprojected
        gc.collect()

        ## weight correction for diff
        reprojected[:3] = reprojected[:3] / self._get_weight()[0]
        reprojected[3:5] = reprojected[3:5] / self._get_weight()[1]
        reprojected = enmap.enmap(reprojected.astype(np.float32), wcs=self.wcs)

        ## apply transfer functions
        kmap = enmap.fft(reprojected[:3])
        for j, compt_idx in enumerate(self.compts[:3]):
            if verbose: print(f"applying the transfer functions to {compt_idx}")
            xtransf = self.transf_2dspec[compt_idx]['px']
            ytransf = self.transf_2dspec[compt_idx]['py']
            kmap[j] = kmap[j] * np.outer(ytransf, xtransf)
            reprojected[j] = enmap.ifft(kmap[j]).real
            alm = curvedsky.map2alm(reprojected[j].astype(np.float64), lmax=10000)
            alm = hp.almxfl(alm, self.transf_1dspec[compt_idx])
            reprojected[j] = curvedsky.alm2map(alm, reprojected[j])
        del kmap

        def boxcox(arr, lamb):
            return ((arr + 1) ** lamb - 1) / lamb

        reprojected[3:5] *= 1 / self._get_jysr2thermo(mode="car")
        reprojected[3] *= 1.1
        loc = np.where(reprojected[3] > 1)
        reprojected[3][loc] = reprojected[3][loc] ** 0.63;
        del loc
        reprojected[4] = boxcox(reprojected[4], 1.25)
        loc = np.where(reprojected[3:5] > flux_cut)
        reprojected[3:5][loc] = 0.;
        del loc
        reprojected[3:5] *= self._get_jysr2thermo(mode="car")
        gc.collect()

        return reprojected.astype(np.float32)
