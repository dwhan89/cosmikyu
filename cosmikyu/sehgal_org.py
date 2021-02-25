import os

import healpy as hp
import numpy as np
import scipy.interpolate
import torch
from orphics import maps as omaps
from pixell import enmap, utils, curvedsky, powspec

from . import transforms, nn as cnn, model

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

    def get_fits_path(self, dirpath, rot_angle1, rot_angle2, compt_idx):
        freq_idx = "" if compt_idx == "kappa" else "148_"
        file_name = "%s%s_%s_%s_%s_000.fits" % (freq_idx, compt_idx, "alm", "%0.3d" % rot_angle1, "%0.3d" % rot_angle2)
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


class SehgalNetwork(object):
    def __init__(self, shape, wcs, cuda, ngpu, nbatch, norm_info_file, pixgan_state_file, tuner_state_file,
                 clkk_spec_file, transfer_file, taper_width, cache_dir=None):
        self.shape = shape[-2:]
        self.wcs = wcs
        self.template = enmap.zeros(shape, wcs)
        self.stamp_shape = (5, 128, 128)
        self.nbatch = nbatch
        self.lmax = 10000
        self.taper_width = taper_width
        self.cache_dir = cache_dir
        if self.cache_dir is None:
            self.cache_dir = os.path.join(os.getcwd(), "cache")

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
        STanh = cnn.ScaledTanh(30., 2. / 30.)
        nconv_fc = 64
        kernal_size = 4
        stride = 2
        padding = 1
        output_padding = 0
        dropout_rate = 0

        ## prepare input clkk
        #ps_scalar = powspec.read_camb_scalar(clkk_spec_file)
        #clpp = ps_scalar[1][0][0][:self.lmax + 1]
        #L = np.arange(self.lmax + 1)
        #clkk = clpp * (L * (L + 1)) ** 2 / 4
        #self.clkk_spec = (L,clkk)
        self.clkk_spec = np.load(clkk_spec_file)

        ## transfer 
        self.transfer_func = np.load(transfer_file)
        #self.transfer_func[:2,1:] = 1.
        ## pixgan layer
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

    def _get_beam(self, beam_fwhm=0.9, ell=None):
        beam_fwhm = np.deg2rad(beam_fwhm / 60.)
        sigma = beam_fwhm / (2. * np.sqrt(2. * np.log(2)))
        ell = ell if ell is not None else np.arange(self.lmax + 1)
        f_ell = np.exp(-(ell) ** 2. * sigma ** 2. / 2)
        return ell, f_ell

    def _generate_input_kappa(self, seed=None, conv_beam=True):
        np.random.seed(seed)
        clkk = self.clkk_spec[:,1]
        #clkk = self.clkk_spec[1]
        alm = curvedsky.rand_alm(clkk)
        if conv_beam: alm = hp.almxfl(alm, self._get_beam()[1])
        return curvedsky.alm2map(alm, self.template.copy())[np.newaxis, ...]

    def generate_samples(self, seed=None, ret_corr=False, wrap=True, wrap_mode=("reflect","wrap"), edge_blend=True, verbose=True, input_kappa=None, transfer=True, deconv_beam=True, use_sht=True, post_processes=[], use_cache=True, flux_cut=None):
        if input_kappa is None:
            if verbose: print("making input gaussian kappa")
            gkmap = self.normalizer(self._generate_input_kappa(seed=seed))
        else:
            if input_kappa.ndim == 2: input_kappa = input_kappa[np.newaxis,...]
            gkmap = self.normalizer(input_kappa.copy())
        if use_cache: 
            os.makedirs(self.cache_dir, exist_ok=True)

        nch, Ny, Nx = gkmap.shape
        wcs = gkmap.wcs
        ny, nx = self.stamp_shape[-2:]
        padded_shape = (np.ceil([Ny / ny, Nx / nx]) * np.array([ny, nx])).astype(np.int)
        Ny_pad, Nx_pad = padded_shape[0], padded_shape[1]

        if wrap:
            gkmap = np.pad(gkmap, ((0, 0), (0, Ny_pad - Ny), (0, 0)), mode=wrap_mode[0])
            gkmap = np.pad(gkmap, ((0, 0), (0, 0), (0, Nx_pad - Nx)), mode=wrap_mode[1])
        def process_ml(input_imgs, ret_corr, batch_maker):
            input_imgs = batch_maker(input_imgs)
            nsample = input_imgs.shape[0]
            output_imgs = np.zeros((nsample, 5, ny, nx))
            if ret_corr:
                corr = output_imgs.copy()
            ctr = 0
            nitr = int(np.ceil(input_imgs.shape[0]/ self.nbatch))
            for batch in np.array_split(np.arange(input_imgs.shape[0]), nitr):
                input_tensor = torch.autograd.Variable(self.Tensor(input_imgs[batch].copy()))
                ret = self.pixgan_generator(input_tensor).detach()
                ret = torch.cat((input_tensor, ret), 1)
                if ret_corr: corr[batch] = ret.data.to(device="cpu").numpy()
                ret = self.forse_generator(ret).detach()
                output_imgs[batch] = ret.data.to(device="cpu").numpy() 
                if verbose and ctr%20 == 0: 
                    print(f"batch {ctr}/{nitr} completed")
                ctr += 1

            return (output_imgs, None) if not ret_corr else (output_imgs, corr)

        unbatch_maker = transforms.UnBatch((nch, Ny_pad, Nx_pad))
        def post_process(output_imgs, corr, ret_corr, unbatch_maker):
            output_imgs = unbatch_maker(output_imgs)
            output_imgs = output_imgs[0, ..., :Ny, :Nx]
            if ret_corr:
                corr = unbatch_maker(corr)
                corr = corr[0, ..., :Ny, :Nx]
                corr = output_imgs - corr
            return output_imgs, corr

        if verbose: print("make the primary images")
        batch_maker = transforms.Batch((nch, ny, nx))
        output_imgs, corr = process_ml(gkmap, ret_corr, batch_maker)
        output_imgs, corr = post_process(output_imgs, corr, ret_corr, unbatch_maker)
        if edge_blend:
            if verbose: print("make the blending images")

            ## inner part
            def get_taper(input_imgs, ywidth, xwidth, batch_maker, unbatch_maker):
                ## taper can be cahsed ...
                taper = batch_maker(input_imgs * 0)
                taper_stamp = omaps.cosine_window(ny, nx, ywidth, xwidth)
                minval = np.min(taper_stamp[taper_stamp != 0])
                taper_stamp[taper_stamp == 0] = minval
                taper[..., :, :] = taper_stamp
                taper = unbatch_maker(taper)[0]
                return taper[..., :Ny, :Nx]

            def get_taper_imgs(input_imgs, shifts):
                y_shift, x_shift = shifts
                if y_shift > 0:
                    taper_imgs = input_imgs[..., y_shift:-1 * y_shift, :].copy()
                    unbatch_maker = transforms.UnBatch((nch, Ny_pad - 2 * y_shift, Nx_pad))
                else:
                    taper_imgs = input_imgs[..., :, x_shift:-1 * x_shift].copy()
                    unbatch_maker = transforms.UnBatch((nch, Ny_pad, Nx_pad - 2 * x_shift))
                taper_imgs, _ = process_ml(taper_imgs, True, batch_maker=batch_maker)
                taper_imgs, _ = post_process(taper_imgs, None, ret_corr=False, unbatch_maker=unbatch_maker)
                return taper_imgs

            if verbose: print("starting vertical blending")
            taper = get_taper(gkmap, self.taper_width, 0, batch_maker, unbatch_maker)
            output_imgs[..., 64:Ny_pad - 64, :] = (output_imgs * taper)[..., 64:Ny_pad - 64, :]
            output_imgs[..., 64:Ny_pad - 64, :] += (
                    get_taper_imgs(gkmap, [64, 0]) * (1 - taper[..., 64:Ny_pad - 64, :]))
            if verbose: print("starting horizontal blending")
            taper = get_taper(gkmap, 0, self.taper_width, batch_maker, unbatch_maker)
            output_imgs[..., :, 64:Nx_pad - 64] = (output_imgs * taper)[..., :, 64:Nx_pad - 64]
            output_imgs[..., :, 64:Nx_pad - 64] += (
                    get_taper_imgs(gkmap, [0, 64]) * (1 - taper[..., :, 64:Nx_pad - 64]))

        output_imgs = enmap.enmap(self.unnormalizer(output_imgs), wcs)
        if ret_corr: corr = enmap.enmap(self.unnormalizer(corr), wcs)
        for post_process in post_processes:
            output_imgs = post_process(output_imgs)
        alms = None

        
        def fft_taper(lcrit, taper_width=200):
            ell = np.arange(lcrit+taper_width+1)
            f = np.ones(len(ell))
            f[lcrit:lcrit+taper_width+1] = np.cos(np.linspace(0,np.pi/2,taper_width+1))
            f[-1] = 0
            return ell, f


        def fft_taper(lcrit, taper_width=10000, order=5):
            ell = np.arange(lcrit+taper_width+1)
            f = np.ones(len(ell))
            f[lcrit:lcrit+taper_width+1] = np.cos(np.linspace(0,np.pi/2,taper_width+1))**5
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

        if deconv_beam or transfer:
            taper = omaps.cosine_window(Ny, Nx, 20, 20)
            minval = np.min(taper[taper!= 0])
            taper[taper==0] = minval
            if use_sht:
                raise NotImplemented()
                l_intp = np.arange(self.lmax + 1)
                f_int = interp_func(l_intp)
                alms = curvedsky.map2alm(output_imgs, lmax=self.lmax, spin=0)
                for i in range(len(compts)):
                    alms[i] = hp.almxfl(alms[i], f_int)
                output_imgs = curvedsky.alm2map(alms, ret, spin=0)
                del alms
            else:
                ftmap = enmap.fft(output_imgs)
                modlmap = enmap.modlmap((Ny, Nx) , wcs).ravel()
                combined_tfs = _load_combined_tf_funcs(modlmap, use_cache, self.cache_dir)
                for i in range(5):
                    ftmap[i] = ftmap[i] * np.reshape(combined_tfs[:, i+1], (Ny, Nx))
                del modlmap
                output_imgs = enmap.ifft(ftmap).real;
                del ftmap
            #output_imgs = output_imgs/taper
        
        if flux_cut is not None:
            flux_cut = flux_cut*1e-3/(0.5*utils.arcmin)**2*jysr2thermo(148)
            for i in range(3,5):
                loc = np.where(output_imgs[i]>flux_cut)
                output_imgs[i][loc]=0.

        return output_imgs  if not ret_corr else (output_imgs, corr)
