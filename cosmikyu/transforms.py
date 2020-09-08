import numpy as np
from pixell import enmap
from orphics import maps as omaps
from . import utils


class SehgalDataNormalizerSymLogMinMax(object):
    def __init__(self, normalization_info_file, log_norm_skip=[0, 1]):
        temp = np.load(normalization_info_file, allow_pickle=True)
        self.norm_info = utils.load_data(normalization_info_file)
        self.channel_idxes = ["kappa", "ksz", "tsz", "ir_pts", "rad_pts"]
        self.nchannel = len(self.channel_idxes)
        self.log_normalizer = ToSymLogScale()
        self.minmax_normalizers = [None] * self.nchannel
        self.log_norm_skip = log_norm_skip
        for i, channel_idx in enumerate(self.channel_idxes):
            temp = self.norm_info[channel_idx]
            self.minmax_normalizers[i] = MinMaxNormalize(temp["mean"], temp["min"], temp["max"])

    def __call__(self, sample):
        assert (len(sample.shape) == 3)
        for i in range(self.nchannel):
            if i not in self.log_norm_skip: sample[i] = self.log_normalizer(sample[i, ...])
            sample[i] = self.minmax_normalizers[i](sample[i, ...])
        return sample


class SehgalDataUnnormalizerSymLogMinMax(object):
    def __init__(self, normalization_info_file, log_norm_skip=[0, 1]):
        temp = np.load(normalization_info_file, allow_pickle=True)
        self.norm_info = utils.load_data(normalization_info_file)
        self.channel_idxes = ["kappa", "ksz", "tsz", "ir_pts", "rad_pts"]
        self.nchannel = len(self.channel_idxes)
        self.log_unnormalizer = FromSymLogScale()
        self.minmax_unnormalizers = [None] * self.nchannel
        self.log_norm_skip = log_norm_skip
        for i, channel_idx in enumerate(self.channel_idxes):
            temp = self.norm_info[channel_idx]
            self.minmax_unnormalizers[i] = MinMaxUnnormalize(temp["mean"], temp["min"], temp["max"])

    def __call__(self, sample):
        assert (len(sample.shape) == 3)
        for i in range(self.nchannel):
            sample[i] = self.minmax_unnormalizers[i](sample[i, ...])
            if i not in self.log_norm_skip: sample[i] = self.log_unnormalizer(sample[i, ...])
        return sample


class SehgalDataNormalizerPowZ(object):
    def __init__(self, normalization_info_file, zfact=2, power_normalize=True, z_normalize=True):
        temp = np.load(normalization_info_file, allow_pickle=True)
        self.norm_info = {key: temp[key].item() for key in temp}
        self.power_normalize = power_normalize
        self.z_normalize = z_normalize
        self.channel_idxes = ["kappa", "ksz", "tsz", "ir_pts", "rad_pts"]
        self.nchannel = len(self.channel_idxes)
        self.power_normalizers = [None] * self.nchannel
        self.z_normalizers = [None] * self.nchannel

        if self.power_normalize:
            for i, channel_idx in enumerate(self.channel_idxes):
                temp = self.norm_info[channel_idx]
                self.power_normalizers[i] = PowerNormalize(temp["pow_neg"], temp["pow_pos"])
        else:
            pass
        if self.z_normalize:
            for i, channel_idx in enumerate(self.channel_idxes):
                temp = self.norm_info[channel_idx]
                zfact = temp["z_fact"] if zfact is None else zfact
                self.z_normalizers[i] = ZNormalize(temp["mean"], temp["std"], zfact)
        else:
            pass

    def __call__(self, sample):
        assert (len(sample.shape) == 3)
        for i in range(self.nchannel):
            if self.power_normalize: sample[i] = self.power_normalizers[i](sample[i, ...])
            if self.z_normalize: sample[i] = self.z_normalizers[i](sample[i, ...])
        return sample


class SehgalDataUnnormalizerPowZ(object):
    def __init__(self, normalization_info_file, zfact=2, power_normalize=True, z_normalize=True):
        temp = np.load(normalization_info_file, allow_pickle=True)
        self.norm_info = {key: temp[key].item() for key in temp}
        self.power_normalize = power_normalize
        self.z_normalize = z_normalize
        self.channel_idxes = ["kappa", "ksz", "tsz", "ir_pts", "rad_pts"]
        self.nchannel = len(self.channel_idxes)
        self.power_normalizers = [None] * self.nchannel
        self.z_normalizers = [None] * self.nchannel

        if self.power_normalize:
            for i, channel_idx in enumerate(self.channel_idxes):
                temp = self.norm_info[channel_idx]
                self.power_normalizers[i] = PowerUnnormalize(temp["pow_neg"], temp["pow_pos"])
        else:
            pass
        if self.z_normalize:
            for i, channel_idx in enumerate(self.channel_idxes):
                temp = self.norm_info[channel_idx]
                zfact = temp["z_fact"] if zfact is None else zfact
                self.z_normalizers[i] = ZUnnormalize(temp["mean"], temp["std"], zfact)
        else:
            pass

    def __call__(self, sample):
        assert (len(sample.shape) == 3)
        for i in range(self.nchannel):
            if self.z_normalize: sample[i] = self.z_normalizers[i](sample[i, ...])
            if self.power_normalize: sample[i] = self.power_normalizers[i](sample[i, ...])
        return sample


class SehgalDataNormalizerScaledLogZ(object):
    def __init__(self, normalization_info_file):
        temp = np.load(normalization_info_file, allow_pickle=True)
        self.norm_info = {key: temp[key].item() for key in temp}
        self.channel_idxes = ["kappa", "ksz", "tsz", "ir_pts", "rad_pts"]
        self.nchannel = len(self.channel_idxes)
        self.log_normalizers = [None] * self.nchannel
        self.z_normalizers = [None] * self.nchannel

        for i, channel_idx in enumerate(self.channel_idxes):
            temp = self.norm_info[channel_idx]
            flip_sign = channel_idx == "tsz"
            self.log_normalizers[i] = ToScaledLogScale(scaling=temp["std"]) if i > 1 else Identity()

        for i, channel_idx in enumerate(self.channel_idxes):
            temp = self.norm_info[channel_idx]
            self.z_normalizers[i] = ZNormalize(temp["logmean"], temp["logstd"], 1.)
        else:
            pass

    def __call__(self, sample):
        assert (len(sample.shape) == 3)
        for i in range(self.nchannel):
            sample[i] = self.log_normalizers[i](sample[i, ...])
            sample[i] = self.z_normalizers[i](sample[i, ...])
        return sample


class SehgalDataUnnormalizerScaledLogZ(object):
    def __init__(self, normalization_info_file):
        temp = np.load(normalization_info_file, allow_pickle=True)
        self.norm_info = {key: temp[key].item() for key in temp}
        self.channel_idxes = ["kappa", "ksz", "tsz", "ir_pts", "rad_pts"]
        self.nchannel = len(self.channel_idxes)
        self.log_unnormalizers = [None] * self.nchannel
        self.z_unnormalizers = [None] * self.nchannel

        for i, channel_idx in enumerate(self.channel_idxes):
            temp = self.norm_info[channel_idx]
            flip_sign = channel_idx == "tsz"
            self.log_unnormalizers[i] = FromScaledLogScale(scaling=temp["std"]) if i > 1 else Identity()

        for i, channel_idx in enumerate(self.channel_idxes):
            temp = self.norm_info[channel_idx]
            self.z_unnormalizers[i] = ZUnnormalize(temp["logmean"], temp["logstd"], 1.)
        else:
            pass

    def __call__(self, sample):
        assert (len(sample.shape) == 3)
        for i in range(self.nchannel):
            sample[i] = self.z_unnormalizers[i](sample[i, ...])
            sample[i] = self.log_unnormalizers[i](sample[i, ...])
        return sample


class SehgalSubcomponets(object):
    def __init__(self, idxes=None):
        if idxes is None:
            idxes = [0, 1, 2, 3, 4]
        self.idxes = idxes

    def __call__(self, sample):
        return sample[self.idxes, ...]


class BlockShape(object):
    # "https://stackoverflow.com/questions/16873441/form-a-big-2d-array-from-multiple-smaller-2d-arrays/16873755#16873755"
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, sample):
        h, w = sample.shape[-2], sample.shape[-1]
        nrows, ncols = self.shape[-2], self.shape[-1]
        return (sample.reshape(h // nrows, nrows, -1, ncols)
                .swapaxes(1, 2)
                .reshape(-1, nrows, ncols))


class UnBlockShape(object):
    # "https://stackoverflow.com/questions/16873441/form-a-big-2d-array-from-multiple-smaller-2d-arrays/16873755#16873755"
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, sample):
        n, nrows, ncols = sample.shape
        h, w = self.shape[-2], self.shape[-1]
        return (sample.reshape(h // nrows, -1, nrows, ncols)
                .swapaxes(1, 2)
                .reshape(1, h, w))


class RandomFlips(object):
    def __init__(self, p_h=0.5, p_v=0.5):
        self.p_v = p_v
        self.p_h = p_h

    def __call__(self, sample):
        temp = np.random.uniform(0., 1.0, size=2)
        if temp[0] < self.p_v:
            sample = np.flip(sample, -2)
        if temp[1] < self.p_h:
            sample = np.flip(sample, -1)
        return sample


class Identity(object):
    def __call__(self, sample):
        return sample


class PowerNormalize(object):
    def __init__(self, pow_neg, pow_pos):
        self.pow_neg = pow_neg
        self.pow_pos = pow_pos

    def __call__(self, sample):
        loc = sample >= 0
        sample[loc] = sample[loc] ** self.pow_pos
        sample[~loc] = -1 * np.abs(sample[~loc]) ** self.pow_neg
        return sample


class PowerUnnormalize(PowerNormalize):
    def __init__(self, pow_neg, pow_pos):
        super().__init__(1. / pow_neg, 1. / pow_pos)


class ZNormalize(object):
    def __init__(self, mean, std, zfact):
        self.mean = mean
        self.std = std
        self.zfact = zfact

    def __call__(self, sample):
        return (sample - self.mean) / (self.std * self.zfact)


class ZUnnormalize(object):
    def __init__(self, mean, std, zfact):
        self.mean = mean
        self.std = std
        self.zfact = zfact

    def __call__(self, sample):
        return sample * (self.std * self.zfact) + self.mean


class MinMaxNormalize(object):
    def __init__(self, mean, minval, maxval):
        self.mean = mean
        self.minval = minval
        self.maxval = maxval

    def __call__(self, sample):
        return (sample - self.mean) / (self.maxval - self.minval) * 2


class MinMaxUnnormalize(object):
    def __init__(self, mean, minval, maxval):
        self.mean = mean
        self.minval = minval
        self.maxval = maxval

    def __call__(self, sample):
        return sample / 2. * (self.maxval - self.minval) + self.mean


class ToScaledLogScale(object):
    def __init__(self, scaling=1.):
        self.scaling = scaling

    def __call__(self, sample):
        loc = sample >= 0
        sample[loc] = np.log(sample[loc] / self.scaling + 1.)
        sample[~loc] = -1 * np.log(np.abs(sample[~loc] / self.scaling) + 1.)
        return sample


class FromScaledLogScale(object):
    def __init__(self, scaling=1.):
        self.scaling = scaling

    def __call__(self, sample):
        loc = sample >= 0
        sample[loc] = (np.exp(sample[loc]) - 1.) * self.scaling
        sample[~loc] = -1 * ((np.exp(np.abs(sample[~loc])) - 1.) * self.scaling)
        return sample


class FromSymLogScale(object):
    def __call__(self, sample):
        loc = sample >= 0
        sample[loc] = np.exp(sample[loc]) - 1.
        sample[~loc] = -1 * (np.exp(np.abs(sample[~loc])) - 1.)
        return sample


class ToSymLogScale(object):
    def __call__(self, sample):
        loc = sample >= 0
        sample[loc] = np.log(sample[loc] + 1.)
        sample[~loc] = -1 * np.log(np.abs(sample[~loc]) + 1.)
        return sample


class ToEnmap(object):
    def __init__(self, wcs):
        self.wcs = wcs

    def __call__(self, sample):
        return enmap.enmap(sample, wcs=self.wcs)


class ToArray(object):
    def __call__(self, sample):
        return np.array(sample)


class Taper(object):
    def __init__(self, shape):
        self.shape = shape
        self.taper, _ = omaps.get_taper(shape, pad_percent=0.)
        loc = self.taper == 0
        self.taper[loc] = np.min(self.taper[~loc])

    def __call__(self, sample):
        assert ('map' in sample)
        sample['map'] = sample['map'] * self.taper
        return sample


class UnTaper(object):
    def __init__(self, shape):
        self.shape = shape
        self.taper, _ = omaps.get_taper(shape, pad_percent=0.)
        loc = self.taper == 0
        self.taper[loc] = np.min(self.taper[~loc])

    def __call__(self, sample):
        assert ('map' in sample)
        sample['map'] = np.nan_to_num(sample['map'] / self.taper)
        return sample


class TakePS(object):
    def __init__(self, bin_edges, shape, return_dl=True):
        self.bin_edges = bin_edges
        self.shape = shape
        self.taper, _ = omaps.get_taper(shape)
        loc = self.taper == 0
        self.taper[loc] = np.min(self.taper[~loc])
        self.return_dl = return_dl

    def __call__(self, sample):
        lbin, ps = omaps.binned_power(sample, self.bin_edges, mask=self.taper)
        ps = ps if not self.return_dl else ps * (lbin * (lbin + 1) / (2 * np.pi))
        return lbin, ps
