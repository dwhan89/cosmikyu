import torch
import torch.nn as nn

class Sinh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample):
        return torch.sinh(sample)

class SehgalActivationLayer(nn.Module):
    def __init__(self, threshold_settings):
        super().__init__()
        self.threshold_settings = threshold_settings

    def forward(self, sample):
        ret = sample.clone()
        ret[:, :2, :, :] = torch.tanh(sample[:, :2, :, :] / 20.) * 20.
        for i in [2, 3, 4]:
            minval, maxval = self.threshold_settings[i]
            loc = sample[:, i, :, :] != torch.clamp(sample[:, i, :, :], minval, maxval)
            ret[:, i, :, :][loc] = 0.
        return ret

class ScaledTanh(nn.Module):
    def __init__(self, a=15., b=2. / 15.):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, sample):
        return torch.tanh(sample * self.b) * self.a


class MultiHardTanh(nn.Module):
    def __init__(self, tanh_settings):
        super().__init__()
        self.nchannels = len(tanh_settings)
        self.tanh_settings = tanh_settings

    def forward(self, sample):
        ret = sample.clone()
        for i in range(self.nchannels):
            minval, maxval = self.tanh_settings[i]
            ret[:, i, :, :] = nn.functional.hardtanh(sample[:, i, :, :], minval, maxval)

        return ret

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)
