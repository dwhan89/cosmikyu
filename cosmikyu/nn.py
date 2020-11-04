import torch
import torch.nn as nn
import numpy as np

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

class LinearFeature(nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(LinearFeature, self).__init__()
        assert(out_features == in_features)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.normal_(self.weight,1,0.02)
        if self.bias is not None:
            nn.init.normal_(self.bias,0,0.02)
    
    def forward(self, input):
        device = torch.device("cuda" if input.is_cuda else "cpu")
        eye = nn.Parameter(torch.zeros(input.shape), requires_grad=False).to(device=device)
        idxes = np.arange(input.shape[-1])
        eye[...,idxes, idxes] = 1.
        mat_weight = torch.einsum("i,...ijk->...ijk", self.weight, eye)

        if self.bias is not None:
            ones = nn.Parameter(torch.ones(input.shape), requires_grad=False).to(device=device)
            mat_bias = torch.einsum("i,...ijk->...ijk", self.bias, ones)

        return torch.matmul(input,mat_weight) if self.bias is None else torch.matmul(input,mat_weight) + mat_bias 

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

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
