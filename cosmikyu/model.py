import numpy as np
from torch import nn as nn
import torch

import cosmikyu.nn as cnn

class WGAN_Generator(nn.Module):
    def __init__(self, shape, latent_dim, ngpu=1):
        super(WGAN_Generator, self).__init__()
        self.shape = shape
        self.latent_dim = latent_dim
        self.ngpu = ngpu

        def custom_layer(dim_in, dim_out, batch_normalize=True):
            layers = [nn.Linear(dim_in, dim_out)]
            if batch_normalize:
                layers.append(nn.BatchNorm1d(dim_out, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *custom_layer(self.latent_dim, 128, batch_normalize=False),
            *custom_layer(128, 256),
            *custom_layer(256, 512),
            *custom_layer(512, 1024),
            nn.Linear(1024, int(np.prod(shape))),
            nn.Tanh()
        )

    def forward(self, z):
        if z.is_cuda and self.ngpu > 0:
            img = nn.parallel.data_parallel(self.model, z, range(self.ngpu))
        else:
            img = self.model(z)
        img = img.view(img.shape[0], *self.shape)
        return img


class WGAN_Discriminator(nn.Module):
    def __init__(self, shape, ngpu=1):
        super(WGAN_Discriminator, self).__init__()
        self.shape = shape
        self.ngpu = ngpu

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, img):
        flattened = img.view(img.shape[0], -1)
        if img.is_cuda and self.ngpu > 0:
            ret = nn.parallel.data_parallel(self.model, flattened, range(self.ngpu))
        else:
            ret = self.model(flattened)
        return ret


class DCGAN_SIMPLE_Generator(nn.Module):
    def __init__(self, shape, latent_dim, nconv_layer=2, nconv_fc=32, ngpu=1):
        super().__init__()

        self.shape = shape
        self.nconv_layer = nconv_layer
        self.latent_dim = latent_dim
        self.ngpu = ngpu
        self.nconv_fc = nconv_fc
        self.ds_size = shape[-1] // 2 ** (self.nconv_layer)

        nconv_lc = nconv_fc * 2 ** (self.nconv_layer - 1)

        def _get_conv_layers(nconv_layer, nconv_lc):
            conv_layers = []
            for i in range(nconv_layer - 1):
                conv_layers.extend([nn.Upsample(scale_factor=2),
                                    nn.Conv2d(nconv_lc // 2 ** i, nconv_lc // 2 ** (i + 1), 3, stride=1, padding=1),
                                    nn.BatchNorm2d(nconv_lc // 2 ** (i + 1), 0.8),
                                    nn.LeakyReLU(0.2, inplace=True)])
            return conv_layers

        layers = [nn.Linear(self.latent_dim, nconv_lc * self.ds_size ** 2),
                  cnn.Reshape((nconv_lc, self.ds_size, self.ds_size)),
                  nn.BatchNorm2d(nconv_lc)]

        layers.extend(_get_conv_layers(self.nconv_layer, nconv_lc))

        layers.extend([nn.Upsample(scale_factor=2),
                       nn.Conv2d(self.nconv_fc, self.shape[0], 3, stride=1, padding=1),
                       nn.Tanh()])

        self.model = nn.Sequential(*layers)

    def forward(self, z):
        if z.is_cuda and self.ngpu > 0:
            ret = nn.parallel.data_parallel(self.model, z, range(self.ngpu))
        else:
            ret = self.model(z)
        return ret


class DCGAN_SIMPLE_Discriminator(nn.Module):
    def __init__(self, shape, nconv_layer=2, nconv_fc=32, ngpu=1):
        super().__init__()
        self.shape = shape
        self.ngpu = ngpu
        self.nconv_layer = nconv_layer
        self.nconv_fc = nconv_fc
        self.ds_size = shape[-1] // 2 ** (self.nconv_layer)
        nconv_lc = nconv_fc * 2 ** (self.nconv_layer - 1)

        def discriminator_block(in_filters, out_filters, normalize=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if normalize:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        layers = [*discriminator_block(self.shape[0], nconv_fc, normalize=False)]
        for i in range(self.nconv_layer - 1):
            layers.extend(discriminator_block(self.nconv_fc * 2 ** (i), self.nconv_fc * 2 ** (i + 1)))

        layers.extend(
            [cnn.Reshape((nconv_lc * self.ds_size ** 2,)), nn.Linear(nconv_lc * self.ds_size ** 2, 1), nn.Sigmoid()])
        self.model = nn.Sequential(*layers)

    def forward(self, img):
        if img.is_cuda and self.ngpu > 0:
            ret = nn.parallel.data_parallel(self.model, img, range(self.ngpu))
        else:
            ret = self.model(img)
        return ret


class DCGAN_Generator(nn.Module):
    def __init__(self, shape, latent_dim, nconv_layer=2, nconv_fc=32, ngpu=1, kernal_size=5, stride=2, padding=2,
                 output_padding=1, activation=None):
        super().__init__()

        self.shape = shape
        self.nconv_layer = nconv_layer
        self.latent_dim = latent_dim
        self.ngpu = ngpu
        self.nconv_fc = nconv_fc
        self.kernal_size = kernal_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.ds_size = shape[-1] // self.stride ** self.nconv_layer
        self.activation = activation
        nconv_lc = nconv_fc * self.stride ** (self.nconv_layer - 1)

        def _get_conv_layers(nconv_layer, nconv_lc):
            conv_layers = []
            for i in range(nconv_layer - 1):
                conv_layers.extend([nn.ConvTranspose2d(nconv_lc // self.stride ** i, nconv_lc // self.stride ** (i + 1),
                                                       self.kernal_size, stride=self.stride, padding=self.padding,
                                                       output_padding=self.output_padding),
                                    nn.BatchNorm2d(nconv_lc // self.stride ** (i + 1)),
                                    nn.LeakyReLU(0.2, inplace=True)])
            return conv_layers

        layers = [nn.Linear(self.latent_dim, nconv_lc * self.ds_size ** 2),
                  cnn.Reshape((nconv_lc, self.ds_size, self.ds_size)),
                  nn.BatchNorm2d(nconv_lc),
                  nn.LeakyReLU(0.2, inplace=True)]

        layers.extend(_get_conv_layers(self.nconv_layer, nconv_lc))

        layers.extend([nn.ConvTranspose2d(self.nconv_fc, self.shape[0], self.kernal_size, stride=self.stride,
                                          padding=self.padding,
                                          output_padding=output_padding)])
        if self.activation is not None:
            layers.extend(self.activation)
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        if z.is_cuda and self.ngpu > 0:
            ret = nn.parallel.data_parallel(self.model, z, range(self.ngpu))
        else:
            ret = self.model(z)
        return ret

class DCGAN_Discriminator_BASE(nn.Module):
    def __init__(self, shape, nconv_layer=2, nconv_fc=32, ngpu=1, kernal_size=5, stride=2, padding=2, normalize=True):
        super().__init__()
        self.shape = shape
        self.ngpu = ngpu
        self.nconv_layer = nconv_layer
        self.nconv_fc = nconv_fc
        self.kernal_size = kernal_size
        self.stride = stride
        self.padding = padding
        self.ds_size = shape[-1] // self.stride ** self.nconv_layer

        nconv_lc = nconv_fc * self.stride ** (self.nconv_layer - 1)
        self.nconv_lc = nconv_lc

        layers = [*self.discriminator_block(self.shape[0], nconv_fc, normalize=False)]
        for i in range(self.nconv_layer - 1):
            layers.extend(
                self.discriminator_block(self.nconv_fc * self.stride ** i, self.nconv_fc * self.stride ** (i + 1),
                                    normalize=normalize))

        layers.extend(self.get_last_layer())

        self.model = nn.Sequential(*layers)

    def discriminator_block(self, in_filters, out_filters, kernal_size=None, stride=None,  normalize=True):
        if kernal_size is None : kernal_size=self.kernal_size
        if stride is None : stride=self.stride 
        block = [nn.Conv2d(in_filters, out_filters, kernal_size, stride, padding=self.padding)]
        if normalize:
            block.append(nn.BatchNorm2d(out_filters))
        block.append(nn.LeakyReLU(0.2, inplace=True))
        return block

    def get_last_layer(self):
        raise NotImplemented()
        
    def forward(self, img):
        if img.is_cuda and self.ngpu > 0:
            ret = nn.parallel.data_parallel(self.model, img, range(self.ngpu))
        else:
            ret = self.model(img)
        return ret

class DCGAN_Discriminator(DCGAN_Discriminator_BASE):
    def __init__(self, shape, nconv_layer=2, nconv_fc=32, ngpu=1, kernal_size=5, stride=2, padding=2, normalize=False):
        super().__init__(shape=shape, nconv_layer=nconv_layer, nconv_fc=nconv_fc, ngpu=ngpu, kernal_size=kernal_size,
                         stride=stride, padding=padding, normalize=normalize)

    def get_last_layer(self):
        return [cnn.Reshape((self.nconv_lc * self.ds_size ** 2,)),
                nn.Linear(self.nconv_lc * self.ds_size ** 2, 1)]

class UNET_Discriminator(DCGAN_Discriminator):
    def __init__(self, shape, nconv_layer=2, nconv_fc=32, ngpu=1, kernal_size=5, stride=2, padding=2, normalize=True, nthresh_layer=1):
        self.nthresh_layer = nthresh_layer

        super().__init__(shape=shape, nconv_layer=nconv_layer, nconv_fc=nconv_fc, ngpu=ngpu, kernal_size=kernal_size,
                         stride=stride, padding=padding, normalize=normalize)
    
    def get_last_layer(self):
        layers = []
        for i in range(self.nthresh_layer):
            nin_filt = self.nconv_fc * self.stride ** ((self.nconv_layer - 2) + 1) if i == 0 else 512
            layers.extend(self.discriminator_block(nin_filt, 512, stride=1, normalize=True))
        layers.extend([nn.Conv2d(512, 1, self.kernal_size, stride=1, padding=self.padding)])
        return layers

class UNET_Discriminator_WGP(DCGAN_Discriminator_BASE):
    def __init__(self, shape, nconv_layer=2, nconv_fc=32, ngpu=1, kernal_size=5, stride=2, padding=2, normalize=False, nthresh_layer=1):
        self.nthresh_layer = nthresh_layer
        super().__init__(shape=shape, nconv_layer=nconv_layer, nconv_fc=nconv_fc, ngpu=ngpu, kernal_size=kernal_size,
                         stride=stride, padding=padding, normalize=normalize)
    
    def get_last_layer(self):
        nin_filt = self.nconv_fc * self.stride ** ((self.nconv_layer - 2) + 1)
        layers = []
        for i in range(self.nthresh_layer):

            layers.extend(self.discriminator_block(nin_filt, nin_filt, normalize=False))
        final_ds_size = self.ds_size//self.stride**self.nthresh_layer
        layers.extend([cnn.Reshape((self.nconv_lc * final_ds_size ** 2,)),nn.Linear(self.nconv_lc * final_ds_size ** 2, 1)])
        return layers

class UNET_Generator(nn.Module):
    def __init__(self, shape, nconv_layer=2, nconv_fc=32, ngpu=1, kernal_size=5, stride=2, padding=2,
                 output_padding=1, normalize=True, activation=None, nin_channel=3, nout_channel=3,
                 nthresh_layer=1, dropoff=True):
        super().__init__()
        self.shape = shape 
        self.nconv_layer = nconv_layer
        self.normalize = normalize
        self.ngpu = ngpu
        self.nconv_fc = nconv_fc
        self.kernal_size = kernal_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.ds_size = shape[-1] // self.stride ** self.nconv_layer
        self.activation = activation
        self.model_dict = nn.ModuleDict()
        self.nin_channel = nin_channel
        self.nout_channel = nout_channel
        self.nthresh_layer = nthresh_layer
        self.ntotal_layer = nthresh_layer + nconv_layer

        nconv_lc = nconv_fc * self.stride ** (self.nconv_layer - 1)
        class UNetDown(nn.Module):
            def __init__(self, in_filters, out_filters, normalize, dropout_rate=0, kernal_size=self.kernal_size,
                         stride=self.stride, padding=self.padding, ngpu=ngpu, use_leaky=True):
                super().__init__()
                block = [nn.Conv2d(in_filters, out_filters, kernal_size, stride=stride, padding=padding)]
                if dropout_rate != 0:
                    assert(1 >= dropout_rate >= 0)
                    block.append(nn.Dropout(dropout_rate))
                if normalize:
                    block.append(nn.BatchNorm2d(out_filters))
                if use_leaky:
                    block.append(nn.LeakyReLU(0.2, inplace=True))
                else:
                    block.append(nn.ReLU(inplace=True))
                self.model = nn.Sequential(*block)
                self.ngpu = ngpu

            def forward(self, z):
                if z.is_cuda and self.ngpu > 0:
                    ret = nn.parallel.data_parallel(self.model, z, range(self.ngpu))
                else:
                    ret = self.model(z)
                return ret

        class UNetUP(nn.Module):
            def __init__(self, in_filters, out_filters, normalize, dropout_rate=0, kernal_size=self.kernal_size,
                         stride=self.stride, padding=self.padding, output_padding=self.output_padding, ngpu=ngpu, activation=True):
                super().__init__()
                block = [nn.ConvTranspose2d(in_filters, out_filters, kernal_size, stride=stride, padding=padding
                                            ,output_padding=output_padding)]
                if normalize:
                    block.append(nn.BatchNorm2d(out_filters))
                if dropout_rate != 0:
                    assert(1 >= dropout_rate >= 0)
                    block.append(nn.Dropout(dropout_rate))
                if activation:
                    block.append(nn.ReLU(inplace=True))
                self.model = nn.Sequential(*block)
                self.ngpu = ngpu

            def forward(self, z, skip_input):
                if z.is_cuda and self.ngpu > 0:
                    ret = nn.parallel.data_parallel(self.model, z, range(self.ngpu))
                else:
                    ret = self.model(z)
                if skip_input is not None:
                    ret = torch.cat((ret, skip_input),1)

                return ret
        ## define down layers
        self.model_dict["down0"] = UNetDown(self.nin_channel, nconv_fc, normalize=False, dropout_rate=0.0)
        for i in range(1, self.nconv_layer):
            self.model_dict["down%d"%(i)] = UNetDown(self.nconv_fc * self.stride ** (i-1),
                                                       self.nconv_fc * self.stride ** i, normalize=True)
        
        ## bottom treshold layers
        for i in range(self.nthresh_layer):
            down_idx = "down%d"%(self.nconv_layer + i)
            use_leaky = i < self.nthresh_layer-1
            normalize = i < self.nthresh_layer-1
            self.model_dict[down_idx] = UNetDown(nconv_lc, nconv_lc, normalize=normalize, use_leaky=use_leaky) 
        for i in range(self.nthresh_layer):
            up_idx = "up%d" %i
            upin_channel = nconv_lc if i == 0 else nconv_lc*2
            dropout_rate = 0.5 if i > 0 and dropoff else 0.0
            self.model_dict[up_idx] = UNetUP(upin_channel, nconv_lc, normalize=True,dropout_rate=dropout_rate)
        
        ## up layers

        for i in range(self.nconv_layer+1):
            self.model_dict["up%d"%(i+self.nthresh_layer)] = UNetUP(int(nconv_lc * self.stride ** (-i+1)), int(nconv_lc * self.stride ** (-i-1)),
                                                    normalize=True,dropout_rate=0)
        self.model_dict["up%d"% (self.ntotal_layer-1)] = UNetUP(self.nconv_fc *2, self.nout_channel,  normalize=False,dropout_rate=0, activation=False)
        if self.activation is not None:
            self.model_dict["final"] = nn.Sequential(*self.activation)
    def forward(self, img):
        ret = {"down-1": img}
        for i in range(self.ntotal_layer):
            input_key = "down%d"%(i-1)
            model_key = "down%d"%(i)
            ret[model_key] = self.model_dict[model_key](ret[input_key])
        ret.pop("down-1")
        ret_up = ret[model_key]
        
        for i in range(self.ntotal_layer):
            skip_key = "down%d"% (self.ntotal_layer -2 -i)
            model_key = "up%d" % (i)
            ret_up = self.model_dict[model_key](ret_up, ret[skip_key] if skip_key in ret else None)
        return ret_up if "final" not in self.model_dict else self.model_dict["final"](ret_up)
