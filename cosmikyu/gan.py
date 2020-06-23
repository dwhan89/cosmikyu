from cosmikyu import config
import numpy as np
import os

from torchvision.utils import save_image
from torch.autograd import Variable

import torch.nn as nn
import torch

import mlflow

class GAN(object):
    def __init__(self, identifier, shape, latent_dim, output_path=None, sample_path=None, cuda=False, ngpu=1):
        self.cuda = cuda
        if not self.cuda:
            ngpu = 0
        self.shape = shape
        self.latent_dim = latent_dim
        self.identifier = identifier
        
        self.output_path = output_path or os.path.join(config.default_output_dir)
        self.tracking_path = os.path.join(self.output_path, "mlruns")
        mlflow.set_tracking_uri(self.tracking_path)
        self.experiment = mlflow.get_experiment_by_name(identifier) or mlflow.create_experiment(identifier)

        if torch.cuda.is_available() and not self.cuda:
            print("[WARNING] You have a CUDA device. You probably want to run with CUDA enabled")
        self.device = torch.device("cuda:0" if self.cuda else "cpu")

        self.generator = None
        self.discriminator = None

    def load_states(self):
        generator_state_file = os.path.join(self.output_path, "generator.pt")
        discriminator_state_file = os.path.join(self.output_path, "discriminator.pt")

        try:
            print("loading saved states")
            ## fix here
            self.generator.load_state_dict(torch.load(generator_state_file, map_location=self.device))
            self.discriminator.load_state_dict(torch.load(discriminator_state_file, map_location=self.device))
        except Exception:
            print("failed to load saved states")

    def save_states(self):
        print("saving states")
        generator_state_file = os.path.join(self.output_path, "generator.pt")
        discriminator_state_file = os.path.join(self.output_path, "discriminator.pt")
        torch.save(self.generator.state_dict(), generator_state_file)
        torch.save(self.discriminator.state_dict(), discriminator_state_file)

    def generate_image(self, num_imgs, seed=None):
        if not seed: np.random.seed(seed)
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        z = Variable(Tensor(np.random.normal(0, 1, (num_imgs, self.latent_dim))))
        return self.generator(z).detach()

    def train(self, dataloader, lr=0.00005, nepochs=200, clip_tresh=0.01, num_critic=5, sample_interval=1000,
              save_interval=10000, load_states=True, save_states=True, verbose=True, visdom_plotter=None, mlflow_runid=None):

        if load_states:
            self.load_states()

        opt_gen = torch.optim.RMSprop(self.generator.parameters(), lr=lr)
        opt_disc = torch.optim.RMSprop(self.discriminator.parameters(), lr=lr)
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        batches_done = 0

        for epoch in range(nepochs):
            for i, sample in enumerate(dataloader):
                imgs = sample[0]
                real_imgs = Variable(imgs.type(Tensor))

                opt_disc.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))))

                # Generate a batch of images
                fake_imgs = self.generator(z).detach()
                # Adversarial loss
                loss_D = -torch.mean(self.discriminator(real_imgs)) + torch.mean(self.discriminator(fake_imgs))

                loss_D.backward()
                opt_disc.step()

                # Clip weights of discriminator
                for p in self.discriminator.parameters():
                    p.data.clamp_(-clip_tresh, clip_tresh)

                if i % num_critic == 0:
                    opt_gen.zero_grad()

                    # Generate a batch of images
                    gen_imgs = self.generator(z)
                    # Adversarial loss
                    loss_G = -torch.mean(self.discriminator(gen_imgs))

                    loss_G.backward()
                    opt_gen.step()

                    if verbose:
                        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                              % (epoch, nepochs, batches_done % len(dataloader), len(dataloader), -1 * loss_D.item(),
                                 -1 * loss_G.item())
                              )

                if batches_done % sample_interval == 0:
                    pass
                    # save_image(gen_imgs.data[:5], os.path.join(self.sample_path, "%d.png" % batches_done), normalize=True)
                if batches_done % save_interval == 0 and save_states:
                    self.save_states()
                batches_done += 1

                if visdom_plotter is not None:
                    visdom_plotter.plot("D loss", 'D loss', np.array([batches_done]), np.array([-1 * loss_D.item()]),
                                        xlabel='batches_done')
                    visdom_plotter.plot("G loss", 'G_loss', np.array([batches_done]), np.array([-1 * loss_G.item()]),
                                        xlabel='batches_done')
        if save_states:
            self.save_states()


class WGAN(GAN):
    def __init__(self, identifier, shape, latent_dim, output_path=None, sample_path=None, cuda=False, ngpu=1):
        super().__init__(identifier, shape, latent_dim, output_path=output_path, sample_path=sample_path, cuda=cuda, ngpu=ngpu)

        self.generator = WGAN_Generator(shape, latent_dim, ngpu=ngpu).to(device=self.device)
        self.discriminator = WGAN_Discriminator(shape, ngpu=ngpu).to(device=self.device)


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
        if z.is_cuda and self.ngpu > 1:
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
        if img.is_cuda and self.ngpu > 1:
            ret = nn.parallel.data_parallel(self.model, flattened, range(self.ngpu))
        else:
            ret = self.model(flattened)
        return ret
