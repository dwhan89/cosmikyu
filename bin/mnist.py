from cosmikyu import visualization as covis
from cosmikyu import gan, config
import numpy as np
import os
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import torch
import mlflow
import sys

data_dir = config.default_data_dir
mnist_dir = os.path.join(data_dir, 'mnist')
cuda = True
shape = (1,32,32)
latent_dim = 64
sample_interval = 1000
save_interval = 50000
batch_size = 64
nepochs=100

# Configure data loader
os.makedirs(data_dir, exist_ok=True)
os.makedirs(mnist_dir, exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        mnist_dir,
        train=True,
        download=True,
        transform=transforms.Compose([transforms.Resize(shape[-1]), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
    ),
    batch_size=batch_size,
    shuffle=True,
)

## DCGAN SIMPLE
DCGAN = gan.DCGAN("mnist_cosmogan", shape, latent_dim, cuda=cuda, ngpu=4, nconv_layer_gen=2, nconv_layer_disc=2, nconv_fcgen=32, nconv_fcdis=32,
        kernal_size=5, stride=2, padding=2, output_padding=1)
mlflow.set_experiment(DCGAN.identifier)
with mlflow.start_run(experiment_id=DCGAN.experiment.experiment_id) as mlflow_run:
    torch.cuda.empty_cache()
    DCGAN.train(
        dataloader,
        nepochs=nepochs,
        ncritics=1,
        sample_interval=1000,
        save_interval=10000,
        load_states=True,
        save_states=True,
        verbose=True,
        mlflow_run=mlflow_run,
        lr=2e-04,
        betas=(0.5, 0.999)
    )

## DCGAN SIMPLE
DCGAN_SIMPLE = gan.DCGAN_SIMPLE("mnist_dcgan_simple", shape, latent_dim, cuda=cuda, ngpu=4, nconv_layer_gen=2, nconv_layer_disc=2, nconv_fcgen=32, nconv_fcdis=32)
mlflow.set_experiment(DCGAN_SIMPLE.identifier)
with mlflow.start_run(experiment_id=DCGAN_SIMPLE.experiment.experiment_id) as mlflow_run:
    torch.cuda.empty_cache()
    DCGAN_SIMPLE.train(
        dataloader,
        nepochs=nepochs,
        ncritics=1,
        sample_interval=1000,
        save_interval=10000,
        load_states=True,
        save_states=True,
        verbose=True,
        mlflow_run=mlflow_run,
        lr=2e-04,
        betas=(0.5, 0.999)
    )

## WGAN_GP
WGAN_GP = gan.WGAN_GP("mnist_wgan_gp", shape, latent_dim, cuda=cuda, ngpu=4)
mlflow.set_experiment(WGAN_GP.identifier)
with mlflow.start_run(experiment_id=WGAN_GP.experiment.experiment_id) as mlflow_run:
    torch.cuda.empty_cache()
    WGAN_GP.train(
        dataloader,
        nepochs=nepochs,
        ncritics=5,
        sample_interval=1000,
        save_interval=10000,
        load_states=True,
        save_states=True,
        verbose=True,
        mlflow_run=mlflow_run,
        lr=2e-04,
        betas=(0.5, 0.999),
        lambda_gp=10,
    )

## WGAN
WGAN = gan.WGAN("mnist_wgan", shape, latent_dim, cuda=cuda, ngpu=4)
mlflow.set_experiment(WGAN.identifier)
with mlflow.start_run(experiment_id=WGAN.experiment.experiment_id) as mlflow_run:
    torch.cuda.empty_cache()
    WGAN.train(
        dataloader,
        nepochs=nepochs,
        ncritics=5,
        sample_interval=1000,
        save_interval=10000,
        load_states=True,
        save_states=True,
        verbose=True,
        mlflow_run=mlflow_run,
        lr=2e-04,
        clip_tresh=0.01,
    )
