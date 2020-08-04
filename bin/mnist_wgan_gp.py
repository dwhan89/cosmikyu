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

data_dir = config.default_data_dir
mnist_dir = os.path.join(data_dir, 'mnist')
cuda = False
shape = (1,28,28)
latent_dim = 100
sample_interval = 1000
save_interval = 50000
batch_size = 64

# Configure data loader
os.makedirs(data_dir, exist_ok=True)
os.makedirs(mnist_dir, exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        mnist_dir,
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
    ),
    batch_size=batch_size,
    shuffle=True,
)



for nepochs in [200]:
    WGAN_GP = gan.WGAN_GP("mnist_wgan_gp", shape, latent_dim, cuda=True, ngpu=4)
    mlflow.set_experiment(WGAN_GP.identifier)
    with mlflow.start_run(experiment_id=WGAN_GP.experiment.experiment_id) as mlflow_run:
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
    
    
    
