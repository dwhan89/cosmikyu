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
import torchsummary

data_dir = config.default_data_dir
lsun_dir = os.path.join(data_dir, 'lsun')
cuda = True
shape = (3,128,128)
latent_dim = 64
sample_interval = 1000
save_interval = 50000
batch_size = 256
nepochs = 10
nsamples = batch_size*1000

# Configure data loader
os.makedirs(data_dir, exist_ok=True)
os.makedirs(lsun_dir, exist_ok=True)
LSUN_DATASET = datasets.LSUN(
        root=lsun_dir,
        classes=['bedroom_train'],
        transform=transforms.Compose([transforms.Resize(shape[-1]), transforms.CenterCrop(shape[-1]), transforms.ToTensor()])
    )
LSUN_DATASET,_ = torch.utils.data.random_split(LSUN_DATASET, [nsamples, len(LSUN_DATASET)-nsamples])

dataloader = torch.utils.data.DataLoader(
    LSUN_DATASET,
    batch_size=batch_size,
    shuffle=True,
)
print("training")

sys.exit()
COSMOGAN = gan.COSMOGAN("lsun_cosmogan", shape, latent_dim, cuda=cuda, ngpu=4)
mlflow.set_experiment(COSMOGAN.identifier)          
with mlflow.start_run(experiment_id=COSMOGAN.experiment.experiment_id) as mlflow_run:
    torch.cuda.empty_cache()
    COSMOGAN.train(
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

os.exit()
WGAN = gan.WGAN("lsun_wgan", shape, latent_dim, cuda=cuda, ngpu=4)
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
os.exit()
WGAN_GP = gan.WGAN_GP("lsun_wgan_gp", shape, latent_dim, cuda=cuda, ngpu=4)
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


