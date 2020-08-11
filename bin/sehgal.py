from cosmikyu import gan, config, datasets, transforms
import os
import torch
import mlflow
from orphics import maps
import sys

data_dir = config.default_data_dir
sehgal_dir = os.path.join(data_dir, 'sehgal')
cuda = True
compt_idxes = [0, 1, 2, 3, 4]
shape = (len(compt_idxes), 128, 128)
# shape = (1, 256, 256)
latent_dim = 64
sample_interval = 200
save_interval = 782
batch_size = 128
nepochs = 50
norm_info_file = "/home/dwhan89/workspace/cosmikyu/data/sehgal/normalization_info.npz"

_, wcs = maps.rect_geometry(width_arcmin=64., px_res_arcmin=0.5)

# Configure data loader
os.makedirs(data_dir, exist_ok=True)
os.makedirs(sehgal_dir, exist_ok=True)
SDN = transforms.SehgalDataNormalizer(norm_info_file, zfact=4)
SC = transforms.SehgalSubcomponets(compt_idxes)
# SUBS = transforms.UnBlockShape(shape)
RF = transforms.RandomFlips(p_v=0.5, p_h=0.5)
SDS_train = datasets.SehgalDataSet(sehgal_dir, data_type="train", transforms=[SDN, RF, SC], dummy_label=True)

dataloader = torch.utils.data.DataLoader(
    SDS_train,
    batch_size=batch_size,
    shuffle=True,
)

DCGAN_WGP = gan.DCGAN_WGP("sehgal_dcganwgp", shape, latent_dim, cuda=cuda, nconv_fcgen=128,
                          nconv_fcdis=128, ngpu=4, nconv_layer_gen=5, nconv_layer_disc=5, kernal_size=5, stride=2,
                          padding=2, output_padding=1)
mlflow.set_experiment(DCGAN_WGP.identifier)
with mlflow.start_run(experiment_id=DCGAN_WGP.experiment.experiment_id) as mlflow_run:
    torch.cuda.empty_cache()
    DCGAN_WGP.train(
        dataloader,
        nepochs=nepochs,
        ncritics=5,
        sample_interval=sample_interval,
        save_interval=save_interval,
        load_states=True,
        save_states=True,
        verbose=True,
        mlflow_run=mlflow_run,
        lr=2e-04,
        betas=(0.5, 0.999),
        lambda_gp=10.
    )

sys.exit()
COSMOGAN_WGAP = gan.COSMOGAN_WGP("sehgal_cosmoganwgp", shape, latent_dim, cuda=cuda, ngpu=4)
COSMOGAN_WGAP.load_states(
    "/home/dwhan89/workspace/cosmikyu/output/sehgal_cosmoganwgp/6803124a28554a13b4d9acdff40b5c97/model")
mlflow.set_experiment(COSMOGAN_WGAP.identifier)
with mlflow.start_run(experiment_id=COSMOGAN_WGAP.experiment.experiment_id) as mlflow_run:
    torch.cuda.empty_cache()
    COSMOGAN_WGAP.train(
        dataloader,
        nepochs=nepochs,
        ncritics=5,
        sample_interval=200,
        save_interval=10000,
        load_states=True,
        save_states=True,
        verbose=True,
        mlflow_run=mlflow_run,
        lr=2e-04,
        betas=(0.5, 0.999),
        lambda_gp=10.
    )

WGAN_GP = gan.WGAN_GP("sehgal_wgan_gp", shape, latent_dim, p_fliplabel=0., cuda=cuda, ngpu=4)
mlflow.set_experiment(WGAN_GP.identifier)
with mlflow.start_run(experiment_id=WGAN_GP.experiment.experiment_id) as mlflow_run:
    torch.cuda.empty_cache()
    WGAN_GP.train(
        dataloader,
        nepochs=nepochs,
        ncritics=5,
        sample_interval=200,
        save_interval=782,
        load_states=True,
        save_states=True,
        verbose=True,
        mlflow_run=mlflow_run,
        lr=1e-3,
        betas=(0.9, 0.999),
        lambda_gp=10,
    )

COSMOGAN = gan.COSMOGAN("sehgal_cosmogan", shape, latent_dim, p_fliplabel=0.05, cuda=cuda, ngpu=4, nconv_fcdis=64)
mlflow.set_experiment(COSMOGAN.identifier)
with mlflow.start_run(experiment_id=COSMOGAN.experiment.experiment_id) as mlflow_run:
    torch.cuda.empty_cache()
    COSMOGAN.train(
        dataloader,
        nepochs=nepochs,
        ncritics=1,
        sample_interval=200,
        save_interval=782,
        load_states=True,
        save_states=True,
        verbose=True,
        mlflow_run=mlflow_run,
        lr=2e-04,
        betas=(0.5, 0.999)
    )
