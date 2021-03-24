from cosmikyu import gan, config, datasets, transforms, utils
from cosmikyu import nn as cnn
import os
import torch
import mlflow
from orphics import maps
import numpy as np

data_dir = config.default_data_dir
sehgal_dir = os.path.join(data_dir, 'sehgal')
cuda = True
compts = ["kappa", "ksz", "tsz", "ir_pts", "rad_pts"]
compt_idxes = [0, 1, 2, 3, 4]
shape = (len(compt_idxes), 128, 128)
latent_dim = 256
sample_interval = 500
save_interval = 1
batch_size = 32
nepochs = 100
#norm_info_file = "/home/dwhan89/workspace/cosmikyu/data/sehgal/201020_logzshrink_normalization_info_validation.npz"
norm_info_file = "/home/dwhan89/workspace/cosmikyu/data/sehgal/281220_logz_normalization_info_validation.npz"

_, wcs = maps.rect_geometry(width_arcmin=64., px_res_arcmin=0.5)

# Configure data loader
os.makedirs(data_dir, exist_ok=True)
os.makedirs(sehgal_dir, exist_ok=True)
SDN = transforms.SehgalDataNormalizerScaledLogZShrink(norm_info_file)
SC = transforms.SehgalSubcomponets(compt_idxes)
RF = transforms.RandomFlips(p_v=0.5, p_h=0.5)

#SDS_train = datasets.SehgalDataSet(sehgal_dir, "train141020", transforms=[SDN, RF, SC], dummy_label=True, dtype=np.float64,
#                         shape=shape)

SDS_train = datasets.SehgalDataSet(sehgal_dir, "train281220_fromcat", transforms=[SDN, RF, SC], dummy_label=True, dtype=np.float64,
                         shape=shape)

dataloader = torch.utils.data.DataLoader(
    SDS_train,
    batch_size=batch_size,
    shuffle=True,
)


LF = cnn.LinearFeature(5,5, bias=True)
STanh = cnn.ScaledTanh(15., 2./15.)
experiment_id = "63101414f2434a3e93cd28663a89f879"
model_dir = "/home/dwhan89/workspace/cosmikyu/output/sehgal_dcganwgp_211020/{}/model".format(experiment_id)
DCGAN_WGP = gan.DCGAN_WGP("sehgal_dcganwgp_281220", shape, latent_dim, nconv_fcgen=64,
                          nconv_fcdis=64, cuda=cuda, ngpu=4, nconv_layer_gen=5, nconv_layer_disc=5, kernal_size=4, stride=2,
                          padding=1, output_padding=0, gen_act=[LF, STanh])
#DCGAN_WGP.load_states(model_dir, "_{}".format(0))
mlflow.set_experiment(DCGAN_WGP.identifier)
print(DCGAN_WGP.experiment.experiment_id)
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
        lr=1e-4,
        betas=(0.5, 0.9),
        lambda_gp=10.,
        lambda_l1=0.
    )

