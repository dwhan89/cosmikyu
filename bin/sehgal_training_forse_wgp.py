import os

import mlflow
import numpy as np
import torch
from orphics import maps

from cosmikyu import gan, config, datasets, transforms
from cosmikyu import nn as cnn

data_dir = config.default_data_dir
sehgal_dir = os.path.join(data_dir, 'sehgal')
cuda = True
compts = ["kappa", "ksz", "tsz", "ir_pts", "rad_pts"]
compt_idxes = [0, 1, 2, 3, 4]
shape = (5, 128, 128)
sample_interval = 500
save_interval = 1
batch_size = 32
nepochs = 50
mode = "full"
norm_info_file = "/home/dwhan89/workspace/cosmikyu/data/sehgal/281220_logz_normalization_info_validation.npz"
_, wcs = maps.rect_geometry(width_arcmin=64., px_res_arcmin=0.5)

# Configure data loader
os.makedirs(data_dir, exist_ok=True)
os.makedirs(sehgal_dir, exist_ok=True)
SDN = transforms.SehgalDataNormalizerScaledLogZShrink(norm_info_file)
RF = transforms.RandomFlips(p_v=0.5, p_h=0.5)

assert (mode in ["low", "high", "full"])
if mode == "full":
    primary_idx = None
    tertiary_idx = None
else:
    frac = 0.20
    primary_idx = np.load("/home/dwhan89/workspace/cosmikyu/data/sehgal/ps_dist_train010121_tszflux_sortedidx.npy")
    tertiary_idx = np.load(
        "/home/dwhan89/workspace/cosmikyu/data/sehgal/ps_dist_tertiary_train010121_tszflux_sortedidx.npy")
    nsample = int(len(primary_idx) * frac)
    if mode == "low":
        primary_idx = primary_idx[:-nsample]
        tertiary_idx = tertiary_idx[:-nsample]
    elif mode == "high":
        primary_idx = primary_idx[-nsample:]
        tertiary_idx = tertiary_idx[-nsample:]
    print(len(primary_idx))

SDS_target = datasets.SehgalDataSet(sehgal_dir, "train281220_fromcat", transforms=[SDN, RF], dummy_label=False,
                                    dtype=np.float64, subset=primary_idx)
SDS_input = datasets.SehgalDataSet(sehgal_dir, "train_tertiary281220_fromcat", transforms=[SDN, RF], dummy_label=False,
                                   dtype=np.float32, subset=tertiary_idx)

DSJ = datasets.DataSetJoiner([SDS_input, SDS_target], dtype=np.float64, shape=(10, 128, 128), dummy_label=True,
                             shuffle=True)

dataloader = torch.utils.data.DataLoader(
    DSJ,
    batch_size=batch_size,
    shuffle=True,
)

LF = cnn.LinearFeature(5, 5, bias=True)
STanh = cnn.ScaledTanh(15, 2 / 15)
nconv_layer = 5
experiment_id = "f833f5c803c148bd91d25de4f49cee81"
model_dir = "/home/dwhan89/workspace/cosmikyu/output/sehgal_forse_281220/{}/model".format(experiment_id)
FORSE = gan.VAEGAN_WGP("sehgal_forse_281220", shape, nconv_fcgen=64,
                       nconv_fcdis=64, cuda=cuda, ngpu=4, nconv_layer_gen=nconv_layer, nconv_layer_disc=nconv_layer,
                       kernal_size=4, stride=2,
                       padding=1, output_padding=0, gen_act=[LF, STanh], nin_channel=5, nout_channel=5,
                       nthresh_layer_gen=0, nthresh_layer_disc=0, dropout_rate=0)
FORSE.load_states(model_dir, "_{}".format(50))
mlflow.set_experiment(FORSE.identifier)
with mlflow.start_run(experiment_id=FORSE.experiment.experiment_id) as mlflow_run:
    torch.cuda.empty_cache()
    FORSE.train(
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
        lambda_l1=100.
    )
