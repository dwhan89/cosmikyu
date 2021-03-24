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
shape = (5, 128, 128)
sample_interval = 500
save_interval = 1
batch_size = 32
nepochs = 100
norm_info_file = "/home/dwhan89/workspace/cosmikyu/data/sehgal/201020_logzshrink_normalization_info_validation.npz"
_, wcs = maps.rect_geometry(width_arcmin=64., px_res_arcmin=0.5)

# Configure data loader
os.makedirs(data_dir, exist_ok=True)
os.makedirs(sehgal_dir, exist_ok=True)
SDN = transforms.SehgalDataNormalizerScaledLogZShrink(norm_info_file)
RF = transforms.RandomFlips(p_v=0.5, p_h=0.5)
nsample = None#batch_size*20
SDS_target = datasets.SehgalDataSet(sehgal_dir, "train141020", transforms=[SDN, RF], dummy_label=False, dtype=np.float64, nsample=nsample)
SDS_input = datasets.SehgalDataSet(sehgal_dir, "train_tertiary191120", transforms=[SDN, RF], dummy_label=False, dtype=np.float32, nsample=nsample)
DSJ = datasets.DataSetJoiner([SDS_input,SDS_target],  dtype=np.float64, shape=(10, 128, 128), dummy_label=True)

dataloader = torch.utils.data.DataLoader(
    DSJ,
    batch_size=batch_size,
    shuffle=True,
)


LF = cnn.LinearFeature(5,5, bias=True)
STanh = cnn.ScaledTanh(30, 2/30)
experiment_id = "3d40c2845c214f46b560c16ea02f95a7"
#model_dir = "/home/dwhan89/workspace/cosmikyu/output/sehgal_pixganwgp_301020/{}/model".format(experiment_id)
model_dir = "/home/dwhan89/workspace/cosmikyu/output/sehgal_forse_081020/{}/model".format(experiment_id)
FORSE = gan.FORSE("sehgal_forse_201120", shape, nconv_fcgen=64,
                          nconv_fcdis=64, cuda=cuda, ngpu=4, nconv_layer_gen=5, nconv_layer_disc=3,
                          kernal_size=4, stride=2,
                          padding=1, output_padding=0, gen_act=[LF, STanh], nin_channel=5, nout_channel=5, nthresh_layer_gen=0, nthresh_layer_disc=1, dropout_rate=0)
#FORSE.load_states(model_dir, "_{}".format(5))
mlflow.set_experiment(FORSE.identifier)
with mlflow.start_run(experiment_id=FORSE.experiment.experiment_id) as mlflow_run:
    torch.cuda.empty_cache()
    FORSE.train(
        dataloader,
        nepochs=nepochs,
        ncritics=1,
        sample_interval=sample_interval,
        save_interval=save_interval,
        load_states=True,
        save_states=True,
        verbose=True,
        mlflow_run=mlflow_run,
        lr=1e-4,
        betas=(0.5, 0.9),
        lambda_l1=0.1
    )
