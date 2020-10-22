from cosmikyu import gan, config, datasets, transforms, utils
from cosmikyu import nn as cnn
import os
import torch
import mlflow
from orphics import maps

data_dir = config.default_data_dir
sehgal_dir = os.path.join(data_dir, 'sehgal')
cuda = True
compts = ["kappa", "ksz", "tsz", "ir_pts", "rad_pts"]
compt_idxes = [0, 1, 2, 3, 4]
shape = (len(compt_idxes), 128, 128)
latent_dim = 256
sample_interval = 1600
save_interval = 5
batch_size = 32
nepochs = 100
norm_info_file = "/home/dwhan89/workspace/cosmikyu/data/sehgal/201020_logzshrink_normalization_info_validation.npz"


_, wcs = maps.rect_geometry(width_arcmin=64., px_res_arcmin=0.5)

# Configure data loader
os.makedirs(data_dir, exist_ok=True)
os.makedirs(sehgal_dir, exist_ok=True)
SDN = transforms.SehgalDataNormalizerScaledLogZShrink(norm_info_file)
SC = transforms.SehgalSubcomponets(compt_idxes)
RF = transforms.RandomFlips(p_v=0.5, p_h=0.5)
SDS_train = datasets.SehgalDataSet(sehgal_dir, data_type="train141020", transforms=[SDN, RF, SC], dummy_label=True)

dataloader = torch.utils.data.DataLoader(
    SDS_train,
    batch_size=batch_size,
    shuffle=True,
)


STanh = cnn.ScaledTanh(30., 2./30.)
experiment_id = "c0eb302516f64c87adf202830677a0da"
model_dir = "/home/dwhan89/workspace/cosmikyu/output/sehgal_dcganwgp_211020/{}/model".format(experiment_id)
DCGAN_WGP = gan.DCGAN_WGP("sehgal_dcganwgp_211020", shape, latent_dim, nconv_fcgen=64,
                          nconv_fcdis=64, cuda=cuda, ngpu=4, nconv_layer_gen=5, nconv_layer_disc=5, kernal_size=5, stride=2,
                          padding=2, output_padding=1, gen_act=STanh)
#DCGAN_WGP.load_states(model_dir, "_{}".format(25))
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
        lr=1e-4,
        betas=(0.5, 0.9),
        lambda_gp=10.
    )

