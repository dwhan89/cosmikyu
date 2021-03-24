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
compt_idxes = [0, 1,2,3,4]
shape = (len(compt_idxes), 128, 128)
sample_interval = 200
save_interval = 5
batch_size = 32
nepochs = 100
norm_info_file = "/home/dwhan89/workspace/cosmikyu/data/sehgal/281020_logzshrink_normalization_info_validation.npz"

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


STanh = cnn.ScaledTanh(1,1)
experiment_id = "c0eb302516f64c87adf202830677a0da"
model_dir = "/home/dwhan89/workspace/cosmikyu/output/sehgal_dcganwgp_211020/{}/model".format(experiment_id)
PIXGAN = gan.PIXGAN("sehgal_pixgan_261020", shape, nconv_fcgen=64,
                          nconv_fcdis=64, cuda=cuda, ngpu=4, nconv_layer_gen=4, nconv_layer_disc=3, kernal_size=4, stride=2,
                          padding=1, output_padding=0, gen_act=STanh, nin_channel=1, nout_channel=4, nthresh_layer_gen=3, nthresh_layer_disc=2)

mlflow.set_experiment(PIXGAN.identifier)
with mlflow.start_run(experiment_id=PIXGAN.experiment.experiment_id) as mlflow_run:
    torch.cuda.empty_cache()
    PIXGAN.train(
        dataloader,
        nepochs=nepochs,
        ncritics=1,
        sample_interval=sample_interval,
        save_interval=save_interval,
        load_states=True,
        save_states=True,
        verbose=True,
        mlflow_run=mlflow_run,
        lr=2e-4,
        betas=(0.5, 0.999),
        lambda_l1=100.
    )

