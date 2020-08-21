from pitas import power, flipper_tools
from orphics import maps as omaps
from pixell import enmap, curvedsky
import numpy as np
from cosmikyu import stats, mpi, datasets, config, utils
import os
from itertools import product
import healpy as hp
import matplotlib.pyplot as plt
import sys
import random

mpi.init(True)

nside = 10
nbatch = nside*nside

shape = (128, 128)
bshape = (128*nside,128*nside)
_, bwcs = omaps.rect_geometry(width_arcmin=64.*nside, px_res_arcmin=0.5)

def get_template(shape, wcs):
    return enmap.zeros(shape, wcs)

def get_mask(taper_percent=15.):
    taper, _ = omaps.get_taper(bshape, taper_percent=taper_percent, pad_percent=0.)
    loc = taper == 0
    taper[loc] = np.min(taper[~loc])
    temp = get_template(bshape, bwcs)
    temp[...] = taper
    return temp

mask = get_mask()

bin_edges = np.linspace(0,8000., 40)
PITAS = power.PITAS("082020_128x128taperv5", mask, mask, bin_edges=bin_edges, lmax=8000)

data_dir = config.default_data_dir
sehgal_dir = os.path.join(data_dir, 'sehgal')
stat_dir = os.path.join(sehgal_dir, "stats")
SDS_test = datasets.SehgalDataSet(sehgal_dir, data_type="testv3", transforms=[], dummy_label=False)

overwrite = False
STAT = stats.STATS("sehgal_cosmoganwgpv4_test", output_dir=stat_dir, overwrite=overwrite)
lmax = 8000
compts = ["kappa", "ksz", "tsz", "ir", "rad"]
nsample = len(SDS_test)/3
subtasks = mpi.taskrange(int(nsample/nbatch)-1)


def get_data(wcs=bwcs, nside=nside, dataset=SDS_test, taper=True):
    data = SDS_test[0].copy()
    nx, ny = shape
    nshape = (data.shape[0], nx*nside, ny*nside)
    temp = enmap.zeros(nshape, wcs)
    for i in range(nside):
        for j in range(nside):
            cidx = random.randint(0, len(SDS_test)-1)
            sx = i*nx
            sy = j*ny
            temp[:,sy:sy+ny,sx:sx+nx] = SDS_test[cidx][...]
    if taper:
        temp[...] *= get_mask()
    return temp

for sim_idx in subtasks:
    continue
    print(sim_idx)
    emap = get_data()

    alms = {}
    for i, compt_idx in enumerate(compts):
        alms[compt_idx] = curvedsky.map2alm(emap[i], lmax=lmax)
    del compt_idx

    for compt_idx1, compt_idx2 in product(compts, compts):
        compt_keys = [compt_idx1, compt_idx2]
        compt_keys.sort()
        stat_key = "dls_{}x{}".format(compt_keys[0], compt_keys[1])
        if STAT.has_data(stat_key, sim_idx) and not overwrite:
            continue
        else:
            sys.stdout.flush()
            cl = hp.alm2cl(alms[compt_idx1], alms[compt_idx2])
            l = np.arange(len(cl), dtype=np.float)
            dl = l*(l+1.)/(2*np.pi)*cl
            lbin, dlbin = PITAS.binner.bin(l, dl)
            dlbin = np.dot(PITAS.mcm_dltt_inv, dlbin)
            STAT.add_data("lbin", 0, lbin)
            STAT.add_data(stat_key, sim_idx, dlbin)

    del alms
STAT.get_stats()

DCGAN_WGP = gan.DCGAN_WGP("sehgal_dcganwgp", shape, latent_dim, cuda=False, nconv_fcgen=64,
                                  nconv_fcdis=64, ngpu=4, nconv_layer_gen=4, nconv_layer_disc=4, kernal_size=5, stride=2,
                                                            padding=2, output_padding=1)
DCGAN_WGP.load_states("/home/dwhan89/workspace/cosmikyu/output/sehgal_cosmoganwgpv3/5c7805d7066c419a8685a163871eb242/model")


