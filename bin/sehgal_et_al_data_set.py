from orphics import sehgal, maps
import healpy as hp
from pixell import utils, enmap, curvedsky, enplot, wcsutils
import os
import numpy as np
import matplotlib.pyplot as plt

sehgal_dir = "/home/dwhan89/scratch/data/sehgal_et_al_sims/cosmo_sim_maps/July2009/inputs/microwaveSky"
SEHGAL10 = sehgal.SehgalSky2010(path=sehgal_dir, data_type='alm')
rfs = {"kappa": lambda : SEHGAL10.get_kappa(),
       "ksz": lambda : SEHGAL10.get_ksz(148),
       "tsz": lambda :SEHGAL10.get_tsz(148),
       "rad_pts": lambda :SEHGAL10.get_radio(148),
       "ir_pts": lambda :SEHGAL10.get_cib(148)
      }
lmax = 8000
shape, wcs = enmap.fullsky_geometry(res=0.5*utils.arcmin)


output_dir = "/home/dwhan89/scratch/data/sehgal_et_al_sims/cosmo_sim_maps/July2009/output/rotated_scaled"
def output_path(x):
    return os.path.join(output_dir, x)

test = {}
rot_angles1 = [0, 15, 30, 45, 60, 75]
rot_angles2 = [0, 20, 40, 60, 80]
for compt_idx in ["kappa"]:
    print(compt_idx)   
    alm_ref = np.complex128(rfs[compt_idx]())
    for rot_angle1 in rot_angles1:
        for rot_angle2 in rot_angles2:
            #if rot_angle1 == 0 and rot_angle2 != 0: continue
            file_path = output_path("%s_{}_%s_%s_000.fits")
            file_path = file_path %(compt_idx, "%0.3d"%rot_angle1, "%0.3d"%rot_angle2)
            print(file_path)
            alm_path = file_path.format("alm")
            enmap_path = file_path.format("enmap")
            if not os.path.exists(alm_path):
                print("rotating")
                alm = alm_ref.copy()
                if rot_angle2 != 0:
                    hp.rotate_alm(alm, rot_angle1*utils.degree, rot_angle2*utils.degree, 0, lmax=lmax)
                elif rot_angle2 == 0 and rot_angle1 != 0:
                    continue
                else:
                    pass
                hp.fitsfunc.write_alm(alm_path, alm, overwrite=True)
            else:
                alm = hp.read_alm(alm_path, hdu=(1))
            alm = np.complex128(alm)

            if not os.path.exists(enmap_path) and False:
                print("alm2map")
                emap = curvedsky.alm2map(alm, enmap.zeros(shape, wcs))
                #test[(rot_angle1, rot_angle2)] = emap.copy()
                enmap.write_fits(enmap_path, emap)
                del emap

            del alm
    

for compt_idx in ["ksz", "tsz", "ir_pts", "rad_pts"]:
    print(compt_idx) 
    alm_ref = np.complex128(rfs[compt_idx]())
    for rot_angle1 in rot_angles1:
        for rot_angle2 in rot_angles2:
            #if rot_angle1 == 0 and rot_angle2 != 0: continue
            file_path = output_path("148_%s_{}_%s_%s_000.fits")
            file_path = file_path %(compt_idx, "%0.3d"%rot_angle1, "%0.3d"%rot_angle2)
            print(file_path)
            alm_path = file_path.format("alm")
            enmap_path = file_path.format("enmap")
            if not os.path.exists(alm_path):
                print("rotating")
                alm = alm_ref.copy()
                if rot_angle2 != 0:
                    hp.rotate_alm(alm, rot_angle1*utils.degree, rot_angle2*utils.degree, 0, lmax=lmax)
                elif rot_angle2 == 0 and rot_angle1 != 0:
                    continue
                else:
                    pass
                hp.fitsfunc.write_alm(alm_path, alm, overwrite=True)
            else:
                alm = hp.read_alm(alm_path, hdu=(1))
            alm = np.complex128(alm)

            if not os.path.exists(enmap_path) and False:
                print("alm2map")
                emap = curvedsky.alm2map(alm, enmap.zeros(shape, wcs))
                #test[(rot_angle1, rot_angle2)] = emap.copy()
                enmap.write_fits(enmap_path, emap)
                del emap

            del alm

