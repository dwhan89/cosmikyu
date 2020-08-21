from orphics import sehgal
import healpy as hp
from pixell import utils, enmap, curvedsky
import os
import numpy as np

sehgal_dir = "/home/dwhan89/scratch/data/sehgal_et_al_sims/cosmo_sim_maps/July2009/inputs/microwaveSky"
SEHGAL10 = sehgal.SehgalSky2010(path=sehgal_dir, data_type='alm')
rfs = {"kappa": lambda: SEHGAL10.get_kappa(),
       "ksz": lambda: SEHGAL10.get_ksz(148),
       "tsz": lambda: SEHGAL10.get_tsz(148),
       "rad_pts": lambda: SEHGAL10.get_radio(148),
       "ir_pts": lambda: SEHGAL10.get_cib(148)
       }
lmax = 10000
shape, wcs = enmap.fullsky_geometry(res=0.5 * utils.arcmin)

output_dir = "/home/dwhan89/scratch/data/sehgal_et_al_sims/cosmo_sim_maps/July2009/output//rotated_lmax10000_20mjycuts"


def output_path(x):
    return os.path.join(output_dir, x)


test = {}
rot_angles1 = [0, 15, 30, 45, 60, 75]
rot_angles2 = [0, 20, 40, 60, 80]
for compt_idx in ["kappa"]:
    continue
    print(compt_idx)
    alm_ref = np.complex128(rfs[compt_idx]())
    for rot_angle1 in rot_angles1:
        for rot_angle2 in rot_angles2:
            # if rot_angle1 == 0 and rot_angle2 != 0: continue
            file_path = output_path("%s_{}_%s_%s_000.fits")
            file_path = file_path % (compt_idx, "%0.3d" % rot_angle1, "%0.3d" % rot_angle2)
            print(file_path)
            alm_path = file_path.format("alm")
            enmap_path = file_path.format("enmap")
            if not os.path.exists(alm_path):
                print("rotating")
                alm = alm_ref.copy()
                if rot_angle2 != 0:
                    hp.rotate_alm(alm, rot_angle1 * utils.degree, rot_angle2 * utils.degree, 0, lmax=lmax)
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
                # test[(rot_angle1, rot_angle2)] = emap.copy()
                enmap.write_fits(enmap_path, emap)
                del emap

            del alm

for compt_idx in ["ksz", "tsz", "ir_pts", "rad_pts"]:
    continue
    print(compt_idx)
    alm_ref = np.complex128(rfs[compt_idx]())
    for rot_angle1 in rot_angles1:
        for rot_angle2 in rot_angles2:
            # if rot_angle1 == 0 and rot_angle2 != 0: continue
            file_path = output_path("148_%s_{}_%s_%s_000.fits")
            file_path = file_path % (compt_idx, "%0.3d" % rot_angle1, "%0.3d" % rot_angle2)
            print(file_path)
            alm_path = file_path.format("alm")
            enmap_path = file_path.format("enmap")
            if not os.path.exists(alm_path):
                print("rotating")
                alm = alm_ref.copy()
                if rot_angle2 != 0:
                    hp.rotate_alm(alm, rot_angle1 * utils.degree, rot_angle2 * utils.degree, 0, lmax=lmax)
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
                # test[(rot_angle1, rot_angle2)] = emap.copy()
                enmap.write_fits(enmap_path, emap)
                del emap

            del alm

# with the flux cut
SEHGAL10 = sehgal.SehgalSky2010(path=sehgal_dir, data_type='healpix')
rfs = {"kappa": lambda : SEHGAL10.get_kappa(),
       "ksz": lambda : SEHGAL10.get_ksz(148),
       "tsz": lambda :SEHGAL10.get_tsz(148),
       "rad_pts": lambda :SEHGAL10.get_radio(148),
       "ir_pts": lambda :SEHGAL10.get_cib(148)
      }

print("applying cuts")
NSIDE = 8192
flux_cut = 20*1e-3 # flux cut in Jy
lmax = 10000
for compt_idx in ["ir_pts", "rad_pts"]:
    file_format = output_path("148_%s_{}_%s_%s_000.fits")
    alm_ref = None
    for rot_angle1 in rot_angles1:
        for rot_angle2 in rot_angles2:
            file_path = file_format %(compt_idx, "%0.3d"%rot_angle1, "%0.3d"%rot_angle2)
            alm_path = file_path.format("alm")
            enmap_path = file_path.format("enmap")
            if not os.path.exists(alm_path):
                print(alm_path)
                if rot_angle1 == 0 and rot_angle2 == 0:
                    print("making flux cuts")
                    hpmap = rfs[compt_idx]()
                    area_strad = hp.pixelfunc.nside2pixarea(NSIDE)
                    final_cut = flux_cut/area_strad*sehgal.jysr2thermo(148)
                    loc = np.where(np.abs(hpmap) >= final_cut)
                    hpmap[loc] = 0.
                    alm = hp.map2alm(hpmap, lmax=lmax)
                    hp.fitsfunc.write_alm(alm_path, alm, overwrite=True)
                    continue
                    del hpmap
                elif alm_ref is None:
                    alm_ref_file = file_format %(compt_idx, "%0.3d"%0, "%0.3d"%0)
                    alm_ref_file = alm_ref_file.format("alm")
                    alm_ref = hp.read_alm(alm_ref_file, hdu=(1))
                
                print("rotating")
                alm = alm_ref.copy()
                if rot_angle2 != 0:
                    hp.rotate_alm(alm, rot_angle1*utils.degree, rot_angle2*utils.degree, 0, lmax=lmax)
                elif rot_angle2 == 0 and rot_angle1 != 0:
                    continue
                else:
                    pass
                hp.fitsfunc.write_alm(alm_path, alm, overwrite=True)
