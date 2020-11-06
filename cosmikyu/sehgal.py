class Sehgal10Reprojected(object):

    def __init__(self):
        self.rot_angles1 = [0, 15, 30, 45, 60, 75]
        self.rot_angles2 = [0, 20, 40, 60, 80]


input_dir = "/home/dwhan89/scratch/data/sehgal_et_al_sims/cosmo_sim_maps/July2009/output/131020_hybrid_projection"
def input_path(rot_angle1, rot_angle2, compt_idx, dtype):
    freq_idx = "" if compt_idx == "kappa" else "148_"
    file_name = "%s%s_%s_%s_%s_000.fits" %(freq_idx, compt_idx, dtype, "%0.3d"%rot_angle1, "%0.3d"%rot_angle2)
    return os.path.join(input_dir, file_name)

def input_cat(rot_angle1, rot_angle2, compt_idx):
    freq_idx = "" if compt_idx == "kappa" else "148_"
    file_name = "%s%s_highflux_cat.npy"%(freq_idx, compt_idx)
    return os.path.join(input_dir, file_name)


output_dir = "/home/dwhan89/workspace/cosmikyu/data/sehgal"
def output_path(x):
    return os.path.join(output_dir, x)

highflux_cats = ["rad_pts", "ir_pts"]
def get_input_map(rot_angle1, rot_angle2, nshape, nwcs, compts=compts, highflux_cats=highflux_cats):
    ishape = nshape[-2:]
    ret = enmap.zeros(nshape, nwcs)
    for i, compt_idx in enumerate(compts):
        input_file = input_path(rot_angle1, rot_angle2, compt_idx, "alm")
        print("loading", input_file)
        alm = np.complex128(hp.read_alm(input_file, hdu=(1)))
        print(ishape, nshape)
        ret[i,...] = curvedsky.alm2map(alm, enmap.zeros(ishape, nwcs))
        del alm
        if compt_idx in highflux_cats:
            print("adding high flux cats")

            hiflux_cat = np.load(input_cat(rot_angle1, rot_angle2, compt_idx))
            hiflux_cat[:,:2] = pix2hp(hiflux_cat[:,:2])
            
            mat_rot, _, _ = hp.rotator.get_rotation_matrix((rot_angle1*utils.degree*-1,rot_angle2*utils.degree,0))
            uvec = hp.ang2vec(hiflux_cat[:,0],hiflux_cat[:,1])
            rot_vec = np.inner(mat_rot, uvec).T
            temppos = hp.vec2ang(rot_vec)
            rot_pos = np.zeros(hiflux_cat[:,:2].shape)
            rot_pos[:,0] = temppos[0]
            rot_pos[:,1] = temppos[1]
            rot_pos = hp2pix(rot_pos)
            del temppos
            rot_pix = np.round(enmap.sky2pix(nshape[-2:], nwcs , rot_pos.T).T).astype(np.int)
            loc = np.where((rot_pix[:,0]>=0)&(rot_pix[:,0]<nshape[-2])&(rot_pix[:,1]>=0.)&(rot_pix[:,1]<nshape[-1]))
            hiflux_cat = hiflux_cat[loc[0],2]
            rot_pix = rot_pix[loc[0],:]
            
            hiflux_map = enmap.zeros(nshape[-2:], nwcs)
            hiflux_map[rot_pix[:,0], rot_pix[:,1]] = hiflux_cat
            hiflux_map = hiflux_map/areamap
            ret[i,...] = ret[i,...]+hiflux_map
            del hiflux_map
            
    ftmap = enmap.fft(ret)
    _, f_ell = get_f_ell(modlmap.ravel())
    ftmap = ftmap*np.reshape(f_ell,(modlmap.shape))
    ret = enmap.ifft(ftmap).real; del ftmap
    return ret
