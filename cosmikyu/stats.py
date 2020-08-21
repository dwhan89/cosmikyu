from . import  mpi, config, utils
import numpy as np
import os

class STATS(object):

    def __init__(self, stat_identifier=None, output_dir =  None, overwrite=False, tag = 3235):
        self.output_dir  = output_dir
        if mpi.rank == 0 :
            print("[STATS] output_dir is %s" %self.output_dir)
            os.makedirs(self.output_dir, exist_ok=True)
        mpi.barrier()

        file_name        = "stats.npz" if not stat_identifier else "stats_%s.npz" %stat_identifier

        self.output_file  = os.path.join(self.output_dir, file_name)
        self.storage      = {}
        self.stats        = {}
        self.tag          = tag # can this be randomly assigned 
        if not overwrite: self.reload_data()

    def add_data(self, data_key, data_idx, data, safe=False):
        if not data_key in self.storage: self.storage[data_key] = {}

        if data_key in self.storage[data_key] and safe:
            raise ValueError("[STATS] already have %s" %((data_key,data_idx)))
        else:
            self.storage[data_key][data_idx] = data

    def collect_data(self, dest=0):
        print("[STATS] collecting data")

        if mpi.is_mpion():
            mpi.transfer_data(self.storage, self.tag, dest=dest, mode='merge')
        self.tag += 1

    def has_data(self, data_key, data_idx):
        has_data = data_key in self.storage
        return has_data if not has_data else data_idx in self.storage[data_key]

    def reload_data(self):
        ### passing all data through mpi is through. save it and reload it
        try:
            self.storage = utils.load_data(self.output_file)
            #self.storage = pickle.load(open(self.output_file, 'r'))
            if mpi.rank == 0: print("[STATS] loaded %s" %self.output_file)
        except:
            if mpi.rank == 0: print("[STATS] failed to reload data")

    def save_data(self, root=0, reload_st=True):
        self.collect_data()

        if mpi.rank == root:
            print("[STATS] saving %s from root %d" %(self.output_file, root))
            np.savez(self.output_file, **self.storage)
            #with open(self.output_file, 'w') as handle:
            #    pickle.dump(self.storage, handle)
        else: pass
        mpi.barrier()
        if reload_st: self.reload_data()

    def get_stats(self, subset_idxes=None, save_data=True):
        if save_data: self.save_data(reload_st=True)

        print("calculating stats")
        ret = {}
        for key in self.storage.keys():
            if subset_idxes is None:
                ret[key] = stats(np.array(list(self.storage[key].values())))
            else:
                ret[key] = stats(np.array(list(self.get_subset(subset_idxes, key, False).values())))

        self.stats = ret
        return ret

    def purge_data(self, data_idx, data_key=None):
        self.collect_data()

        def _purge(data_key, data_idx):
            print("[STATS] purging %s %d" %(data_idx, data_key))
            del self.storage[data_key][data_idx]

        if mpi.rank == 0:
            if data_key is not None:
                _purge(data_key, data_idx)
            else:
                for data_key in self.storage.keys():
                    _purge(data_key, data_idx)

        self.reload_data()

    def get_subset(self, data_idxes, data_key=None, collect_data=False):
        if collect_data: self.collect_data()

        def _collect_subset(data_key, data_idxes):
            return dict((k, v) for k, v in self.storage[data_key].iteritems() if k in data_idxes)

        ret = {}
        if data_key is not None:
            ret = _collect_subset(data_key, data_idxes)
        else:
            for data_key in self.storage.keys():
                ret[data_key] = _collect_subset(data_key, data_idxes)

        return ret

def stats(data, axis=0, ddof=1.):
    datasize = data.shape
    mean     = np.mean(data, axis=axis)
    cov      = np.cov(data.transpose(), ddof=ddof)
    cov_mean = cov/ float(datasize[0])
    corrcoef = np.corrcoef(data.transpose())
    std      = np.std(data, axis=axis, ddof=ddof) # use the N-1 normalization
    std_mean = std/ np.sqrt(datasize[0])

    return {'mean': mean, 'cov': cov, 'corrcoef': corrcoef, 'std': std, 'datasize': datasize\
            ,'std_mean': std_mean, 'cov_mean': cov_mean}

def chisq(obs, exp, cov_input, ddof=None, sidx=None, eidx=None, inv_corr=1.0):
    from scipy.stats import chi2
    diff  = obs-exp if not (exp == 0.).all() else obs.copy()
    cov  = cov_input.copy()

    if sidx is None: sidx = 0
    if eidx is None: eidx = len(diff)
    diff = diff[sidx:eidx]
    cov  = cov[sidx:eidx, sidx:eidx]

    norm = np.mean(np.abs(cov))
    cov  /= norm
    diff /= np.sqrt(norm)

    print(inv_corr)
    cov_inv = np.linalg.pinv(cov)*inv_corr
    chisq = np.dot(cov_inv, diff)
    chisq = np.dot(diff.T, chisq)

    if ddof is None: ddof = len(diff)
    p     = chi2.sf(chisq, ddof)

    return(chisq, p)

def reduced_chisq(obs, exp, cov, ddof_cor=0., sidx=None, eidx=None, inv_corr=1.):
    if sidx is None: sidx = 0
    if eidx is None: eidx = len(obs)
    obs = obs[sidx:eidx]
    exp = exp[sidx:eidx]
    cov = cov[sidx:eidx, sidx:eidx]

    ddof     = len(obs)# ddof
    ddof_cor = float(ddof_cor)
    ddof     = ddof - ddof_cor

    _chisq, p = chisq(obs,exp,cov,ddof, sidx=None, eidx=None, inv_corr=inv_corr)
    rchisq    = _chisq / ddof
    print("DDOF: ",ddof)
    return(rchisq, p)


class MultBinner(object):
    def __init__(self, bin_edges, nchannels):
        self.binners = [None]*nchannels
        self.nchannels = nchannels
        if type(bin_edges) == type([]):
            assert(len(bin_edges) == components)
            for i in range(nchannels):
                self.binners[i] = BINNER(bin_edges[i])
        else:
            for i in range(nchannels):
                self.binners[i] = BINNER(bin_edges)

    def bin(self, arr, right=True):
        assert(arr.shape[0] == self.nchannels)
        for i in range(self.nchannels):
            self.binners[i].bin(arr[i], right=right)

    def get_info(self):
        ret = {}
        for i in range(self.nchannels):
            ret[i] = {"bin_centers":self.binners[i].bin_center,
                    "hist":self.binners[i].storage,
                    "bin_edges":self.binners[i].bin_edges} 
        return ret


class BINNER(object):
    def __init__(self, bin_edges):
        bin_lower      = bin_edges[:-1].copy()
        bin_upper      = bin_edges[1:].copy()
        bin_lower = bin_lower[:len(bin_upper)]

        self.bin_edges  = bin_edges
        self.bin_lower  = bin_lower
        self.bin_upper  = bin_upper
        self.bin_center = (bin_lower + bin_upper)/2.
        self.bin_sizes  = bin_upper - bin_lower + 1
        self.nbin       = len(bin_lower)
        self.storage   = np.zeros(len(self.bin_center))

        assert((self.bin_sizes > 0)).all()


    def bin(self, arr, right=True):
        digitized = np.digitize(arr.flatten(), self.bin_edges, right=right) 
        for i in range(self.nbin):
            self.storage[i] += np.sum(digitized == i)
        return (self.bin_center, self.storage)
