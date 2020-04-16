import numpy as np
from pixell import enmap
from orphics import maps as omaps

class LogScale(object):
    def __call__(self, sample):
        data = sample['data']
        sign = np.sign(data)
        sample['data'] = np.nan_to_num(sign*np.log(np.abs(data)))
        return sample
    
    
class Normalize(object):
    def __init__(self, meta, scale='log'):
        assert(scale in ['linear', 'log'])       
        self.scale = scale
        self.meta = meta
    
    def __call__(self, sample):
        if self.scale == 'log':
            shift = self.meta[2]+1.
            sample['data'] = sample['data'] + shift
            sample['data'] = np.log(sample['data'])
            meta = self.meta + shift
            meta = np.log(meta)
        else:
            meta = self.meta
        
        mmean, mstd, mmax, mmin = meta    
        sample['data'] = (sample['data']-mmean)/(mmax-mmin)

        return sample  
    
class UnNormalize(Normalize):
    def __init__(self, meta, scale='log'):
        super().__init__(meta, scale)
    
    
    def __call__(self, sample):
        if self.scale == 'log':
            meta = self.meta + (self.meta[2]+1.)
            meta = np.log(meta)
        else:
            meta = self.meta
                
        mmean, mstd, mmax, mmin = meta 
        sample['data'] = sample['data']*(mmax-mmin)+mmean
        
        if self.scale == 'log':
            shift = self.meta[2]+1.
            sample['data'] = np.exp(sample['data'])
            sample['data'] = sample['data'] - shift
             
        return sample


class ToEnmap(object):
    def __call__(self, sample):
        sample['map'] = enmap.enmap(sample['data'], wcs=sample['wcs'])
        del sample['data'], sample['wcs']

        return sample

class DropWCS(object):
    def __call__(self, sample):
        if 'wcs' in sample: del sample['wcs']
        return sample

class Taper(object):
    def __init__(self, shape):
        self.shape = shape
        self.taper, _ = omaps.get_taper(shape, pad_percent=0.)
        loc = self.taper == 0
        self.taper[loc] = np.min(self.taper[~loc])
        
        
    def __call__(self, sample):
        assert('map' in sample)
        sample['map'] = sample['map']*self.taper
        return sample
    
class UnTaper(object):
    def __init__(self, shape):
        self.shape = shape
        self.taper, _ = omaps.get_taper(shape, pad_percent=0.)
        loc = self.taper == 0
        self.taper[loc] = np.min(self.taper[~loc])
        
    def __call__(self, sample):
        assert('map' in sample)
        sample['map'] = np.nan_to_num(sample['map']/self.taper)
        return sample
    
class TakePS(object):
    def __init__(self, bin_edges, shape, is_tapered=False, return_dl=True):
        self.bin_edges = bin_edges
        self.shape = shape
        self.taper, _ = omaps.get_taper(shape)
        loc = self.taper == 0
        self.taper[loc] = np.min(self.taper[~loc])
        self.is_tapered = False
        self.return_dl = return_dl
        
    def __call__(self, sample):
        emap = sample['map'].copy() if 'map' in sample else  enmap.enmap(sample['data'], wcs=sample['wcs'])
        if self.is_tapered:
            emap = np.nan_to_num(emap/self.taper)
        lbin, ps = omaps.binned_power(emap, self.bin_edges, mask=self.taper)
        ps = ps if not self.return_dl else ps*(lbin*(lbin+1)/(2*np.pi))
        
        sample['ps'] = (lbin, np.nan_to_num(ps))
        return sample
