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

class TakePS(object):
    def __init__(self, bin_edges, shape):
        self.bin_edges = bin_edges
        self.shape = shape
        self.taper, _ = omaps.get_taper(shape)

    def __call__(self, sample):
        emap = sample['map'] if 'map' in sample else  enmap.enmap(sample['data'], wcs=sample['wcs'])
        


class ToEnmap(object):
    def __call__(self, sample):
        sample['map'] = enmap.enmap(sample['data'], wcs=sample['wcs'])
        del sample['data'], sample['wcs']

        return sample


