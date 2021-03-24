import numpy as np


def smooth(x, window_len=11, window='hanning'):
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def get_geometry(ra, dec, width, height, res=None, shape=None):
    if shape is None:
        nx = int(np.ceil(width / res))
        ny = int(np.ceil(height / res))
        shape = (ny, nx)
    else:
        pass
    pos = [[-1. * height / 2. + dec, -1. * width / 2. + ra], [height / 2. + dec, width / 2. + ra]]
    shape, wcs = enmap.geometry(pos=pos, shape=shape)
    return shape, wcs, pos


def cl2dl(cl):
    l = np.arange(len(cl))
    dl = cl * (l * (l + 1)) / (2 * np.pi)
    return l, dl


def load_data(data_path):
    data = np.load(data_path, allow_pickle=True)
    return {key: data[key].item() for key in data}


def create_dict(*idxes):
    """
        create nested dictionary with the given idxes
    """

    height = len(idxes)
    output = {}
    stack = [output]

    for depth in range(height):
        stack_temp = []
        while len(stack) > 0:
            cur_elmt = stack.pop()
            for idx in idxes[depth]:
                cur_elmt[idx] = {}
                stack_temp.append(cur_elmt[idx])
        stack = stack_temp

    return output


def merge_dict(a, b, path=None, clean=True):
    """
    merges b into a
    ref: https://stackoverflow.com/questions/7204805/dictionaries-of-dictionaries-merge"
    """
    if path is None: path = []

    for key in b:
        if key in a:
            if isinstance(a[key], dict):
                if isinstance(b[key], dict):
                    merge_dict(a[key], b[key], path + [str(key)])
                else:
                    a[key] = b[key]
            elif isinstance(a[key], np.ndarray) and np.array_equal(a[key], b[key]):
                pass
            elif a[key] == b[key]:
                pass  # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]

    if clean: del b

    return a


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise TypeError("Can't convert 'str' object to 'boolean'")


def get_gaussian_beam(l, beam_fwhm):
    " return f"
    beam_fwhm = np.deg2rad(beam_fwhm / 60.)
    sigma = beam_fwhm / (2. * np.sqrt(2. * np.log(2)))
    f_ell = np.exp(-(l) ** 2. * sigma ** 2. / 2)
    return l, f_ell


def car2hp_coords(pos):
    ret = np.zeros(pos.shape)
    ret[:, 0] = pos[:, 0] + np.pi / 2.
    ret[:, 1] = pos[:, 1] + np.pi
    return ret


def hp2car_coords(pos):
    ret = np.zeros(pos.shape)
    ret[:, 0] = pos[:, 0] - np.pi / 2.
    ret[:, 1] = pos[:, 1] - np.pi
    return ret
