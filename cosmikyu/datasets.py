from torch.utils.data import Dataset
import os
import numpy as np
import lmdb


class SehgalDataSet(Dataset):
    """Dataset class to load Sehgal et al. dataset used in Han et. al 2020 GAN project
    """
    def __init__(self, dataset_root, data_identifier="train", transforms=[], dummy_label=False, dtype=np.float64, shape=(5, 128, 128)):
        """

        Args:
            dataset_root (str): path to where dataset is located
            data_identifier (str): a string to identify each Sehgal data set (i.e "train", "test", etc)
            transforms (list): a sequence of transformation to be applied to this data set.
            dummy_label (boolean): pytorch Dataloader classes expect a label for each data sample. Set "true" to return a dummy label
            dtype (data type): a data type of sample
            shape (tuple): a shape of each sample
        """
        self.dataset_root = dataset_root
        self.dataset_dir = os.path.join(self.dataset_root, "sehgal_{}".format(data_identifier))
        self.shape = shape
        self.lmdb_env = lmdb.open(self.dataset_dir, readonly=True, lock=False)
        self.transforms = transforms
        self.dummy_label = dummy_label
        self.dtype = dtype

    def get_stat(self):
        return self.lmdb_env.stat()

    def __len__(self):
        return self.get_stat()['entries']

    def __getitem__(self, idx):
        str_idx = '{:08}'.format(idx)
        with self.lmdb_env.begin() as txn:
            data = np.frombuffer(txn.get(str_idx.encode('ascii')), dtype=self.dtype).reshape(self.shape).copy()
        for transform in self.transforms:
            data = transform(data)
        return [data, 0] if self.dummy_label else data


class DataSetJoiner(Dataset):
    def __init__(self, datasets=[], dummy_label=False, dtype=np.float64, shape=(10, 128, 128)):
        assert(len(datasets) > 0)
        self.nsample = 0
        for db in datasets:
            self.nsample = max(self.nsample, len(db))
        print("Number of joined samples are {}".format(self.nsample)) 
        self.datasets = datasets
        self.dtype = dtype
        self.shape = shape
        self.dummy_label = dummy_label

    def __len__(self):
        return self.nsample

    def __getitem__(self, idx):
        data = np.zeros(self.shape, dtype=self.dtype)
        sidx = 0
        for db in self.datasets:
            sample = db[idx].astype(self.dtype)
            nchannel = sample.shape[0]
            eidx = sidx + nchannel
            data[sidx:eidx, ...] = sample.copy()
            sidx = eidx
            del sample
        assert(self.shape[0] == eidx) 
        return [data, 0] if self.dummy_label else data

class MapDataSet(Dataset):
    def __init__(self, dataset_root, data_type="train", transforms=[], dummy_label=False):
        from os import listdir
        from skimage import io
        self.dataset_root = dataset_root
        self.dataset_file = os.path.join(self.dataset_root, "{}.npy".format(data_type))
        self.storage = np.load(self.dataset_file)
        self.nmaps = self.storage.shape[0]
        self.shape = (6, 600, 600)  # fixed for now
        self.transforms = transforms
        self.dummy_label = dummy_label
    
    def __len__(self):
        return self.nmaps

    def __getitem__(self, idx):
        data = self.storage[idx].copy()
        for transform in self.transforms:
            data = transform(data)
        return [data, 0] if self.dummy_label else data

