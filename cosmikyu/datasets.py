import os

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset


class SehgalDataSet(Dataset):
    """Dataset class to load Sehgal et al. dataset used in Han et. al 2020 GAN project
    """

    def __init__(self, dataset_root, data_identifier="train", transforms=[], dummy_label=False, dtype=np.float64,
                 shape=(5, 128, 128), subset=None):
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
        self.subset = subset

    def get_stat(self):
        return self.lmdb_env.stat()

    def __len__(self):
        nentries = self.get_stat()['entries']
        return len(self.subset) if self.subset is not None and len(self.subset) < nentries else nentries

    def __getitem__(self, idx):
        cidx = self.subset[idx] if self.subset is not None else idx
        str_idx = '{:08}'.format(cidx)
        with self.lmdb_env.begin() as txn:
            data = np.frombuffer(txn.get(str_idx.encode('ascii')), dtype=self.dtype).reshape(self.shape).copy()
        for transform in self.transforms:
            data = transform(data)
        return [data, 0] if self.dummy_label else data


class CombiendDataSet(Dataset):
    def __init__(self, dataset_root, cmb_identifier, fg_identifier, fg_idxes=[1, 2, 3, 4], transforms=[],
                 dummy_label=False, dtype=np.float64,
                 shape=(3, 128, 128), subset=None):
        self.cmb_dataset = SehgalDataSet(dataset_root, cmb_identifier, [], dummy_label=False, dtype=np.float32,
                                         shape=(3, 128, 128), subset=None)
        self.fg_dataset = SehgalDataSet(dataset_root, fg_identifier, [], dummy_label=False, dtype=np.float64,
                                        shape=(5, 128, 128), subset=None)
        assert (len(self.cmb_dataset) == len(self.fg_dataset))

        self.fg_idxes = fg_idxes
        self.transforms = transforms
        self.dummy_label = dummy_label
        self.dtype = dtype
        self.shape = shape
        self.subset = subset

    def __len__(self):
        return len(self.subset) if self.subset is not None and len(self.subset) < len(self.cmb_dataset) else len(
            self.cmb_dataset)

    def __getitem__(self, idx):
        cidx = self.subset[idx] if self.subset is not None else idx
        data = self.cmb_dataset[cidx]

        data[0] += np.sum(self.fg_dataset[cidx][self.fg_idxes, ...], axis=0)

        for transform in self.transforms:
            data = transform(data)
        return [data, 0] if self.dummy_label else data


class SmoothedCMB(Dataset):
    def __init__(self, dataset_root, cmb_raw_identifier, cmb_smoothed_identifier, transforms=[],
                 dummy_label=False, dtype=np.float64, shape=(6, 128, 128), subset=None):

        self.smoothed_dataset = SehgalDataSet(dataset_root, cmb_smoothed_identifier, [], dummy_label=False, dtype=np.float32,
                                        shape=(3, 128, 128), subset=None)
        self.raw_dataset = SehgalDataSet(dataset_root, cmb_raw_identifier, [], dummy_label=False, dtype=np.float32,
                                         shape=(3, 128, 128), subset=None)
        assert (len(self.raw_dataset) == len(self.smoothed_dataset))

        self.transforms = transforms
        self.dummy_label = dummy_label
        self.dtype = dtype
        self.shape = shape
        self.subset = subset

    def __len__(self):
        return len(self.subset) if self.subset is not None and len(self.subset) < len(self.raw_dataset) else len(
            self.raw_dataset)

    def __getitem__(self, idx):
        cidx = self.subset[idx] if self.subset is not None else idx
        data = np.zeros(self.shape, dtype=self.dtype)
        data[:3,...] = self.smoothed_dataset[cidx]
        data[3:,...] = self.raw_dataset[cidx] - data[:3,...]

        for transform in self.transforms:
            data = transform(data)
        return [data, 0] if self.dummy_label else data

class DataSetJoiner(Dataset):
    def __init__(self, datasets=[], dummy_label=False, dtype=np.float64, shape=(10, 128, 128), shuffle=True,
                 transforms=[]):
        assert (len(datasets) > 0)
        self.nsample = len(datasets[0])
        for db in datasets:
            assert (self.nsample == len(db))
        print("Number of joined samples are {}".format(self.nsample))
        self.datasets = datasets
        self.dtype = dtype
        self.shape = shape
        self.dummy_label = dummy_label
        self.shuffle = shuffle
        self.transforms = transforms

    def __len__(self):
        return self.nsample

    def __getitem__(self, idx):
        data = np.zeros(self.shape, dtype=self.dtype)
        sidx = 0
        for db in self.datasets:
            if self.shuffle: idx = np.random.randint(self.nsample)
            sample = db[idx].astype(self.dtype)
            nchannel = sample.shape[0]
            eidx = sidx + nchannel
            data[sidx:eidx, ...] = sample.copy()
            sidx = eidx
            del sample
        assert (self.shape[0] == eidx)

        for transform in self.transforms:
            data = transform(data)
        return [data, 0] if self.dummy_label else data


class MapDataSet(Dataset):
    def __init__(self, dataset_root, data_type="train", transforms=[], dummy_label=False):
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
        data = torch.Tensor(self.storage[idx].copy())
        for transform in self.transforms:
            data = transform(data)
        return [data, 0] if self.dummy_label else data
