from torch.utils.data import Dataset
import os
import numpy as np
import lmdb


class SehgalDataSet(Dataset):
    def __init__(self, dataset_root, data_type="train", transforms=[], dummy_label=False):
        #assert (data_type in ["train", "test", "model"])
        self.dataset_root = dataset_root
        self.dataset_dir = os.path.join(self.dataset_root, "sehgal_{}".format(data_type))
        self.lmdb_env = lmdb.open(self.dataset_dir, readonly=True, lock=False)
        self.shape = (5, 128, 128)  # fixed for now
        self.transforms = transforms
        self.dummy_label = dummy_label

    def get_stat(self):
        return self.lmdb_env.stat()

    def __len__(self):
        return self.get_stat()['entries']

    def __getitem__(self, idx):
        str_idx = '{:08}'.format(idx)
        with self.lmdb_env.begin() as txn:
            data = np.frombuffer(txn.get(str_idx.encode('ascii'), np.float64)).reshape(self.shape).copy()
        for transform in self.transforms:
            data = transform(data)
        return [data, 0] if self.dummy_label else data
