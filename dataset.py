import h5py
import numpy as np

import torch
from torch.utils import data


class HDF5Dataset(data.Dataset):
    # Class to read an hdf5 file as pytorch dataset

    def __init__(self, file_path):
        super(HDF5Dataset, self).__init__()
        # HDF5 dataset path
        self.file_path = file_path

        # Read file and groups
        with h5py.File(self.file_path, 'r') as h5_file:
            grp1 = h5_file['earthquake']['local']
            grp2 = h5_file['non_earthquake']['noise']
            self.traces_len = len(grp1)
            self.noise_len = len(grp2)

    def __len__(self):
        return self.traces_len + self.noise_len

    def __getitem__(self, item):
        with h5py.File(self.file_path, 'r') as h5_file:

            # If item is a noise trace
            if item >= self.traces_len:
                item -= self.traces_len
                grp = h5_file['non_earthquake']['noise']
                for idx, dts in enumerate(grp):
                    if idx == item:
                        # Return normalized trace and label
                        out = grp[dts][:]
                        return torch.from_numpy(out), torch.tensor([0])

            else:
                grp = h5_file['earthquake']['local']
                for idx, dts in enumerate(grp):
                    if idx == item:
                        out = grp[dts][:]
                        return torch.from_numpy(out), torch.tensor([1])


class NpyDataset(data.Dataset):
    # Class to read an npy dataset

    def __init__(self, file_path):
        super(NpyDataset, self).__init__()

        self.file_path = file_path
        self.dset = np.load(self.file_path)

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, item):

        data = self.dset[item]
        trace = data[:6000].astype(np.float32)
        label = data[-1].astype(np.float32)

        return torch.from_numpy(trace), torch.tensor([label])
