import numpy as np
from torch.utils.data import Dataset

from .builder import DATASETS, build_dataset


@DATASETS.register_module()
class AdversarialDataset(Dataset):
    """Mix Dataset for the adversarial training in 3D human mesh estimation
    task.

    The dataset combines data from two datasets and
    return a dict containing data from two datasets.
    Args:
        train_dataset (Dataset): Dataset for 3D human mesh estimation.
        adversarial_dataset (Dataset): Dataset for adversarial learning,
            provides real SMPL parameters.
    """

    def __init__(self, train_dataset, adv_dataset):
        super().__init__()
        self.train_dataset = build_dataset(train_dataset)
        self.adv_dataset = build_dataset(adv_dataset)
        self.length = len(self.train_dataset)

    def __len__(self):
        """Get the size of the dataset."""
        return self.length

    def __getitem__(self, i):
        """Given index, get the data from train dataset and randomly sample an
        item from adversarial dataset.

        Return a dict containing data from train and adversarial dataset.
        """
        data = self.train_dataset[i]
        ind_adv = np.random.randint(
            low=0, high=len(self.adv_dataset), dtype=int)
        num_adv_data = len(self.adv_dataset)
        adv_data = self.adv_dataset[ind_adv % num_adv_data]
        for k, v in adv_data.items():
            data['adv_' + k] = v
        return data
