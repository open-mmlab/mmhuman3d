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
        train_dataset (:obj:`Dataset`): Dataset for 3D human mesh estimation.
        adv_dataset (:obj:`Dataset`): Dataset for adversarial learning.
    """

    def __init__(self, train_dataset: Dataset, adv_dataset: Dataset):
        super().__init__()
        self.train_dataset = build_dataset(train_dataset)
        self.adv_dataset = build_dataset(adv_dataset)
        self.num_train_data = len(self.train_dataset)
        self.num_adv_data = len(self.adv_dataset)

    def __len__(self):
        """Get the size of the dataset."""
        return self.num_train_data

    def __getitem__(self, idx: int):
        """Given index, get the data from train dataset and randomly sample an
        item from adversarial dataset.

        Return a dict containing data from train and adversarial dataset.
        """
        data = self.train_dataset[idx]
        adv_idx = np.random.randint(low=0, high=self.num_adv_data, dtype=int)
        adv_data = self.adv_dataset[adv_idx]
        for k, v in adv_data.items():
            data['adv_' + k] = v
        return data
