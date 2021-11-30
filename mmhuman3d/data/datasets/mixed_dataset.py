from typing import Optional, Union

import numpy as np
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler

from .builder import DATASETS, build_dataset


@DATASETS.register_module()
class MixedDataset(Dataset):
    """Mixed Dataset.

    Args:
        config (list): the list of different datasets.
        partition (list): the ratio of datasets in each batch.
        num_data (int | None, optional): if num_data is not None, the number
            of iterations is set to this fixed value. Otherwise, the number of
            iterations is set to the maximum size of each single dataset.
            Default: None.
    """

    def __init__(self,
                 configs: list,
                 partition: list,
                 num_data: Optional[Union[int, None]] = None):
        """Load data from multiple datasets."""
        assert min(partition) >= 0
        datasets = [build_dataset(cfg) for cfg in configs]
        self.dataset = ConcatDataset(datasets)
        if num_data is not None:
            self.length = num_data
        else:
            self.length = max(len(ds) for ds in datasets)
        weights = [
            np.ones(len(ds)) * p / len(ds)
            for (p, ds) in zip(partition, datasets)
        ]
        weights = np.concatenate(weights, axis=0)
        self.sampler = WeightedRandomSampler(weights, 1)

    def __len__(self):
        """Get the size of the dataset."""
        return self.length

    def __getitem__(self, idx):
        """Given index, sample the data from multiple datasets with the given
        proportion."""
        idx_new = list(self.sampler)[0]
        return self.dataset[idx_new]
