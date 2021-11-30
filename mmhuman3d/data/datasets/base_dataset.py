import copy
from abc import ABCMeta, abstractmethod
from typing import Optional, Union

from torch.utils.data import Dataset

from .pipelines import Compose


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base dataset.

    Args:
        data_prefix (str): the prefix of data path.
        pipeline (list): a list of dict, where each element represents
            a operation defined in `mmhuman3d.datasets.pipelines`.
        ann_file (str | None, optional): the annotation file. When ann_file is
            str, the subclass is expected to read from the ann_file. When
            ann_file is None, the subclass is expected to read according
            to data_prefix.
        test_mode (bool): in train mode or test mode. Default: None.
        dataset_name (str | None, optional): the name of dataset. It is used
            to identify the type of evaluation metric. Default: None.
    """

    def __init__(self,
                 data_prefix: str,
                 pipeline: list,
                 ann_file: Optional[Union[str, None]] = None,
                 test_mode: Optional[bool] = False,
                 dataset_name: Optional[Union[str, None]] = None):
        super(BaseDataset, self).__init__()

        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        if dataset_name is not None:
            self.dataset_name = dataset_name

        self.load_annotations()

    @abstractmethod
    def load_annotations(self):
        """Load annotations from ``ann_file``"""
        pass

    def prepare_data(self, idx: int):
        """"Prepare raw data for the f'{idx'}-th data."""
        results = copy.deepcopy(self.data_infos[idx])
        results['dataset_name'] = self.dataset_name
        results['sample_idx'] = idx
        return self.pipeline(results)

    def __len__(self):
        """Return the length of current dataset."""
        return self.num_data

    def __getitem__(self, idx: int):
        """Prepare data for the ``idx``-th data.

        As for video dataset, we can first parse raw data for each frame. Then
        we combine annotations from all frames. This interface is used to
        simplify the logic of video dataset and other special datasets.
        """
        return self.prepare_data(idx)
