## How to add a dataset converter to mmhuman3d

### Overview

All datasets are first preprocessed (using respective converters) into the same convention before they are loaded during training or testing. This documentation outlines how to add a new converter to support a new dataset.

### 1. Create a python script for your converter

For this example, we create `mmhuman3d/data/data_converters/lsp.py` for the LSP dataset we are adding.

We have two types of converters that we inherit from:

(1) BaseConverter (refer to `mmhuman3d/data/data_converters/coco.py` for an example)

CocoConverter has a `convert` function to output a single preprocessed .npz file.
```
@DATA_CONVERTERS.register_module()
class CocoConverter(BaseConverter):

    def convert(self, dataset_path: str, out_path: str) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where raw images and
            annotations are stored.
            out_path (str): Path to directory to save preprocessed npz file

        Returns:
            dict:
                A dict containing keys image_path, bbox_xywh, keypoints2d,
                keypoints2d_mask stored in HumanData() format
        """
```

(2) BaseModeConverter (refer to `mmhuman3d/data/data_converters/lsp.py` for an example)

If your dataset requires different handling (modes) for training and test set, you can inherit BaseModeConverter. For instance, LspConverter has a `convert_by_mode` function which outputs multiple preprocessed .npz file with different modes defined in ACCEPTED_MODES.
```
@DATA_CONVERTERS.register_module()
class LspConverter(BaseModeConverter):

    """
    Args:
        modes (list): 'test' and/or 'train' for accepted modes
    """
    ACCEPTED_MODES = ['test', 'train']

    def __init__(self, modes: List = []) -> None:
        super(LspConverter, self).__init__(modes)

    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:

```

Please refer to [keypoints conventions](https://github.com/open-mmlab/mmhuman3d/blob/main/docs/keypoints_convention.md) to see if your dataset has an existing convention. If not, you can define a new convention following [this documentation](https://github.com/open-mmlab/mmhuman3d/blob/main/docs/customize_keypoints_convention.md).


```
# store keypoints according to specified convention
keypoints2d_, mask = convert_kps(keypoints2d_, 'lsp', 'human_data')

```

Our data pipeline use `HumanData` structure for storing and loading. You can refer to further explanation of its functionalities [here](https://github.com/open-mmlab/mmhuman3d/blob/main/docs/human_data.md).

```
    # use HumanData to store all data
    human_data = HumanData()

    ...

    # store the necessary keys i.e. image path, bbox, keypoints2d, keypoints2d_mask, keypoints3d, keypoints3d_mask
    human_data['image_path'] = image_path_
    human_data['bbox_xywh'] = bbox_xywh_
    human_data['keypoints2d_mask'] = mask
    human_data['keypoints2d'] = keypoints2d_
    human_data['config'] = 'lsp'
    human_data.compress_keypoints_by_mask()

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    out_file = os.path.join(out_path, 'lsp_{}.npz'.format(mode))
    human_data.dump(out_file)
```

### 2. Initialise your converter in `mmhuman3d/data/data_converters/__init__.py`

Add your converter to the registry of data_converters:

```python
# import your converter
from .lsp import LspConverter

# add your converter to the list of converters
__all__ = [
    'build_data_converter', 'AgoraConverter', 'MpiiConverter', 'H36mConverter', ...
    'LspConverter'
]
```


### 3. Add your dataset configuration to `DATASET-CONFIGS` under `mmhuman3d/tools/convert_datasets.py`

The available dataset configurations are listed [here](https://github.com/open-mmlab/mmhuman3d/tree/main/tools/convert_datasets.py).

An example is
```
DATASET_CONFIGS = dict(
    ...
    lsp=dict(type='LspConverter', modes=['train', 'test'], prefix='lsp')
)
```

where `lsp` is an example of a `dataset-name`. The available modes are `train` and `test` and the prefix specifies the name of the dataset folder. In this case, the prefix `lsp` is the name of the dataset folder containing the raw annotations and images (see example folder structure for LSP [here](https://github.com/open-mmlab/mmhuman3d/blob/main/docs/preprocess_dataset.md#lsp)).


### 4. Add your dataset license and recommended folder structure to `preprocess_dataset.md`

Example of dataset citation in `lsp.py`:
- https://github.com/open-mmlab/mmhuman3d/blob/9ec38db89cb896c318ff830c12ec007f60c447ad/mmhuman3d/data/data_converters/lsp.py#L15-L22

Example of dataset citation, download link and folder structure in `preprocess_dataset.md`
- https://github.com/open-mmlab/mmhuman3d/blob/main/docs/preprocess_dataset.md#lsp


### 5. Check that the converter works


Check that running this command

```bash
python tools/convert_datasets.py \
  --datasets lsp \ # dataset-name defined in DATASET_CONFIGS
  --root_path data/datasets \
  --output_path data/preprocessed_datasets
```

would allow us to obtain the preprocessed npz files under `data/preprocessed_datasets`:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    ├── datasets
    └── preprocessed_datasets
        ├── lsp_train.npz
        └── lsp_test.npz
```
