## 向mmhuman3d中添加数据转换

### 总览

所有的数据集在加载之前，需要先使用不同的转换脚本，将其预处理为相同的结构。该文档描述了如何给新的数据集编写转换工具。

### 1. 首先给转换工具创建python脚本

在下面的例子中, 我们给LSP数据集创建转换脚本`mmhuman3d/data/data_converters/lsp.py`。

我们继承了两种类型的转换工具:

(1) BaseConverter (详情见`mmhuman3d/data/data_converters/coco.py`)

CocoConverter 使用 `convert` 函数输出处理过后的单个`.npz`文件。
```
@DATA_CONVERTERS.register_module()
class CocoConverter(BaseConverter):

    def convert(self, dataset_path: str, out_path: str) -> dict:
        """
        参数:
            dataset_path (str): 存放原始图像和标注的文件夹路径。
            out_path (str): 保存处理过的`.npz`文件的路径

        返回值:
            dict:
                一个基于HumanData()结构的字典，包含了
                image_path, bbox_xywh, keypoints2d,
                keypoints2d_mask等关键字。
        """
```

(2) BaseModeConverter (详情见`mmhuman3d/data/data_converters/lsp.py`)

如果需要对训练集和测试集进行不同的操作，您可以继承`BaseModeConverter`。 例如，`LspConverter`通过`convert_by_mode`函数输出多个定义在`ACCEPTED_MODES`中, 经过处理的`.npz`文件。
```
@DATA_CONVERTERS.register_module()
class LspConverter(BaseModeConverter):

    """
    Args:
        modes (list): 'test' 或者 'train'
    """
    ACCEPTED_MODES = ['test', 'train']

    def __init__(self, modes: List = []) -> None:
        super(LspConverter, self).__init__(modes)

    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:

```

请参考[keypoints conventions](https://github.com/open-mmlab/mmhuman3d/blob/main/docs/keypoints_convention.md)查看您的数据集是否已经拥有现成的convention。没有的话, 您可以按照[customize_keypoints_convention](https://github.com/open-mmlab/mmhuman3d/blob/main/docs/customize_keypoints_convention.md)的指导定义一个新的convention。

```
# 根据给定的convention储存关键点
keypoints2d_, mask = convert_kps(keypoints2d_, 'lsp', 'human_data')

```

MMHuman3D使用 `HumanData` 结构来储存和加载数据。 您可以在[human_data](https://github.com/open-mmlab/mmhuman3d/blob/main/docs/human_data.md)中找到对其功能进一步的描述。

```
    # 使用 HumanData 储存数据
    human_data = HumanData()

    ...

    # 存储必要的关键字，例如: image path, bbox, keypoints2d, keypoints2d_mask, keypoints3d, keypoints3d_mask
    human_data['image_path'] = image_path_
    human_data['bbox_xywh'] = bbox_xywh_
    human_data['keypoints2d_mask'] = mask
    human_data['keypoints2d'] = keypoints2d_
    human_data['config'] = 'lsp'
    human_data.compress_keypoints_by_mask()

    # 储存数据结构
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    out_file = os.path.join(out_path, 'lsp_{}.npz'.format(mode))
    human_data.dump(out_file)
```

### 2. 在`mmhuman3d/data/data_converters/__init__.py`中初始化转换工具

将您的转换工具注册在data_converters中:

```python
# 引用您的转换工具
from .lsp import LspConverter

# 将您的转换工具添加在如下列表中
__all__ = [
    'build_data_converter', 'AgoraConverter', 'MpiiConverter', 'H36mConverter', ...
    'LspConverter'
]
```


### 3. 将您的数据集的配置文件添加在`mmhuman3d/tools/convert_datasets.py`的`DATASET-CONFIGS`中

现有数据集的配置文件罗列在[convert_datasets.py](https://github.com/open-mmlab/mmhuman3d/tree/main/tools/convert_datasets.py)中。

例如:
```
DATASET_CONFIGS = dict(
    ...
    lsp=dict(type='LspConverter', modes=['train', 'test'], prefix='lsp')
)
```

其中 `lsp` 是 `dataset-name`的一个实例。 可使用的模式包括 `train` 和 `test`, `prefix` 指定了数据集文件夹的名称。 在该例中,  `prefix = 'lsp'` 是存放了原始标注和图像的文件夹(文件夹的具体结构可以参考[preprocess_dataset](https://github.com/open-mmlab/mmhuman3d/blob/main/docs/preprocess_dataset.md#lsp)).


### 4. 将您的数据集的license和相应的文件夹结构添加到`preprocess_dataset.md`中

`lsp.py`中引用数据集的例子:
- https://github.com/open-mmlab/mmhuman3d/blob/9ec38db89cb896c318ff830c12ec007f60c447ad/mmhuman3d/data/data_converters/lsp.py#L15-L22

下载链接和文件夹结构放置在 `preprocess_dataset.md`
- https://github.com/open-mmlab/mmhuman3d/blob/main/docs/preprocess_dataset.md#lsp


### 5. 检查转换工具是否正常工作


运行如下脚本进行检查

```bash
python tools/convert_datasets.py \
  --datasets lsp \ # 定义在`DATASET_CONFIGS`中的数据集名称
  --root_path data/datasets \
  --output_path data/preprocessed_datasets
```

获得放置在 `data/preprocessed_datasets` 下的`.npz`文件:

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
