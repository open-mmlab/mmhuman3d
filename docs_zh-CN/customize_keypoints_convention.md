## 客制化关键点类型

### 总览

如果使用的数据集在MMHuman3D中不受支持，您可以根据这个文档，添加一个新的关键点类型。

下列是MMHuman3D中支持的关键点类型:
  - agora
  - coco
  - coco_wholebody
  - crowdpose
  - h36m
  - human_data
  - hybrik
  - lsp
  - mpi_inf_3dhp
  - mpii
  - openpose
  - penn_action
  - posetrack
  - pw3d
  - smpl
  - smplx


**1. 创建新的关键点类型**

请根据
[`mmhuman3d/core/conventions/keypoints_mapping/human_data.py`](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/core/conventions/keypoints_mapping/human_data.py) , 创建一个名为 `NEW_CONVENTION.py` 的新文件。
该文件中, `NEW_KEYPOINTS` 是一个包含关键点名字和具体顺序的列表。

例如, 想要为`AGORA`数据集创建一个新的关键点类型, `agora.py` 文件应该包含以下项目:
```
AGORA_KEYPOINTS = [
  'pelvis',
  'left_hip',
  'right_hip'
  ...
]
```

**2. 寻找`human_data`中的关键点名字**

在MMHuman3D中，不同数据集中具有相同名字的关键点应该对应相同的人体部件。`human_data`类型已经整合了支持的数据集中不同名字的关键点和对应关系。

对于`NEW_KEYPOINTS`中的每一个关键点, 需要进行如下检查:

(1)关健点名称是否存在于[`mmhuman3d/core/conventions/keypoints_mapping/human_data.py`](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/core/conventions/keypoints_mapping/human_data.py);

(2)关键点是否有相对于`human_data`中相同位置关键点的映射关系。

如果两个条件都满足, 请在`NEW_CONVENTION.py`中保留关键点的名字。


**3. 寻找`human_data`中关键点的对应关系**

如果`NEW_KEYPOINTS`中的关键点与`human_data`中不同名字的关键点有相同的对应关系, 例如`NEW_KEYPOINTS`中的`head`对应于`human_data`中的`head_extra`, 将`NEW_KEYPOINTS`中的关键点按照`human_data`的规范重命名, 例如`head` -> `head_extra`。

**4. 在`human_data`中添加新的关键点**

如果新的关键点与`human_data`中的关键点没有对应关系，就需要将其罗列出来，并且在原始名称上加上前缀以作区分，例如`spine_3dhp`。

如有需要的话，我们会对`human_data`进行拓展以匹配新的关键点，但这必须在检查新的关键点没有对应关系，且没有命名冲突的情况下完成。

**5. 初始化新的关键点类型集合**

在[`mmhuman3d/core/conventions/keypoints_mapping/__init__.py`](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/core/conventions/keypoints_mapping/__init__.py#L8-25)中添加`NEW_CONVENTION.py`的`import`, 并在[KEYPOINTS_FACTORY](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/core/conventions/keypoints_mapping/__init__.py#L27-52)的字典中添加标识符。

例如, 我们所添加的新的关键点类型为`agora`:
```
# 添加 import
from mmhuman3d.core.conventions.keypoints_mapping import (
    agora,
    ...
)

# 添加到 factory
KEYPOINTS_FACTORY = {
    'agora': agora.AGORA_KEYPOINTS,
    ...
}
```

**6. 将新的关键点类型用于关键点转换**

想要将现有的关键点类型转换为新的类型，可以使用[`mmhuman3d/core/conventions/keypoints_mapping/__init__.py`](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/core/conventions/keypoints_mapping/__init__.py)中的`convert_kps`函数。该函数会产生一个由0,1构成的mask，指示是否应该滤除或者保留。

将`coco`关键点转换为新的关键点:
```
  new_kps, mask = convert_kps(smplx_keypoints, src='coco', dst='NEW_CONVENTION')
```

将新的关键点转换为`human_data`:
```
  new_kps, mask = convert_kps(smplx_keypoints, src='NEW_CONVENTION', dst='human_data')
```
