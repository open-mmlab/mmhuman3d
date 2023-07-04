## HumanData数据格式

### 总览

`HumanData`是Python内置字典的子类，主要用于存放包含人体的单视角图像的信息。它具有通用的基础结构，也兼容具有新特性的客制化数据。
原生的`HumanData`包含`numpy.ndarray`或其他的Python内置的数据结构，但不包含`torch.Tensor`的数据。可以使用`human_data.to()`将其转换为`torch.Tensor`(支持CPU和GPU)。

### `Key/Value`的定义：如下是对`HumanData`支持的`Key`和`Value`的描述.

#### 路径:

通常包含图片路径，如果数据集有提供额外的深度或者分割图，也可以记录下来。
- image_path: (N, ), 字符串组成的列表, 每一个元素是图像相对于根目录的路径。
- segmantation_path (可选): (N, ), 字符串组成的列表, 每一个元素是图像分割图相对于根目录的路径。
- depth_path (可选): (N, ), 字符串组成的列表, 每一个元素是图像深度图相对于根目录的路径。

#### 关键点：

以下关键点keys如果适用，则应包含在HumanData中。任何一个关键点的key，应存在一个mask，表示其中哪些关键点有效。如`keypoints3d_original`应对应`keypoints3d_original_mask`。
`HumanData` 中的关键点存储格式为`HUMAN_DATA`, 包含190个关键点。MMHuman3d中提供了很多常用关键点格式的转换（2d及3d均支持）, 详见 [keypoints_convention](../docs_zh-CN/keypoints_convention.md).
- keypoints3d_smpl / keypoints3d_smplx: (N, 190, 4), numpy array, `smplx / smplx`模型的3d关节点与置信度, 每一个数据集的关节点映射到了`HUMAN_DATA`的关节点。
- keypoints3d_original: (N, 190, 4), numpy array, 由数据集本身提供的3d关节点与置信度, 每一个数据集的关节点映射到了`HUMAN_DATA`的关节点。
- keypoints2d_smpl / keypoints2d_smplx: (N, 190, 3), numpy array, `smpl / smplx`模型的2d关节点与置信度, 每一个数据集的关节点映射到了`HUMAN_DATA`的关节点。
- keypoints2d_original: (N, 190, 3), numpy array, 由数据集本身提供的2d关节点与置信度, 每一个数据集的关节点映射到了`HUMAN_DATA`的关节点。
- （mask示例） keypoints2d_smpl_mask: (190, ), numpy array, 表示`keypoints2d_smpl`中关键点是否有效的掩膜。 0表示该位置的关键点在原始数据集中无法找到。

#### 检测框：

身体（smpl），手脸（smplx）的检测框，标注为`[x_min, y_min, width, height, confidence]`，且不应超出图片。
- bbox_xywh: (N, 5), numpy array, 边界框的置信度, 边界框左下角点的坐标x和y, 边界框的宽w和高h, 置信度得分放置在最后。
- face_bbox_xywh, lhand_bbox_xywh, rhand_bbox_xywh（可选）： (N, 5), numpy array, 如果数据标注中含有`smplx`, 则应包括这三个key，由smplx2d关键点得出，格式同上。

#### 人体模型参数：

通常以smpl/smplx格式存储。
- smpl: (1, ), 字典, `keys` 分别为 ['body_pose': numpy array, (N, 23, 3), 'global_orient': numpy array, (N, 3), 'betas': numpy array, (N, 10), 'transl': numpy array, (N, 3)].
- smplx: (1, ), 字典, `keys` 分别为 ['body_pose': numpy array, (N, 21, 3),'global_orient': numpy array, (N, 3), 'betas': numpy array, (N, 10), 'transl': numpy array, (N, 3), 'left_hand_pose': numpy array, (N, 15, 3), 'right_hand_pose': numpy array, (N, 15, 3), 'expression': numpy array (N, 10), 'leye_pose': numpy array (N, 3), 'reye_pose': (N, 3), 'jaw_pose': numpy array (N, 3)].

#### 其它keys

- config: (), 字符串, 单个数据集的配置的标志。
- meta: (1, ), 字典, `keys`为数据集中的各种元数据。
- misc: (1, ), 字典, `keys`为数据集中各种独特设定，也可以由用户自定义。`misc`占用的空间（可以通过`sys.getsizeof(misc)`获取）不能超过6MB。

#### `HumanData['misc']`中建议（可能）包含的内容:
Miscellaneous部分中包含了每个数据集的独特设定，包括相机种类，关键点标注来源，检测框来源，是否包含smpl/smplx标注等等，用于便利数据读取。
`HumanData['misc']`中包含一个dictionary，建议包含的key如下所示：
- kps3d_root_aligned： Bool 描述keypoints3d是否经过root align，建议不进行root_alignment，如果不包含这个key，则默认没有进行过root_aligenment
- flat_hand_mean：Bool 对于smplx标注的数据，应该存在此项，大多数数据集中`flat_hand_mean=False`
- bbox_source：描述检测框的来源，`bbox_soruce='keypoints2d_smpl' or 'keypoints2d_smplx' or 'keypoints2d_original'`，描述检测框是由哪种关键点得出的，或者`bbox_source='provide_by_dataset'`表示检测框由数据集直接给出（比如用其自带检测器生成而不是由关键点推导得出）
- bbox_body_scale: 如果检测框由关键点推导得出，则应包含此项，描述由smpl/smplx/2d_gt关键点推导出的身体检测框的放大比例，建议`bbox_body_scale=1.2`
- bbox_hand_scale, bbox_face_scale: 如果检测框由关键点推导得出，则应包含这两项，描述由smpl/smplx/2d_gt关键点推导出的身体检测框的放大比例，建议`bbox_hand_scale=1.0, bbox_face_scale=1.0`
- smpl_source / smplx_source: 描述smpl/smplx的来源，`'original', 'nerual_annot', 'eft', 'osx_annot', 'cliff_annot'`, 来描述smpl/smnplx是来源于数据集提供，或者其它标注来源
- cam_param_type: 描述相机参数的种类，`cam_param_type='prespective' or 'predicted_camera' or 'eft_camera'`
- principal_point, focal_length: (1, 2), numpy array，如果数据集中相机参数恒定，则应包含这两项，通常适用于生成数据集。
- image_shape: (1, 2), numpy array，如果数据集中图片大小恒定，则应包含此项。

#### `HumanData['meta']`中建议（可能）包含的内容:
- gender: (N, )， 字符串组成的列表, 每一个元素是smplx模型的性别（中性则不必标注）
- height（width）：(N, )， 字符串组成的列表, 每一个元素是图片的高（或宽），这里不推荐使用`image_shape=(width, height): (N, 2)`，因为有时需要按反顺序读取图片格式。（数据集图片分辨率一致则应标注在`HumanData['misc']`中）
- 其它有标识性的key，若数据集中该key不一致，且会影响keypoints or smpl/smplx，则建议标注，如focal_length与principal_point, focal_length = (N, 2), principal_point = (N, 2)

#### 关于HumanData的一些说明

- 所有数据标注均已从世界坐标转移到opencv相机空间，进行smpl/smplx的相机空间转换可以用

```from mmhuman3d.models.body_models.utils import transform_to_camera_frame, batch_transform_to_camera_frame```

#### 检查`HumanData`中的`key`.

在默认的`HumanData`中，只能包含之前描述过`key`。如果无法避免这个问题，还有一个方法。 通过指定`__key_strict__ == False`构建一个`HumanData`实例。

```python
human_data = HumanData.new(key_strict=False)
human_data['video_path'] = 'test.mp4'
```
返回的`HumanData`允许任何客制化的`key`，并在第一次接受新`key`时log一个警告。如果正在使用客制化的`key`，请忽略该警告，在程序结束前它不会再出现。

如果您已经构建了一个`HumanData`, 并且想改变`strict`模式, 可以使用`set_key_strict`:

```python
human_data = HumanData.fromfile('human_data.npz')
key_strict = human_data.get_key_strict()
human_data.set_key_strict(not key_strict)
```



#### 检查`HumanData`中的`value`.

只有指定`human_data[key] == value`时，才会检查之前提到过的`value`，相关规范在`HumanData.SUPPORTED_KEYS`中定义。

对于每个`value`, 必须在其`key`下指定数据类型:

```python
'smpl': {
    'type': dict,
},
```

对于类型为`numpy.ndarray`的`value`, 需要指定`shape`和`dim`参数:

```python
'keypoints3d': {
    'type': np.ndarray,
    'shape': (-1, -1, 4),
    # value.ndim==3 value.shape[2]==4
    # value.shape[0:2] is arbitrary.
    'dim': 0
    # dimension 0 marks time(frame index, or second)
},
```

对于不随帧变化的`value`，指定其`dim=-1`以跳过帧检查:

```python
'keypoints3d_mask': {
    'type': np.ndarray,
    'shape': (-1, ),
    'dim': -1
},
```

### 数据压缩

#### 压缩`mask`

由于`HUMAN_DATA`的关键点定义整合自不同的数据集，部分关键点缺失是很常见的。使用如下的方式，可以通过`mask`滤除缺失的关键点:

```python
# keypoints2d_agora 是一个形状为[frame_num, 127, 3]的numpy数组.
# Agora中定义了127个关键点.
keypoints2d_human_data, mask = convert_kps(keypoints2d_agora, 'agora', 'human_data')
# keypoints2d_human_data 是一个形状为[frame_num, 190, 3]的numpy数组.
# mask 是一个形状为[190, ]的numpy数组, 内含了127个1和63个0.
```

指定`keypoints2d_mask` 和 `keypoints2d`, 很明显`keypoints2d`中有很多冗余的0:

```python
human_data = HumanData()
human_data['keypoints2d_mask'] = mask
human_data['keypoints2d'] = keypoints2d_human_data
```

调用`compress_keypoints_by_mask()`去掉冗余的0。 该方法会检查所有包含`keypoints`的`key`是否具有相应的`mask`，如果同时存在，就进行压缩:

```python
human_data.compress_keypoints_by_mask()
```

调用`get_raw_value()`获得储存在`HumanData`实例中压缩过的`value`. 当通过`[]`获得`item`时, 将返回用0填充后的关键点:

```python
keypoints2d_human_data = human_data.get_raw_value('keypoints2d')
print(keypoints2d_human_data.shape)  # [frame_num, 127, 3]
keypoints2d_human_data = human_data['keypoints2d']
print(keypoints2d_human_data.shape)  # [frame_num, 190, 3]
```

在`keypoints_compressed`模式中, 可以对关键点进行编辑。有两种不同的方式，设置填充数据或直接指定压缩数据:

```python
padded_keypoints2d = np.zeros(shape=[100, 190, 3])
human_data['keypoints2d'] = padded_keypoints2d  # [frame_num, 190, 3]
compressed_keypoints2d = np.zeros(shape=[100, 127, 3])
human_data.set_raw_value('keypoints2d', compressed_keypoints2d)  # [frame_num, 127, 3]
```

当`HumanData`实例处于`keypoints_compressed`模式, 所有关键点的`masks`会被锁定。当您尝试进行修改时, 会log一个警告并且`value`不会被修改。如果需要修改`mask`，使用`decompress_keypoints()`进行解压缩:

```python
human_data.decompress_keypoints()
```

上述特性也适用于其他类似于`keypoints*`和`keypoints*_mask`的`key-value`对。

#### 压缩文件

调用`dump()` 将`HumanData`压缩成`.npz`文件。

`dump`后的`.npz`文件可以使用`load()`进行加载:

```python
# 保存
human_data.dump('./dumped_human_data.npz')
# 加载
another_human_data = HumanData()
another_human_data.load('./dumped_human_data.npz')
```

有时`HumanData`太长了，直接进行`dump`会报错。这种情况下，可以调用`dump_by_pickle`和`load_by_pickle`将`HumanData`压缩成`.pkl`文件。

#### 压缩`key`

如果`HumanData`实例不处于`key_strict`模式, 其可能会包含不受支持的项目。调用`pop_unsupported_items()`清除这些项目，以节省空间:

```python
human_data = HumanData.fromfile('human_data_not_strict.npz')
human_data.pop_unsupported_items()
human_data.set_key_strict(True)
```

### 数据选择

#### 通过形状进行选择

假定`keypoints2d` 是一个`array`，其形状为[200, 190, 3], 只需要最开始的10帧:

```python
first_ten_frames = human_data.get_value_in_shape('keypoints2d', shape=[10, -1, -1])
```

在一些情况下, 我们需要将`array`扩充成特定的形状:

```python
# 通过填0将其形状从[200, 190, 3]扩充成[200, 300, 3]
padded_keypoints2d = human_data.get_value_in_shape('keypoints2d', shape=[200, 300, -1])
# 填充的数值可以被修改
padded_keypoints2d = human_data.get_value_in_shape('keypoints2d', shape=[200, 300, -1], padding_constant=1)
```

#### 通过时间切片进行选择

假设在一个`HumanData`实例中有200帧, 只需要中间10至20帧:

```python
# 所有支持的`value`都会被切片
sub_human_data = human_data.get_slice(10, 21)
```

也可以进行下采样,下面的例子描述了一种选择其33%数据的方法:

```python
# 选择 [0, 3, 6, 9,..., 198]
sub_human_data = human_data.get_slice(0, 200, 3)
```

### 转换为`torch.Tensor`

之前提到，原生的`HumanData`中包含`numpy.ndarray`或python内置类型的`value`，但`numpy.ndarray`可以轻松转换为`torch.Tensor`:

```python
# ndarray会被转换成cpu上的Tensor.
# 其他类型的value不会发生变化.
# 返回一个形如HumanData的字典.
dict_of_tensor = human_data.to()
# 也可以转换为GPU上的Tensor
gpu0_device = torch.device('cuda:0')
dict_of_gpu_tensor = human_data.to(gpu0_device)
```
## MultiHumanData

MulitHumanData 被设计来支持多人重建。它继承于HumanData。在HumanData中，因为图片和数据是一一对应的，所以我们可以直接索引数据。然而，在MultiHumanData中数据和图像是多对一的关系。

MultiHumanData在HumanData的基础上添加一个新的key叫做`frame_range`,它定义如下：

```python
'frame_range': {
        'type': np.ndarray,
        'shape': (-1, 2),
        'dim': 0
    }
```

`frame_range`是和图像一一对应的。 `frame_range`中的每个元素是两个指针，它们指向一个数据区域。

假设我们有一个MultiHumanData的实例，我们想索引第i张图像对应的数据。 首先我们用主索引索引`frame_range`，它会返回两个指针，用这两个指针我们就可以索引第i张图像对应的所有数据。

```
image_0  ----> human_0      <--- frame_range[0][0]
         -       .
          -      .
           --> human_(n-1)  <--- frame_range[0][0] + (n-1)
            -> human_n      <--- frame_range[0][1]
    .
    .
    .


image_n  ----> human_0     <--- frame_range[n][0]
         -       .
          -      .
           --> human_(n-1)  <--- frame_range[n][0] + (n-1)
            -> human_n     <--- frame_range[n][1]

```
