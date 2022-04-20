## HumanData数据格式

### 总览

HumanData是Python内置字典类的子集，主要用于存放包含人体的单视角图像。它具有用于通用数据的良好的基础结构，也兼容具有新特征的客制化数据。
原生的HumanData包含numpy.ndarray或其他的Python内置的数据结构，但不包含torch.Tensor的数据。但可以使用`human_data.to()`将其转换为torch.Tensor(支持CPU和GPU)。

### Key/Value的定义

#### 如下是HumanData支持的Key和Value的描述.

- image_path: (N, ), 字符串组成的列表, 每一个元素是图像相对于根目录的路径。
- bbox_xywh: (N, 5), numpy array, 边界框的置信度, 边界框左下角点的坐标x和y, 边界框的宽w和高h, 得分放置在最后。
- config: (), 字符串, the flag name of config for individual dataset.
- keypoints2d: (N, 190, 3), numpy array, smplx模型的2d关节点与置信度, 每一个数据集的关节点映射到了HUMAN_DATA的关节点。
- keypoints3d: (N, 190, 4), numpy array, smplx模型的3d关节点与置信度, 每一个数据集的关节点映射到了HUMAN_DATA的关节点。
- smpl: (1, ), 字典, keys 分别为 ['body_pose': numpy array, (N, 23, 3), 'global_orient': numpy array, (N, 3), 'betas': numpy array, (N, 10), 'transl': numpy array, (N, 3)].
- smplx: (1, ), 字典, keys 分别为 ['body_pose': numpy array, (N, 21, 3),'global_orient': numpy array, (N, 3), 'betas': numpy array, (N, 10), 'transl': numpy array, (N, 3), 'left_hand_pose': numpy array, (N, 15, 3), 'right_hand_pose': numpy array, (N, 15, 3), 'expression': numpy array (N, 10), 'leye_pose': numpy array (N, 3), 'reye_pose': (N, 3), 'jaw_pose': numpy array (N, 3)].
- meta: (1, ), 字典, keys 为数据集中类似性别的元数据。
- keypoints2d_mask: (190, ), numpy array, 表示keypoints2d中关键点是否有效的掩膜。 0表示该位置的关键点在原始数据集中无法找到。
- keypoints3d_mask: (190, ), numpy array, 表示keypoints3d中关键点是否有效的掩膜。 0表示该位置的关键点在原始数据集中无法找到。
- misc: (1, ), dict, keys和values由用户定义。misc占用的空间(sys.getsizeof(misc))。

#### 检查HumanData中的key.

在默认的HumanData中，只能包含之前描述过key。如果你不能解决这个问题，还有一个方法。 通过指定`__key_strict__ == False`构建一个HumanData实例。

```python
human_data = HumanData.new(key_strict=False)
human_data['video_path'] = 'test.mp4'
```
返回的HumanData允许任何客制化的key，并在第一次接受新key时log一个警告。如果正在使用客制化的key，请忽略该警告，在程序结束前它不会再出现。

如果您已经构建了一个HumanData, 并且想改变 strict 模式, 可以像如下使用`set_key_strict`:

```python
human_data = HumanData.fromfile('human_data.npz')
key_strict = human_data.get_key_strict()
human_data.set_key_strict(not key_strict)
```



#### 检查HumanData中的value.

只有指定`human_data[key] == value`时，才会检查上面提到过的value，相关约束在`HumanData.SUPPORTED_KEYS`中定义。

对于每个value, 必须在其key下指定数据类型:

```python
'smpl': {
    'type': dict,
},
```

对于类型为numpy.ndarray的value, shape和temporal_dim参数需要指定:

```python
'keypoints3d': {
    'type': np.ndarray,
    'shape': (-1, -1, 4),
    # value.ndim==3, and value.shape[2]==4
    # value.shape[0:2] is arbitrary.
    'temporal_dim': 0
    # dimension 0 marks time(frame index, or second)
},
```

对于不随时间变化的value，指定其temporal_dim=1以忽略时序检查:

```python
'keypoints3d_mask': {
    'type': np.ndarray,
    'shape': (-1, ),
    'temporal_dim': -1
},
```

### 数据压缩

#### 压缩mask

由于HUMAN_DATA的关键点定义是来自不同数据集的集合，部分关键点缺失是很常见的。使用如下的方式，通过mask过滤缺失的关键点:

```python
# keypoints2d_agora is a numpy array in shape [frame_num, 127, 3].
# There are 127 keypoints defined by agora.
keypoints2d_human_data, mask = convert_kps(keypoints2d_agora, 'agora', 'human_data')
# keypoints2d_human_data is a numpy array in shape [frame_num, 190, 3], only 127/190 are valid
# mask is a numpy array in shape [190, ], with 127 ones and 63 zeros inside
```

指定`keypoints2d_mask` 和 `keypoints2d`。 很明显keypoints2d中有很多冗余的0:

```python
human_data = HumanData()
human_data['keypoints2d_mask'] = mask
human_data['keypoints2d'] = keypoints2d_human_data
```

调用`compress_keypoints_by_mask()`去掉冗余的0。 该方法检查所有包含`keypoints`的key是否具有相应的mask，如果同时存在，就进行压缩:

```python
human_data.compress_keypoints_by_mask()
```

调用`get_raw_value()`获得储存在HumanData实例中压缩过的value. 当使用`[]`获得item时, 将返回用0填充后的关键点:

```python
keypoints2d_human_data = human_data.get_raw_value('keypoints2d')
print(keypoints2d_human_data.shape)  # [frame_num, 127, 3]
keypoints2d_human_data = human_data['keypoints2d']
print(keypoints2d_human_data.shape)  # [frame_num, 190, 3]
```

在`keypoints_compressed`模式中, 可以编辑关键的点。有两种不同的方式，设置填充数据或直接设置压缩数据:

```python
padded_keypoints2d = np.zeros(shape=[100, 190, 3])
human_data['keypoints2d'] = padded_keypoints2d  # [frame_num, 190, 3]
compressed_keypoints2d = np.zeros(shape=[100, 127, 3])
human_data.set_raw_value('keypoints2d', compressed_keypoints2d)  # [frame_num, 127, 3]
```

当HumanData实例处于`keypoints_compressed`模式, 关键点的所有masks会被锁定。当你尝试去修改时, 会log一个警告并且value不会被修改。如果需要修改mask，使用`decompress_keypoints()`进行解压缩:

```python
human_data.decompress_keypoints()
```

上述特性也适用于其他类似于`keypoints*`和`keypoints*_mask`的key-value对。

#### 压缩文件

调用`dump()` 将HumanData压缩成`.npz`文件。

dump后的`.npz`文件可以使用`load()`进行加载:

```python
# save
human_data.dump('./dumped_human_data.npz')
# load
another_human_data = HumanData()
another_human_data.load('./dumped_human_data.npz')
```

有时HumanData太长了，直接进行dump会报错`numpy.savez_compressed()`。这种情况下，可以调用`dump_by_pickle`和`load_by_pickle`将HumanData压缩成`.pkl`文件。

#### 压缩key

如果HumanData实例不处于key_strict模式, 其可能会包含不支持的item。调用`pop_unsupported_items()`清除这些item，这样子会节省空间:

```python
human_data = HumanData.fromfile('human_data_not_strict.npz')
human_data.pop_unsupported_items()
# set instance.__key_strict__ from True to False will also do
human_data.set_key_strict(True)
```

### 数据选择

#### 通过shape进行选择

假定`keypoints2d` 是一个array，其shape为[200, 190, 3], 只需要最开始的10帧:

```python
first_ten_frames = human_data.get_value_in_shape('keypoints2d', shape=[10, -1, -1])
```

在一些情况下, 我们需要将array扩充成特定的维度:

```python
# pad keypoints2d from [200, 190, 3] to [200, 300, 3] with zeros
padded_keypoints2d = human_data.get_value_in_shape('keypoints2d', shape=[200, 300, -1])
# padding value can be modified
padded_keypoints2d = human_data.get_value_in_shape('keypoints2d', shape=[200, 300, -1], padding_constant=1)
```

#### Select temporal slice

假设在一个HumanData实例中有200帧, 只需要中间10至20帧:

```python
# all supported values will be sliced
sub_human_data = human_data.get_temporal_slice(10, 21)
```

也可以进行下采样,下面的例子描述了一种选择其33%数据的方法:

```python
# select [0, 3, 6, 9,..., 198]
sub_human_data = human_data.get_temporal_slice(0, 200, 3)
```

### To torch.Tensor

之前提到，原生的HumanData包含numpy.ndarray或python内置类型的value，但numpy.ndarray可以轻松转换为torch.Tensor:

```python
# All values as ndarray will be converted to a cpu Tensor.
# Values in other types will not change.
# It returns a dict like HumanData.
dict_of_tensor = human_data.to()
# GPU is also supported
gpu0_device = torch.device('cuda:0')
dict_of_gpu_tensor = human_data.to(gpu0_device)
```
