## HumanData

### Overview

HumanData is a subclass of python built-in class dict, containing single-view, image-based data for a human being. It has a well-defined base structure for universal data, but it is also compatible with customized data for new features. A native HumanData contains values in numpy.ndarray or python built-in types, it holds no data in torch.Tensor, but you can convert arrays to torch.Tensor(even to GPU Tensor) by `human_data.to()`  easily.

### Key/Value definition

#### Paths:

Image path is included, and optionally path of segmentation map and depth image can be included if provided by dataset.
- image_path: (N, ), list of str, each element is a relative path from the root folder (exclusive) to the image.
- segmentation (optional): (N, ), list of str, each element is a relative path from the root folder (exclusive) to the segmentation map.
- depth_path (optional): (N, ), list of str, each element is a relative path from the root folder (exclusive) to the depth image.

#### Keypoints：

Following keys should be included in `HumanData` if applicable. For each dictionary key of keypoints，a corresponding dictionart key of mask should be included，stating which keypoint is valid. For example `keypoints3d_original` should correspond to `keypoints3d_original_mask`.

In `HumanData`, keypoints are stored as `HUMAN_DATA` format, which includes 190 joints. We provide keypoints format (for both 2d and 3d keypoints) convention for many datasets, please see [keypoints_convention](../docs/keypoints_convention.md).

- keypoints3d_smpl / keypoints3d_smplx: (N, 190, 4), numpy array, `smplx / smplx` 3d joints with confidence, joints from each datasets are mapped to HUMAN_DATA joints.
- keypoints3d_original: (N, 190, 4), numpy array, 3d joints with confidence which provided by the dataset originally, joints from each datasets are mapped to HUMAN_DATA joints.
- keypoints2d_smpl / keypoints2d_smplx: (N, 190, 3), numpy array, `smpl / smplx` 2d joints with confidence, joints from each datasets are mapped to HUMAN_DATA joints.
- keypoints2d_original: (N, 190, 3), numpy array, 2d joints with confidence which provided by the dataset originally, joints from each datasets are mapped to HUMAN_DATA joints.
- (mask sample) keypoints2d_smpl_mask: (190, ), numpy array, mask for which keypoint is valid in `keypoints2d_smpl`. 0 means that the joint in this position cannot be found in original dataset.

#### Bounding Box：

Bounding box of body (smpl), face and hand (smplx), which data type is `[x_min, y_min, width, height, confidence]`，and should not exceed the image boundary.
- bbox_xywh: (N, 5), numpy array, bounding box with confidence, coordinates of bottom-left point x, y, width w and height h of bbox, score at last.
- face_bbox_xywh, lhand_bbox_xywh, rhand_bbox_xywh (optional): (N, 5), numpy array, should be included if `smplx` data is provided, and is derived from smplx2d keypoints. Have the same srtucture as above.

#### Human Pose and Shape Parameters：

Normally saved as smpl/smplx.
- smpl: (1, ), dict, keys are `['body_pose': numpy array, (N, 23, 3), 'global_orient': numpy array, (N, 3), 'betas': numpy array, (N, 10), 'transl': numpy array, (N, 3)]`.
- smplx: (1, ), dict, keys are `['body_pose': numpy array, (N, 21, 3),'global_orient': numpy array, (N, 3), 'betas': numpy array, (N, 10), 'transl': numpy array, (N, 3), 'left_hand_pose': numpy array, (N, 15, 3), 'right_hand_pose': numpy array, (N, 15, 3), 'expression': numpy array (N, 10), 'leye_pose': numpy array (N, 3), 'reye_pose': (N, 3), 'jaw_pose': numpy array (N, 3)]`.


#### Other keys

- config: (), str, the flag name of config for individual dataset.
- meta: (1, ), dict, its keys are meta data from dataset like 'gender'.
- misc: (1, ), dict, keys and values are designed to describe the different settings for each dataset. Can also be defined by user. The space misc takes (sys.getsizeof(misc)) shall be no more than 6MB.

#### Suggestion for WHAT to include in `HumanData['misc']`:

Miscellaneous contains the info of different settings for each dataset, including camaera type, source of keypoints annotation, bounding box etc. Aims to faclitate different usage of data.
`HumanData['misc']` is a dictionary and its keys are described as following:
- kps3d_root_aligned： Bool, stating that if keypoints3d is root-aligned，root_alignment is not preferred for HumanData. If this key does not exist, root_aligenment is by default to be `False`.
- flat_hand_mean：Bool, applicable for smplx data，for most datasets `flat_hand_mean=False`.
- bbox_source：source of bounding box，`bbox_soruce='keypoints2d_smpl' or 'keypoints2d_smplx' or 'keypoints2d_original'`，describing which type of keypoints are used to derive the bounding box，OR `bbox_source='provide_by_dataset'` shows that bounding box if provided by dataset. (For example, from some detection module rather than keypoints)
- bbox_body_scale: applicable if bounding box is derived by keypoints，stating the zoom-in scale of bounding scale from smpl/smplx/2d_gt keypoints，we suggest `bbox_body_scale=1.2`.
- bbox_hand_scale, bbox_face_scale: applicable if bounding box is derived by smplx keypoints，stating the zoom-in scale of bounding scale from smplx/2d_gt keypoints，we suggest `bbox_hand_scale=1.0, bbox_face_scale=1.0`
- smpl_source / smplx_source: describing the source of smpl/smplx annotations，`'original', 'nerual_annot', 'eft', 'osx_annot', 'cliff_annot'`.
- cam_param_type: describing the type of camera parameters，`cam_param_type='prespective' or 'predicted_camera' or 'eft_camera'`
- principal_point, focal_length: (1, 2), numpy array，applicable if camera parameters are same across the whole dataset, which is the case for some synthetic datasets.
- image_shape: (1, 2), numpy array，applicable if image shape are same across the whole dataset.

#### Suggestion for WHAT to include in `HumanData['meta']`:

- gender: (N, ), list of str, each element represents the gender for an smpl/smplx instance. (key not required if dataset use gender-neutral model)
- height (width)：(N, )， list of str, each element represents the height (width) of an image, `image_shape=(width, height): (N, 2)` is not suggested as width and height might need to be referenced in different orders. (keys should be in `HumanData['misc']` if image shape are same across the dataset)
- other keys，applicable if the key value is different，and have some impact on keypoints or smpl/smplx (2d and 3d)，For example, `focal_length` and `principal_point`, focal_length = (N, 2), principal_point = (N, 2)

#### Some other info of HumanData

- All annotations are transformed from world space to opencv camera space, for space transformation we use:

    ```from mmhuman3d.models.body_models.utils import transform_to_camera_frame, batch_transform_to_camera_frame```

#### Key check in HumanData.

Only keys above are allowed as top level key in a default HumanData. If you cannot work with that, there's also a way out. Construct a HumanData instance with `__key_strict__ == False`:

```python
human_data = HumanData.new(key_strict=False)
human_data['video_path'] = 'test.mp4'
```
The returned human_data will allow any customized keys, logging a warning at the first time HumanData sees a new key. Just ignore the warning if you do know that you are using a customized key, it will not appear again before the program ends.

If you have already constructed a HumanData, and you want to change the strict mode, use `set_key_strict`:

```python
human_data = HumanData.fromfile('human_data.npz')
key_strict = human_data.get_key_strict()
human_data.set_key_strict(not key_strict)
```



#### Value check in HumanData.

Only values above will be check when `human_data[key] == value` is called, and the constraints are defined in `HumanData.SUPPORTED_KEYS`.

For each value, an exclusive type must be specified under its key:

```python
'smpl': {
    'type': dict,
},
```

For value as numpy.ndarray, shape and dim shall be defined:

```python
'keypoints3d': {
    'type': np.ndarray,
    'shape': (-1, -1, 4),
    # value.ndim==3, and value.shape[2]==4
    # value.shape[0:2] is arbitrary.
    'dim': 0
    # dimension 0 marks time(frame index, or second)
},
```

For value which is constant along frame axis, set dim to -1 to ignore frame check:

```python
'keypoints3d_mask': {
    'type': np.ndarray,
    'shape': (-1, ),
    'dim': -1
},
```

### Data compression

#### Compression with mask

As the keypoint convention named HUMAN_DATA is a union of keypoint definitions from various datasets, it is common that some keypoints are missing. In this situation, the missing ones are filtered by mask:

```python
# keypoints2d_agora is a numpy array in shape [frame_num, 127, 3].
# There are 127 keypoints defined by agora.
keypoints2d_human_data, mask = convert_kps(keypoints2d_agora, 'agora', 'human_data')
# keypoints2d_human_data is a numpy array in shape [frame_num, 190, 3], only 127/190 are valid
# mask is a numpy array in shape [190, ], with 127 ones and 63 zeros inside
```

Set `keypoints2d_mask` and `keypoints2d`. It is obvious that there are redundant zeros in keypoints2d:

```python
human_data = HumanData()
human_data['keypoints2d_mask'] = mask
human_data['keypoints2d'] = keypoints2d_human_data
```

Call `compress_keypoints_by_mask()` to get rid of the zeros. This method checks if any key containing `keypoints` has a corresponding mask, and performs keypoints compression if both keypoints and masks are present. :

```python
human_data.compress_keypoints_by_mask()
```

Call  `get_raw_value()`  to get the compressed raw value stored in HumanData instance. When getting item with  `[]`, the keypoints padded with zeros will be returned:

```python
keypoints2d_human_data = human_data.get_raw_value('keypoints2d')
print(keypoints2d_human_data.shape)  # [frame_num, 127, 3]
keypoints2d_human_data = human_data['keypoints2d']
print(keypoints2d_human_data.shape)  # [frame_num, 190, 3]
```

In  `keypoints_compressed` mode, keypoints are allowed to be edited. There are two different ways, set with padded data or set the compressed data directly:

```python
padded_keypoints2d = np.zeros(shape=[100, 190, 3])
human_data['keypoints2d'] = padded_keypoints2d  # [frame_num, 190, 3]
compressed_keypoints2d = np.zeros(shape=[100, 127, 3])
human_data.set_raw_value('keypoints2d', compressed_keypoints2d)  # [frame_num, 127, 3]
```

When a HumanData instance is in  `keypoints_compressed` mode, all masks of keypoints are locked. If you are trying to edit it, a warning will be logged and the value won't change. To modify a mask, de-compress it with `decompress_keypoints()`:

```python
human_data.decompress_keypoints()
```

Features above also work with any key pairs like `keypoints*` and `keypoints*_mask`.

#### Compression for file

Call `dump()` to save HumanData into a compressed  `.npz` file.

The dumped file can be load by `load()` :

```python
# save
human_data.dump('./dumped_human_data.npz')
# load
another_human_data = HumanData()
another_human_data.load('./dumped_human_data.npz')
```

Sometimes a HumanData instanse is too large to dump, an error will be raised by `numpy.savez_compressed()`. In this case, call `dump_by_pickle`  and `load_by_pickle` for file operation.

#### Compression by key

If a HumanData instance is in not in key_strict mode, it may contains unsupported items which are not necessary. Call `pop_unsupported_items()` to remove those items will save space for you:

```python
human_data = HumanData.fromfile('human_data_not_strict.npz')
human_data.pop_unsupported_items()
# set instance.__key_strict__ from True to False will also do
human_data.set_key_strict(True)
```

### Data selection

#### Select by shape

Assume that `keypoints2d` is an array in shape [200, 190, 3], only the first 10 frames are needed:

```python
first_ten_frames = human_data.get_value_in_shape('keypoints2d', shape=[10, -1, -1])
```

In some situation, we need to pad all arrays to a certain size:

```python
# pad keypoints2d from [200, 190, 3] to [200, 300, 3] with zeros
padded_keypoints2d = human_data.get_value_in_shape('keypoints2d', shape=[200, 300, -1])
# padding value can be modified
padded_keypoints2d = human_data.get_value_in_shape('keypoints2d', shape=[200, 300, -1], padding_constant=1)
```

#### Select temporal slice

Assume that there are 200 frames in a HumanData instance, only data between 10 and 20 are needed:

```python
# all supported values will be sliced
sub_human_data = human_data.get_slice(10, 21)
```

Downsample is also supported, for example, select 33%:

```python
# select [0, 3, 6, 9,..., 198]
sub_human_data = human_data.get_slice(0, 200, 3)
```

### To torch.Tensor

As introduced, a native HumanData contains values in numpy.ndarray or python built-in types, but the numpy.ndarray can be easily convert to torch.Tensor:

```python
# All values as ndarray will be converted to a cpu Tensor.
# Values in other types will not change.
# It returns a dict like HumanData.
dict_of_tensor = human_data.to()
# GPU is also supported
gpu0_device = torch.device('cuda:0')
dict_of_gpu_tensor = human_data.to(gpu0_device)
```

## MultiHumanData

MulitHumanData is designed to support multi-human body mesh recovery, who inherits from HumanData. In HumanData, the data can be accessed directly through the index, because the data and the image are in one-to-one correspondence. However, data and image have a many-to-one correspondence in MultiHumanData.

Based on HumanData, MultiHumanData adds a new key named `'frame_range'` as follows:

```python
'frame_range': {
        'type': np.ndarray,
        'shape': (-1, 2),
        'dim': 0
    }
```
`frame_range` and image are in one-to-one correspondence.
Each element in `frame_range` has two pointers that point to a data-block.

Suppose we have an instance of MultiHumanData and we want to access the data corresponding to the i-th image. First, we index the `frame_range` using primary index i, which will return two points. We then use these two pointers to access all data corresponding to the i-th image.

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
