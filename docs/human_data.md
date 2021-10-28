## HumanData

### Overview

HumanData is a subclass of python built-in class dict, containing single-view, image-based data for a human being. It has a well-defined base structure for universal data, but it is also compatible with customized data for new features. A native HumanData contains values in numpy.ndarray or python built-in types, it holds no data in torch.Tensor, but you can convert arrays to torch.Tensor(even to GPU Tensor) by `human_data.to()`  easily.

### Key/Value definition

#### The keys and values supported by HumanData are described as below.

- image_path: (N, ), list of str, each element is a relative path from the root folder (exclusive) to the image.
- bbox_xywh: (N, 5), numpy array, bounding box with confidence, coordinates of bottom-left point x, y, width w and height h of bbox, score at last.
- config: (), str, the flag name of config for individual dataset.
- keypoints2d: (N, 190, 3), numpy array, 2d joints of smplx model with confidence, joints from each datasets are mapped to HUMAN_DATA joints.
- keypoints3d: (N, 190, 4), numpy array, 3d joints of smplx model with confidence. Same as above.
- smpl: (1, ), dict, keys are ['body_pose': numpy array, (N, 23, 3), 'global_orient': numpy array, (N, 3), 'betas': numpy array, (N, 10), 'transl': numpy array, (N, 3)].
- smplx: (1, ), dict, keys are ['body_pose': numpy array, (N, 21, 3),'global_orient': numpy array, (N, 3), 'betas': numpy array, (N, 10), 'transl': numpy array, (N, 3), 'left_hand_pose': numpy array, (N, 15, 3), 'right_hand_pose': numpy array, (N, 15, 3), 'expression': numpy array (N, 10), 'leye_pose': numpy array (N, 3), 'reye_pose': (N, 3), 'jaw_pose': numpy array (N, 3)].
- meta: (1, ), dict, its keys are meta data from dataset like 'gender'.
- keypoints2d_mask: (190, ), numpy array, mask for which keypoint is valid in keypoints2d. 0 means that the joint in this position cannot be found in original dataset.
- keypoints3d_mask: (190, ), numpy array, mask for which keypoint is valid in keypoints3d. 0 means that the joint in this position cannot be found in original dataset.
- misc: (1, ), dict, keys and values are defined by user. The space misc takes(sys.getsizeof(misc)) shall be no more than 6MB.

#### Key check in HumanData.

Only keys above are allowed as top level key in a default HumanData. If you cannot work with that, there's also a way out. Construct a HumanData instance with `__key_strict__ == False`:

```python
human_data = HumanData.new(key_strict=False)
human_data['video_path'] = 'test.mp4'
```
The returned human_data will allow any customized keys, logging a warning at the first time HumanData sees a new key. Just ignore the warning if you do know that you are using a customized key, it will not appear again before the program ends.

#### Value check in HumanData.

Only values above will be check when `human_data[key] == value` is called, and the constraints are defined in `HumanData.SUPPORTED_KEYS`.

For each value, an exclusive type must be specified under its key:

```python
'smpl': {
    'type': dict,
},
```

For value as numpy.ndarray, shape and temporal_dim shall be defined:

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

For value which is constant along time axis, set temporal_dim to -1 to ignore temporal check:

```python
'keypoints3d_mask': {
    'type': np.ndarray,
    'shape': (-1, ),
    'temporal_dim': -1
},
```
