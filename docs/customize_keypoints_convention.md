## Customize keypoints convention

### Overview

If your dataset use an unsupported convention, a new convention can be added following this documentation.

These are the conventions that our project currently support:
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


**1. Create a new convention**

Please follow
[`mmhuman3d/core/conventions/keypoints_mapping/human_data.py`](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/core/conventions/keypoints_mapping/human_data.py) to create a file named NEW_CONVENTION.py. In this file,
`NEW_KEYPOINTS` is a list containing keypoints naming and order specific to the new convention.

For instance, if we want to create a new convention for AGORA dataset, `agora.py` would contain:
```
AGORA_KEYPOINTS = [
  'pelvis',
  'left_hip',
  'right_hip'
  ...
]
```

**2. Search for keypoint names in `human_data`.**

In this project, keypoints that share the same naming across datasets should have the exact same semantic definition in the human body. `human_data` convention has already consolidated the different keypoints naming and correspondences across our supported datasets.

For each keypoint in `NEW_KEYPOINTS`, we have to check (1) if the keypoint name exists in [`mmhuman3d/core/conventions/keypoints_mapping/human_data.py`](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/core/conventions/keypoints_mapping/human_data.py) and (2) if the keypoint has a correspondence i.e. maps to the same
location as the ones defined in `human_data`.

If both conditions are met, retain the keypoint name in NEW_CONVENTION.py.


**3. Search for keypoints correspondence in `human_data`.**

If a keypoint in `NEW_KEYPOINTS` shares the same correspondence as a keypoint that is named differently in the `human_data` convention i.e. `head` in NEW_CONVENTION.py maps to `head_extra`
in `human_data`, rename the keypoint to follow the new one in our convention i.e. `head`-> `head_extra`.

**4. Add a new keypoint to `human_data`**

If the keypoint has no correspondence nor share an existing name to the ones defined in `human_data`, please list it as well but add a prefix to the original name to differentiate it from those with existing correspondences i.e. `spine_3dhp`

We may expand `human_data` to the new keypoint if necessary. However, this can only be done after checking that the new keypoint do not have a correspondence and there is no conflicting names.

**5. Initialise the new set of keypoint convention**

Add import for NEW_CONVENTION.py in
[`mmhuman3d/core/conventions/keypoints_mapping/__init__.py`](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/core/conventions/keypoints_mapping/__init__.py#L8-25), and add the identifier to dict [KEYPOINTS_FACTORY](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/core/conventions/keypoints_mapping/__init__.py#L27-52).

For instance, if our new convention is `agora`:
```
# add import
from mmhuman3d.core.conventions.keypoints_mapping import (
    agora,
    ...
)

# add to factory
KEYPOINTS_FACTORY = {
    'agora': agora.AGORA_KEYPOINTS,
    ...
}
```

**6. Using keypoints convention for keypoints mapping**

To convert keypoints from any existing convention to your newly defined convention (or vice versa), you can use the `convert_kps` function [`mmhuman3d/core/conventions/keypoints_mapping/__init__.py`](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/core/conventions/keypoints_mapping/__init__.py), which produce a mask containing 0 or 1 indicating if the corresponding point should be filtered or retained.

To convert from coco to new convention:
```
  new_kps, mask = convert_kps(smplx_keypoints, src='coco', dst='NEW_CONVENTION')
```

To convert from new convention to human_data:
```
  new_kps, mask = convert_kps(smplx_keypoints, src='NEW_CONVENTION', dst='human_data')
```
