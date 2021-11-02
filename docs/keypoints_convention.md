# About convention

This doc is to help users understand and use our convention tools.

## Keypoints mapping
- **discription:**

    Our convention means the keypoints definition of human body. The same keypoint names mean that the keypoints are semantically the same location on human body. We can convert keypoints following the mapping correspondence defined by the keypoints names.

    In our project, we use the `smplx` convention as basic convention and named the corresponding keypoint names following [smplx](mmhuman3d/core/conventions/keypoints_mapping/smplx.py). Those keypoints that do not have correspondences to smplx should be inde=icated with an extra prefix (such as COCO_, or GTA_).

- **simple example:**

    You can convert keypoint between different conventions easily with the help of function `convert_kps`. E.g., you have a `mmpose` keypoints array of shape (100, 133, 2) and you want to convert it to a `coco` convention array, you can do:
    ```python
    from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
    keypoints_mmpose = np.zeros((100, 133, 2))
    keypoints_coco, mask = convert_kps(keypoints_mmpose, src='mmpose', dst='coco')
    ```
    The output `mask` should be all ones if the `dst` convention is the subset of the `src` convention.
    You can use the mask as the confidence of the keypoints since those keypoints with no correspondence are set to a default value with 0 confidence.
- **original mask:**

    If you have occlusion information of your keypoints, you can use an original mask to mark it, then the information will be updated into the returned mask.
    E.g., you want to convert a `smpl` keypoints to `coco` keypoints, and you know its `left_hip` is occluded. You want to carry forward this information during the converting. So you can set an original_mask and convert it to `coco` by doing:

    ```python
    keypoints = np.zeros((1, len(KEYPOINTS_FACTORY['smpl']), 3))
    original_mask = np.ones((len(KEYPOINTS_FACTORY['smpl'])))
    original_mask[KEYPOINTS_FACTORY['smpl'].index('left_hip')] = 0
    _, mask_coco = convert_kps(
        keypoints=keypoints, mask=original_mask, src='smpl', dst='coco')
    _, mask_coco_full = convert_kps(
        keypoints=keypoints, src='smpl', dst='coco')
    assert mask_coco[KEYPOINTS_FACTORY['coco'].index('left_hip')] == 0
    mask_coco[KEYPOINTS_FACTORY['coco'].index('left_hip')] = 1
    assert (mask_coco == mask_coco_full).all()
    ```

- **convert with your defined convention:**

    Please refer to [customize_keypoints_convention](docs/customize_keypoints_convention.md).
