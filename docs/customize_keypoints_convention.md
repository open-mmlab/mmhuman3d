# Customize keypoints convention

This doc is about how to add a new set of keypoints (NEW_KEYPOINTS) into use in this project.

**1. Search for keypoint names in smplx.py.** Please follow
[`mmhuman3d/core/conventions/keypoints_mapping/smplx.py`](mmhuman3d/core/conventions/keypoints_mapping/smplx.py)
to create a file named NEW_KEYPOINTS.py. In this project, you can find corresponding names in smplx for NEW_KEYPOINTS. We use names to map keypoints. Then list them as smplx.py.

**2. Search for keypoint names in other sets of keypoints.** If a keypoint is not in smplx.py, please also go through other sets of keypoints under
[`mmhuman3d/core/conventions/keypoints_mapping`](mmhuman3d/core/conventions/keypoints_mapping) and change to the same name if you
found it in the above folder. This is because smplx does contain all the definition of keypoints. For example, smplx does not have 'head_top' while [mpi_inf_3dhp](mmhuman3d/core/conventions/keypoints_mapping/mpi_inf_3dhp.py) does.
The keypoints in different datasets while sharing the same name should have the
exactly same semantic definitions in human body.

**3. Add a new keypoint name.** If you cannot find an existing keypoint in [existing
conventions](mmhuman3d/core/conventions/keypoints_mapping) for a keypoint in
NEW_KEYPOINTS, please list it as well but add a prefix to the original name to differentiate it from those with smplx correspondences.

**4. Modify keypoints mapping code to add new set of keypoints into use.** Add import for NEW_KEYPOINTS.py in
[`mmhuman3d/core/conventions/keypoints_mapping/__init__.py`](mmhuman3d/core/conventions/keypoints_mapping/__init__.py#L6-15), and also add one line
in dict [KEYPOINTS_FACTORY](mmhuman3d/core/conventions/keypoints_mapping/__init__.py#L17-27).

**5. Use new keypoints set for keypoints mapping.** Now you can use convert_kps(both 2d and 3d keypoints are supported) in
[`mmhuman3d/core/conventions/keypoints_mapping/__init__.py`](mmhuman3d/core/conventions/keypoints_mapping/__init__.py) to convert
NEW_KEYPOINTS to other sets of keypoints like smplx, coco, etc.
For example, you can convert your smplx keypoints to coco keypoints by running the following code, mask only contains 0 or 1. 0 meaning that the corresponding point should be excluded or not, respectively.
```
  new_kps, mask = convert_kps(smplx_keypoints, src='smplx', dst='NEW_KEYPOINT_NAME')
```
