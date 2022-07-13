"""SPIN in smplx convention.

SPIN_SMPLX_KEYPOINTS can be found in https://github.com/vchoutas/expose
"""

# TODO: SMPL_24->HumanData->SMPLX causes hip, spine to be lost.
# SMPL_24: left/right_hip_extra
# SMPLX: left/right_hip

SPIN_SMPLX_KEYPOINTS = [
    'right_ankle',
    'right_knee',
    'right_hip',
    'left_hip',
    'left_knee',
    'left_ankle',
    'right_wrist',
    'right_elbow',
    'right_shoulder',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'neck',
    'head_top',
    'pelvis',
    'thorax',
    'spine',
    'h36m_jaw',
    'h36m_head',
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
]
