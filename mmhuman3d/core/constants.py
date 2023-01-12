# yapf: disable
from mmhuman3d.core.conventions.keypoints_mapping.mano import (
    MANO_RIGHT_REORDER_KEYPOINTS,
)
from mmhuman3d.core.conventions.keypoints_mapping.openpose import (
    OPENPOSE_25_KEYPOINTS,
)
from mmhuman3d.core.conventions.keypoints_mapping.smplx import SMPLX_KEYPOINTS
from mmhuman3d.core.conventions.keypoints_mapping.spin_smplx import (
    SPIN_SMPLX_KEYPOINTS,
)

# yapf: enable

# We create a superset of joints containing the OpenPose joints together with
# the ones that each dataset provides. We keep a superset of 24 joints.
# If a dataset doesn't provide annotations for a specific joint,
# we simply ignore it.
# The joints used here are the following:

JOINT_NAMES = OPENPOSE_25_KEYPOINTS + SPIN_SMPLX_KEYPOINTS

# Dict containing the joints in numerical order
JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}

# Map joints to SMPL joints
JOINT_MAP = {
    'nose_openpose': 24,
    'neck_openpose': 12,
    'right_shoulder_openpose': 17,
    'right_elbow_openpose': 19,
    'right_wrist_openpose': 21,
    'left_shoulder_openpose': 16,
    'left_elbow_openpose': 18,
    'left_wrist_openpose': 20,
    'pelvis_openpose': 0,
    'right_hip_openpose': 2,
    'right_knee_openpose': 5,
    'right_ankle_openpose': 8,
    'left_hip_openpose': 1,
    'left_knee_openpose': 4,
    'left_ankle_openpose': 7,
    'right_eye_openpose': 25,
    'left_eye_openpose': 26,
    'right_ear_openpose': 27,
    'left_ear_openpose': 28,
    'left_bigtoe_openpose': 29,
    'left_smalltoe_openpose': 30,
    'left_heel_openpose': 31,
    'right_bigtoe_openpose': 32,
    'right_smalltoe_openpose': 33,
    'right_heel_openpose': 34,
    'right_ankle': 8,
    'right_knee': 5,
    'right_hip': 45,
    'left_hip': 46,
    'left_knee': 4,
    'left_ankle': 7,
    'right_wrist': 21,
    'right_elbow': 19,
    'right_shoulder': 17,
    'left_shoulder': 16,
    'left_elbow': 18,
    'left_wrist': 20,
    'neck': 47,
    'head_top': 48,
    'pelvis': 49,
    'thorax': 50,
    'spine': 51,
    'h36m_jaw': 52,
    'h36m_head': 53,
    'nose': 24,
    'left_eye': 26,
    'right_eye': 25,
    'left_ear': 28,
    'right_ear': 27
}

# Permutation of SMPL pose parameters when flipping the shape
SMPL_JOINTS_FLIP_PERM = [
    0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21,
    20, 23, 22
]
SMPL_POSE_FLIP_PERM = []
for i in SMPL_JOINTS_FLIP_PERM:
    SMPL_POSE_FLIP_PERM.append(3 * i)
    SMPL_POSE_FLIP_PERM.append(3 * i + 1)
    SMPL_POSE_FLIP_PERM.append(3 * i + 2)
# Permutation indices for the 24 ground truth joints
J24_FLIP_PERM = [
    5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18, 19, 21,
    20, 23, 22
]
# Permutation indices for the full set of 49 joints
J49_FLIP_PERM = [
    0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11,
    16, 15, 18, 17, 22, 23, 24, 19, 20, 21]\
              + [25+i for i in J24_FLIP_PERM]
SMPL_J49_FLIP_PERM = [
    0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11,
    16, 15, 18, 17, 22, 23, 24, 19, 20, 21]\
              + [25+i for i in SMPL_JOINTS_FLIP_PERM]

SMPLX2SMPL_J45 = [i for i in range(22)] + [30, 45
                                           ] + [i for i in range(55, 55 + 21)]

SMPLX_JOINT_IDS = {SMPLX_KEYPOINTS[i]: i for i in range(len(SMPLX_KEYPOINTS))}

FOOT_NAMES = ['bigtoe', 'smalltoe', 'heel']

# LRHAND_FLIP_PERM = [i for i in range(16, 32)] + [i for i in range(16)]
LRHAND_FLIP_PERM = [
    i for i in range(
        len(MANO_RIGHT_REORDER_KEYPOINTS),
        len(MANO_RIGHT_REORDER_KEYPOINTS) * 2)
] + [i for i in range(len(MANO_RIGHT_REORDER_KEYPOINTS))]

SINGLE_HAND_FLIP_PERM = [i for i in range(len(MANO_RIGHT_REORDER_KEYPOINTS))]

FEEF_FLIP_PERM = [i for i in range(len(FOOT_NAMES),
                                   len(FOOT_NAMES) * 2)
                  ] + [i for i in range(len(FOOT_NAMES))]

FACE_FLIP_PERM = [
    9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 10, 11, 12, 13, 18, 17, 16, 15, 14, 28, 27,
    26, 25, 30, 29, 22, 21, 20, 19, 24, 23, 37, 36, 35, 34, 33, 32, 31, 42, 41,
    40, 39, 38, 47, 46, 45, 44, 43, 50, 49, 48
]
FACE_FLIP_PERM = FACE_FLIP_PERM + [
    67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51
]
