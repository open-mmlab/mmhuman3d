import smplx.joint_names as smplx_joint_name

# We create a superset of joints containing the OpenPose joints together with
# the ones that each dataset provides. We keep a superset of 24 joints.
# If a dataset doesn't provide annotations for a specific joint,
# we simply ignore it.
# The joints used here are the following:

OP_JOINT_NAMES = [
    # 25 OpenPose joints (in the order provided by OpenPose)
    'OP Nose',
    'OP Neck',
    'OP RShoulder',
    'OP RElbow',
    'OP RWrist',
    'OP LShoulder',
    'OP LElbow',
    'OP LWrist',
    'OP MidHip',
    'OP RHip',
    'OP RKnee',
    'OP RAnkle',
    'OP LHip',
    'OP LKnee',
    'OP LAnkle',
    'OP REye',
    'OP LEye',
    'OP REar',
    'OP LEar',
    'OP LBigToe',
    'OP LSmallToe',
    'OP LHeel',
    'OP RBigToe',
    'OP RSmallToe',
    'OP RHeel',
]
SPIN_JOINT_NAMES = [
    # 24 Ground Truth joints (superset of joints from different datasets)
    'Right Ankle',
    'Right Knee',
    'Right Hip',  # 2
    'Left Hip',
    'Left Knee',  # 4
    'Left Ankle',
    'Right Wrist',  # 6
    'Right Elbow',
    'Right Shoulder',  # 8
    'Left Shoulder',
    'Left Elbow',  # 10
    'Left Wrist',
    'Neck (LSP)',  # 12
    'Top of Head (LSP)',
    'Pelvis (MPII)',  # 14
    'Thorax (MPII)',
    'Spine (H36M)',  # 16
    'Jaw (H36M)',
    'Head (H36M)',  # 18
    'Nose',
    'Left Eye',
    'Right Eye',
    'Left Ear',
    'Right Ear'
]
JOINT_NAMES = OP_JOINT_NAMES + SPIN_JOINT_NAMES

# Dict containing the joints in numerical order
JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}

# Map joints to SMPL joints
JOINT_MAP = {
    'OP Nose': 24,
    'OP Neck': 12,
    'OP RShoulder': 17,
    'OP RElbow': 19,
    'OP RWrist': 21,
    'OP LShoulder': 16,
    'OP LElbow': 18,
    'OP LWrist': 20,
    'OP MidHip': 0,
    'OP RHip': 2,
    'OP RKnee': 5,
    'OP RAnkle': 8,
    'OP LHip': 1,
    'OP LKnee': 4,
    'OP LAnkle': 7,
    'OP REye': 25,
    'OP LEye': 26,
    'OP REar': 27,
    'OP LEar': 28,
    'OP LBigToe': 29,
    'OP LSmallToe': 30,
    'OP LHeel': 31,
    'OP RBigToe': 32,
    'OP RSmallToe': 33,
    'OP RHeel': 34,
    'Right Ankle': 8,
    'Right Knee': 5,
    'Right Hip': 45,
    'Left Hip': 46,
    'Left Knee': 4,
    'Left Ankle': 7,
    'Right Wrist': 21,
    'Right Elbow': 19,
    'Right Shoulder': 17,
    'Left Shoulder': 16,
    'Left Elbow': 18,
    'Left Wrist': 20,
    'Neck (LSP)': 47,
    'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49,
    'Thorax (MPII)': 50,
    'Spine (H36M)': 51,
    'Jaw (H36M)': 52,
    'Head (H36M)': 53,
    'Nose': 24,
    'Left Eye': 26,
    'Right Eye': 25,
    'Left Ear': 28,
    'Right Ear': 27
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

HAND_NAMES = [
    'wrist',
    'thumb1',
    'thumb2',
    'thumb3',
    'thumb',
    'index1',
    'index2',
    'index3',
    'index',
    'middle1',
    'middle2',
    'middle3',
    'middle',
    'ring1',
    'ring2',
    'ring3',
    'ring',
    'pinky1',
    'pinky2',
    'pinky3',
    'pinky',
]

SMPLX_JOINT_NAMES = smplx_joint_name.JOINT_NAMES
SMPLX_JOINT_IDS = {
    SMPLX_JOINT_NAMES[i]: i
    for i in range(len(SMPLX_JOINT_NAMES))
}

FOOT_NAMES = ['big_toe', 'small_toe', 'heel']

FACIAL_LANDMARKS = [
    'right_eye_brow1',
    'right_eye_brow2',
    'right_eye_brow3',
    'right_eye_brow4',
    'right_eye_brow5',
    'left_eye_brow5',
    'left_eye_brow4',
    'left_eye_brow3',
    'left_eye_brow2',
    'left_eye_brow1',
    'nose1',
    'nose2',
    'nose3',
    'nose4',
    'right_nose_2',
    'right_nose_1',
    'nose_middle',
    'left_nose_1',
    'left_nose_2',
    'right_eye1',
    'right_eye2',
    'right_eye3',
    'right_eye4',
    'right_eye5',
    'right_eye6',
    'left_eye4',
    'left_eye3',
    'left_eye2',
    'left_eye1',
    'left_eye6',
    'left_eye5',
    'right_mouth_1',
    'right_mouth_2',
    'right_mouth_3',
    'mouth_top',
    'left_mouth_3',
    'left_mouth_2',
    'left_mouth_1',
    'left_mouth_5',  # 59 in OpenPose output
    'left_mouth_4',  # 58 in OpenPose output
    'mouth_bottom',
    'right_mouth_4',
    'right_mouth_5',
    'right_lip_1',
    'right_lip_2',
    'lip_top',
    'left_lip_2',
    'left_lip_1',
    'left_lip_3',
    'lip_bottom',
    'right_lip_3',
    'right_contour_1',
    'right_contour_2',
    'right_contour_3',
    'right_contour_4',
    'right_contour_5',
    'right_contour_6',
    'right_contour_7',
    'right_contour_8',
    'contour_middle',
    'left_contour_8',
    'left_contour_7',
    'left_contour_6',
    'left_contour_5',
    'left_contour_4',
    'left_contour_3',
    'left_contour_2',
    'left_contour_1',
]

# LRHAND_FLIP_PERM = [i for i in range(16, 32)] + [i for i in range(16)]
LRHAND_FLIP_PERM = [i for i in range(len(HAND_NAMES),
                                     len(HAND_NAMES) * 2)
                    ] + [i for i in range(len(HAND_NAMES))]

SINGLE_HAND_FLIP_PERM = [i for i in range(len(HAND_NAMES))]

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
