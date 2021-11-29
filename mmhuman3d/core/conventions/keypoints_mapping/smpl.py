# the keypoints defined in the SMPL paper
SMPL_KEYPOINTS = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine_1',
    'left_knee',
    'right_knee',
    'spine_2',
    'left_ankle',
    'right_ankle',
    'spine_3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hand',
    'right_hand',
]

# the full keypoints produced by the default SMPL J_regressor
SMPL_45_KEYPOINTS = SMPL_KEYPOINTS + [
    'nose',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
    'left_bigtoe',
    'left_smalltoe',
    'left_heel',
    'right_bigtoe',
    'right_smalltoe',
    'right_heel',
    'left_thumb',
    'left_index',
    'left_middle',
    'left_ring',
    'left_pinky',
    'right_thumb',
    'right_index',
    'right_middle',
    'right_ring',
    'right_pinky',
]

# the full keypoints produced by the default SMPL J_regressor and
# extra_J_regressor (provided by SPIN)
SMPL_54_KEYPOINTS = SMPL_45_KEYPOINTS + [
    'right_hip_extra',  # LSP
    'left_hip_extra',  # LSP
    'neck_extra',  # LSP
    'headtop',  # LSP
    'pelvis_extra',  # MPII
    'thorax_extra',  # MPII
    'spine_extra',  # H36M
    'jaw_extra',  # H36M
    'head_extra',  # H36M
]

# SMPL keypoint convention used by SPIN, EFT and so on
SMPL_49_KEYPOINTS = [
    # OpenPose
    'nose_openpose',
    'neck_openpose',  # 'upper_neck'
    'right_shoulder_openpose',
    'right_elbow_openpose',
    'right_wrist_openpose',
    'left_shoulder_openpose',
    'left_elbow_openpose',
    'left_wrist_openpose',
    'pelvis_openpose',
    'right_hip_openpose',
    'right_knee_openpose',
    'right_ankle_openpose',
    'left_hip_openpose',
    'left_knee_openpose',
    'left_ankle_openpose',
    'right_eye_openpose',
    'left_eye_openpose',
    'right_ear_openpose',
    'left_ear_openpose',
    'left_bigtoe_openpose',
    'left_smalltoe_openpose',
    'left_heel_openpose',
    'right_bigtoe_openpose',
    'right_smalltoe_openpose',
    'right_heel_openpose',
    # 24 Keypoints
    'right_ankle',
    'right_knee',
    'right_hip_extra',  # LSP
    'left_hip_extra',  # LSP
    'left_knee',
    'left_ankle',
    'right_wrist',
    'right_elbow',
    'right_shoulder',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'neck_extra',  # LSP
    'headtop',  # LSP mpii peen_action mpi_inf_3dhp
    'pelvis_extra',  # MPII
    'thorax_extra',  # MPII
    'spine_extra',  # H36M
    'jaw_extra',  # H36M
    'head_extra',  # H36M
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear'
]

SMPL_24_KEYPOINTS = SMPL_49_KEYPOINTS[-24:]
