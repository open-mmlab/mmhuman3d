BODY_KEYPOINTS = [
    'nose',
    'left_eye_4',
    'left_eyeball',
    'left_eye_1',
    'right_eye_4',
    'right_eyeball',
    'right_eye_1',
    'left_ear',
    'right_ear',
    'left_mouth_1',
    'right_mouth_1',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_pinky',
    'right_pinky',
    'left_index',
    'right_index',
    'left_thumb',
    'right_thumb',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',#
    'left_ankle',#
    'right_ankle',#
    'left_heel',#
    'right_heel',#
    'left_foot',#
    'right_foot', #
]

BODY_MIRROR_KEYPOINTS = [x.replace('right_', 'left_') if 'right_' in x else x.replace('left_', 'right_') for x in BODY_KEYPOINTS]

MANO_RIGHT_KEYPOINTS = [
    'right_wrist', 'right_index_1', 'right_index_2', 'right_index_3',
    'right_middle_1', 'right_middle_2', 'right_middle_3', 'right_pinky_1',
    'right_pinky_2', 'right_pinky_3', 'right_ring_1', 'right_ring_2',
    'right_ring_3', 'right_thumb_1', 'right_thumb_2', 'right_thumb_3',
    'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_pinky'
]

MANO_LEFT_KEYPOINTS = [
    x.replace('right_', 'left_') for x in MANO_RIGHT_KEYPOINTS
]

# Re-arranged order is compatible with the output of manolayer
# from official [manopth](https://github.com/hassony2/manopth)
# remove wrist
MANO_REORDER_MAP = [
    13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20
]

MANO_RIGHT_REORDER_KEYPOINTS = [
    MANO_RIGHT_KEYPOINTS[i] for i in MANO_REORDER_MAP
]
MANO_LEFT_REORDER_KEYPOINTS = [
    MANO_LEFT_KEYPOINTS[i] for i in MANO_REORDER_MAP
]

# Deprecated: reserved for backward compatibility
MANO_KEYPOINTS = MANO_RIGHT_KEYPOINTS
# Two hands (left + right)
MANO_HANDS_KEYPOINTS = MANO_LEFT_KEYPOINTS + MANO_RIGHT_KEYPOINTS
# Reordered two hands (left + right)
MANO_HANDS_REORDER_KEYPOINTS = \
        MANO_LEFT_REORDER_KEYPOINTS + MANO_RIGHT_REORDER_KEYPOINTS

WHOLE_BODY_KEYPOINTS = BODY_KEYPOINTS + MANO_HANDS_REORDER_KEYPOINTS