BODY_KEYPOINTS = [
    'nose',
    'left_eye_4',
    'left_eye',
    'left_eye_1',
    'right_eye_4',
    'right_eye',
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
    'left_pinky_1',
    'right_pinky_1',
    'left_index_1',
    'right_index_1',
    'left_thumb_1',
    'right_thumb_1',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
    'left_heel',
    'right_heel',
    'left_foot',
    'right_foot',
]


RIGHT_HAND_KEYPOINTS = [
    'right_wrist', 'right_thumb_1', 'right_thumb_2', 'right_thumb_3',
    'right_thumb', 'right_index_1', 'right_index_2', 'right_index_3',
    'right_index', 'right_middle_1', 'right_middle_2', 'right_middle_3',
    'right_middle', 'right_ring_1', 'right_ring_2', 'right_ring_3',
    'right_ring', 'right_pinky_1', 'right_pinky_2', 'right_pinky_3',
    'right_pinky'
]

LEFT_HAND_KEYPOINTS = [
    x.replace('right_', 'left_') for x in RIGHT_HAND_KEYPOINTS
]

MANO_HANDS_REORDER_KEYPOINTS = \
        LEFT_HAND_KEYPOINTS + RIGHT_HAND_KEYPOINTS

WHOLE_BODY_KEYPOINTS = [
    'nose',
    'left_eye_4',
    'left_eye',
    'left_eye_1',
    'right_eye_4',
    'right_eye',
    'right_eye_1',
    'left_ear',
    'right_ear',
    'left_mouth_1',
    'right_mouth_1',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
    'left_heel',
    'right_heel',
    'left_foot',
    'right_foot',
]

WHOLE_BODY_KEYPOINTS = WHOLE_BODY_KEYPOINTS + MANO_HANDS_REORDER_KEYPOINTS