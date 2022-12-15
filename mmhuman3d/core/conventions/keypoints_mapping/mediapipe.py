from .mano import MANO_HANDS_REORDER_KEYPOINTS

# mediapipe pose convention defined in
# https://google.github.io/mediapipe/solutions/pose.html
MP_BODY_KEYPOINTS = [
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

# remove hand keypoints in mediapipe pose
MP_WHOLE_BODY_KEYPOINTS = [
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

# mediapipe keypoint of body and hands
MP_WHOLE_BODY_KEYPOINTS =\
            MP_WHOLE_BODY_KEYPOINTS + MANO_HANDS_REORDER_KEYPOINTS
