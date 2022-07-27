"""These keypoint formats are taken from https://github.com/CMU-Perceptual-
Computing-Lab/openpose/blob/master/src/openpose/pose/poseParameters.cpp.
Openpose mainly supports 25 and 135 now, 118 convention can be found in
https://github.com/vchoutas/smplify-x/issues/152#issuecomment-923715702.

OPENPOSE_137_KEYPOINTS can be found in
https://github.com/vchoutas/expose

- OPENPOSE_25_KEYPOINTS:  body(25)
- OPENPOSE_118_KEYPOINTS: body(25) + hand(42) + face(51)
- OPENPOSE_135_KEYPOINTS: body(25) + hand(40) + face(70)
- OPENPOSE_137_KEYPOINTS: body(27) + hand(40) + face(70)

Note that:
1. 135 and coco17 share the first 17 body keypoints
2. 25 and 118 share the first 25 body keypoints
3. 137 and 135 share the hand and face parts
"""

OPENPOSE_135_KEYPOINTS = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
    'neck',  # upper_neck
    'head',
    'left_bigtoe',
    'left_smalltoe',
    'left_heel',
    'right_bigtoe',
    'right_smalltoe',
    'right_heel',
    'left_thumb_1',
    'left_thumb_2',
    'left_thumb_3',
    'left_thumb',
    'left_index_1',
    'left_index_2',
    'left_index_3',
    'left_index',
    'left_middle_1',
    'left_middle_2',
    'left_middle_3',
    'left_middle',
    'left_ring_1',
    'left_ring_2',
    'left_ring_3',
    'left_ring',
    'left_pinky_1',
    'left_pinky_2',
    'left_pinky_3',
    'left_pinky',
    'right_thumb_1',
    'right_thumb_2',
    'right_thumb_3',
    'right_thumb',
    'right_index_1',
    'right_index_2',
    'right_index_3',
    'right_index',
    'right_middle_1',
    'right_middle_2',
    'right_middle_3',
    'right_middle',
    'right_ring_1',
    'right_ring_2',
    'right_ring_3',
    'right_ring',
    'right_pinky_1',
    'right_pinky_2',
    'right_pinky_3',
    'right_pinky',
    'right_contour_1',  # original name: face_contour_1
    'right_contour_2',  # original name: face_contour_2
    'right_contour_3',  # original name: face_contour_3
    'right_contour_4',  # original name: face_contour_4
    'right_contour_5',  # original name: face_contour_5
    'right_contour_6',  # original name: face_contour_6
    'right_contour_7',  # original name: face_contour_7
    'right_contour_8',  # original name: face_contour_8
    'contour_middle',  # original name: face_contour_9
    'left_contour_8',  # original name: face_contour_10
    'left_contour_7',  # original name: face_contour_11
    'left_contour_6',  # original name: face_contour_12
    'left_contour_5',  # original name: face_contour_13
    'left_contour_4',  # original name: face_contour_14
    'left_contour_3',  # original name: face_contour_15
    'left_contour_2',  # original name: face_contour_16
    'left_contour_1',  # original name: face_contour_17
    'right_eyebrow_1',
    'right_eyebrow_2',
    'right_eyebrow_3',
    'right_eyebrow_4',
    'right_eyebrow_5',
    'left_eyebrow_5',
    'left_eyebrow_4',
    'left_eyebrow_3',
    'left_eyebrow_2',
    'left_eyebrow_1',
    'nosebridge_1',
    'nosebridge_2',
    'nosebridge_3',
    'nosebridge_4',
    'right_nose_2',  # original name: nose_1
    'right_nose_1',  # original name: nose_2
    'nose_middle',  # original name: nose_3
    'left_nose_1',  # original name: nose_4
    'left_nose_2',  # original name: nose_5
    'right_eye_1',
    'right_eye_2',
    'right_eye_3',
    'right_eye_4',
    'right_eye_5',
    'right_eye_6',
    'left_eye_4',
    'left_eye_3',
    'left_eye_2',
    'left_eye_1',
    'left_eye_6',
    'left_eye_5',
    'right_mouth_1',  # original name: mouth_1
    'right_mouth_2',  # original name: mouth_2
    'right_mouth_3',  # original name: mouth_3
    'mouth_top',  # original name: mouth_4
    'left_mouth_3',  # original name: mouth_5
    'left_mouth_2',  # original name: mouth_6
    'left_mouth_1',  # original name: mouth_7
    'left_mouth_5',  # original name: mouth_8
    'left_mouth_4',  # original name: mouth_9
    'mouth_bottom',  # original name: mouth_10
    'right_mouth_4',  # original name: mouth_11
    'right_mouth_5',  # original name: mouth_12
    'right_lip_1',  # original name: lip_1
    'right_lip_2',  # original name: lip_2
    'lip_top',  # original name: lip_3
    'left_lip_2',  # original name: lip_4
    'left_lip_1',  # original name: lip_5
    'left_lip_3',  # original name: lip_6
    'lip_bottom',  # original name: lip_7
    'right_lip_3',  # original name: lip_8
    'right_eyeball',
    'left_eyeball'
]

# TODO: OPENPOSE-25->HumanData->SMPLX causes the whole body to be lost
# OPENPOSE-25: nose_openpose
# SMPLX: nose

OPENPOSE_25_KEYPOINTS = [
    'nose_openpose',
    'neck_openpose',  # 'upper_neck'
    'right_shoulder_openpose',
    'right_elbow_openpose',
    'right_wrist_openpose',
    'left_shoulder_openpose',
    'left_elbow_openpose',
    'left_wrist_openpose',
    'pelvis_openpose',  # 'mid_hip'
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
    'right_heel_openpose'
]

OPENPOSE_118_KEYPOINTS = [
    'nose_openpose',
    'neck_openpose',
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
    'left_wrist',
    'left_thumb_1',
    'left_thumb_2',
    'left_thumb_3',
    'left_thumb',
    'left_index_1',
    'left_index_2',
    'left_index_3',
    'left_index',
    'left_middle_1',
    'left_middle_2',
    'left_middle_3',
    'left_middle',
    'left_ring_1',
    'left_ring_2',
    'left_ring_3',
    'left_ring',
    'left_pinky_1',
    'left_pinky_2',
    'left_pinky_3',
    'left_pinky',
    'right_wrist',
    'right_thumb_1',
    'right_thumb_2',
    'right_thumb_3',
    'right_thumb',
    'right_index_1',
    'right_index_2',
    'right_index_3',
    'right_index',
    'right_middle_1',
    'right_middle_2',
    'right_middle_3',
    'right_middle',
    'right_ring_1',
    'right_ring_2',
    'right_ring_3',
    'right_ring',
    'right_pinky_1',
    'right_pinky_2',
    'right_pinky_3',
    'right_pinky',
    'right_eyebrow_1',
    'right_eyebrow_2',
    'right_eyebrow_3',
    'right_eyebrow_4',
    'right_eyebrow_5',
    'left_eyebrow_5',
    'left_eyebrow_4',
    'left_eyebrow_3',
    'left_eyebrow_2',
    'left_eyebrow_1',
    'nosebridge_1',
    'nosebridge_2',
    'nosebridge_3',
    'nosebridge_4',
    'right_nose_2',  # original name: nose_1
    'right_nose_1',  # original name: nose_2
    'nose_middle',  # original name: nose_3
    'left_nose_1',  # original name: nose_4
    'left_nose_2',  # original name: nose_5
    'right_eye_1',
    'right_eye_2',
    'right_eye_3',
    'right_eye_4',
    'right_eye_5',
    'right_eye_6',
    'left_eye_4',
    'left_eye_3',
    'left_eye_2',
    'left_eye_1',
    'left_eye_6',
    'left_eye_5',
    'right_mouth_1',  # original name: mouth_1
    'right_mouth_2',  # original name: mouth_2
    'right_mouth_3',  # original name: mouth_3
    'mouth_top',  # original name: mouth_4
    'left_mouth_3',  # original name: mouth_5
    'left_mouth_2',  # original name: mouth_6
    'left_mouth_1',  # original name: mouth_7
    'left_mouth_5',  # original name: mouth_8
    'left_mouth_4',  # original name: mouth_9
    'mouth_bottom',  # original name: mouth_10
    'right_mouth_4',  # original name: mouth_11
    'right_mouth_5',  # original name: mouth_12
    'right_lip_1',  # original name: lip_1
    'right_lip_2',  # original name: lip_2
    'lip_top',  # original name: lip_3
    'left_lip_2',  # original name: lip_4
    'left_lip_1',  # original name: lip_5
    'left_lip_3',  # original name: lip_6
    'lip_bottom',  # original name: lip_7
    'right_lip_3',  # original name: lip_8
]

OPENPOSE_JOINTS = [
    'nose',
    'neck',
    'right_shoulder',
    'right_elbow',
    'right_wrist',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'pelvis',
    'right_hip',
    'right_knee',
    'right_ankle',
    'left_hip',
    'left_knee',
    'left_ankle',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
    'left_wrist_openpose',
    'left_thumb_1',
    'left_thumb_2',
    'left_thumb_3',
    'left_thumb',
    'left_index_1',
    'left_index_2',
    'left_index_3',
    'left_index',
    'left_middle_1',
    'left_middle_2',
    'left_middle_3',
    'left_middle',
    'left_ring_1',
    'left_ring_2',
    'left_ring_3',
    'left_ring',
    'left_pinky_1',
    'left_pinky_2',
    'left_pinky_3',
    'left_pinky',
    'right_wrist_openpose',
    'right_thumb_1',
    'right_thumb_2',
    'right_thumb_3',
    'right_thumb',
    'right_index_1',
    'right_index_2',
    'right_index_3',
    'right_index',
    'right_middle_1',
    'right_middle_2',
    'right_middle_3',
    'right_middle',
    'right_ring_1',
    'right_ring_2',
    'right_ring_3',
    'right_ring',
    'right_pinky_1',
    'right_pinky_2',
    'right_pinky_3',
    'right_pinky',
    # Face contour
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
    # Eye brows
    'right_eye_brow_1',
    'right_eye_brow_2',
    'right_eye_brow_3',
    'right_eye_brow_4',
    'right_eye_brow_5',
    'left_eye_brow_5',
    'left_eye_brow_4',
    'left_eye_brow_3',
    'left_eye_brow_2',
    'left_eye_brow_1',
    'nosebridge_1',
    'nosebridge_2',
    'nosebridge_3',
    'nosebridge_4',
    'right_nose_2',
    'right_nose_1',
    'nose_middle',
    'left_nose_1',
    'left_nose_2',
    'right_eye_1',
    'right_eye_2',
    'right_eye_3',
    'right_eye_4',
    'right_eye_5',
    'right_eye_6',
    'left_eye_4',
    'left_eye_3',
    'left_eye_2',
    'left_eye_1',
    'left_eye_6',
    'left_eye_5',
    'right_mouth_1',
    'right_mouth_2',
    'right_mouth_3',
    'mouth_top',
    'left_mouth_3',
    'left_mouth_2',
    'left_mouth_1',
    'left_mouth_5',
    'left_mouth_4',
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
    'right_eyeball_unused',  # not used in expose
    'left_eyeball_unused',  # not used in expose
]

OPENPOSE_FEET_KEYPOINTS = [
    'left_bigtoe', 'left_smalltoe', 'left_heel', 'right_bigtoe',
    'right_smalltoe', 'right_heel'
]
OPENPOSE_137_KEYPOINTS = OPENPOSE_JOINTS[:19] + \
    OPENPOSE_FEET_KEYPOINTS + OPENPOSE_JOINTS[19:]
