SMPLX_KEYPOINTS = [
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
    'jaw',
    'left_eyeball',
    'right_eyeball',
    'left_index_1',
    'left_index_2',
    'left_index_3',
    'left_middle_1',
    'left_middle_2',
    'left_middle_3',
    'left_pinky_1',
    'left_pinky_2',
    'left_pinky_3',
    'left_ring_1',
    'left_ring_2',
    'left_ring_3',
    'left_thumb_1',
    'left_thumb_2',
    'left_thumb_3',
    'right_index_1',
    'right_index_2',
    'right_index_3',
    'right_middle_1',
    'right_middle_2',
    'right_middle_3',
    'right_pinky_1',
    'right_pinky_2',
    'right_pinky_3',
    'right_ring_1',
    'right_ring_2',
    'right_ring_3',
    'right_thumb_1',
    'right_thumb_2',
    'right_thumb_3',
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
]

SMPLX_LIMBS = {
    'body': [['pelvis', 'left_hip'], ['pelvis', 'right_hip'],
             ['left_hip', 'right_hip'], ['left_shoulder', 'right_shoulder'],
             ['pelvis', 'spine_1'], ['spine_1', 'spine_2'],
             ['spine_2', 'spine_3'], ['spine_3', 'neck'], ['neck', 'head'],
             ['left_ankle', 'left_knee'], ['left_knee', 'left_hip'],
             ['right_ankle', 'right_knee'], ['right_knee', 'right_hip'],
             ['right_ankle', 'right_foot'], ['left_ankle', 'left_foot'],
             ['left_hip', 'right_hip'], ['left_shoulder', 'left_hip'],
             ['right_shoulder', 'right_hip'], ['left_collar', 'spine_3'],
             ['right_collar', 'spine_3'], ['right_collar', 'right_shoulder'],
             ['left_collar', 'left_shoulder'],
             ['left_shoulder', 'right_shoulder'],
             ['left_shoulder',
              'left_elbow'], ['right_shoulder', 'right_elbow'],
             ['left_elbow', 'left_wrist'], ['right_elbow', 'right_wrist'],
             ['left_ankle', 'left_bigtoe'], ['left_ankle', 'left_smalltoe'],
             ['left_ankle', 'left_heel'], ['right_ankle', 'right_bigtoe'],
             ['right_ankle', 'right_smalltoe'], ['right_ankle', 'right_heel'],
             ['left_shoulder', 'left_ear'], ['right_shoulder', 'right_ear'],
             ['right_ear', 'right_eye'], ['right_eye', 'nose'],
             ['nose', 'left_eye'], ['left_eye', 'left_ear'], ['nose', 'jaw'],
             ['jaw', 'neck']],
    'face': [['right_contour_1', 'right_contour_2'],
             ['right_contour_2', 'right_contour_3'],
             ['right_contour_3', 'right_contour_4'],
             ['right_contour_4', 'right_contour_5'],
             ['right_contour_5', 'right_contour_6'],
             ['right_contour_6', 'right_contour_7'],
             ['right_contour_7', 'right_contour_8'],
             ['right_contour_8', 'contour_middle'],
             ['contour_middle', 'left_contour_8'],
             ['left_contour_8', 'left_contour_7'],
             ['left_contour_7', 'left_contour_6'],
             ['left_contour_6', 'left_contour_5'],
             ['left_contour_5', 'left_contour_4'],
             ['left_contour_4', 'left_contour_3'],
             ['left_contour_3', 'left_contour_2'],
             ['left_contour_2', 'left_contour_1']],
    'left_hand':
    [['left_wrist', 'left_thumb_1'], ['left_thumb_1', 'left_thumb_2'],
     ['left_thumb_2', 'left_thumb_3'], ['left_thumb_3', 'left_thumb'],
     ['left_wrist', 'left_index_1'], ['left_index_1', 'left_index_2'],
     ['left_index_2', 'left_index_3'], ['left_index_3', 'left_index'],
     ['left_wrist', 'left_middle_1'], ['left_middle_1', 'left_middle_2'],
     ['left_middle_2', 'left_middle_3'], ['left_middle_3', 'left_middle'],
     ['left_wrist', 'left_ring_1'], ['left_ring_1', 'left_ring_2'],
     ['left_ring_2', 'left_ring_3'], ['left_ring_3', 'left_ring'],
     ['left_wrist', 'left_pinky_1'], ['left_pinky_1', 'left_pinky_2'],
     ['left_pinky_2', 'left_pinky_3'], ['left_pinky_3', 'left_pinky']],
    'right_hand': [['right_wrist', 'right_thumb_1'],
                   ['right_thumb_1', 'right_thumb_2'],
                   ['right_thumb_2', 'right_thumb_3'],
                   ['right_thumb_3', 'right_thumb'],
                   ['right_wrist', 'right_index_1'],
                   ['right_index_1', 'right_index_2'],
                   ['right_index_2', 'right_index_3'],
                   ['right_index_3', 'right_index'],
                   ['right_wrist', 'right_middle_1'],
                   ['right_middle_1', 'right_middle_2'],
                   ['right_middle_2', 'right_middle_3'],
                   ['right_middle_3', 'right_middle'],
                   ['right_wrist', 'right_ring_1'],
                   ['right_ring_1', 'right_ring_2'],
                   ['right_ring_2', 'right_ring_3'],
                   ['right_ring_3', 'right_ring'],
                   ['right_wrist', 'right_pinky_1'],
                   ['right_pinky_1', 'right_pinky_2'],
                   ['right_pinky_2', 'right_pinky_3'],
                   ['right_pinky_3', 'right_pinky']],
    'right_eye':
    [['right_eye_1', 'right_eye_2'], ['right_eye_2', 'right_eye_3'],
     ['right_eye_3', 'right_eye_4'], ['right_eye_4', 'right_eye_5'],
     ['right_eye_5', 'right_eye_6'], ['right_eye_6', 'right_eye_1'],
     ['right_eyebrow_1', 'right_eyebrow_2'],
     ['right_eyebrow_2', 'right_eyebrow_3'],
     ['right_eyebrow_3', 'right_eyebrow_4'],
     ['right_eyebrow_4', 'right_eyebrow_5']],
    'left_eye': [['left_eye_4', 'left_eye_3'], ['left_eye_3', 'left_eye_2'],
                 ['left_eye_2', 'left_eye_1'], ['left_eye_1', 'left_eye_6'],
                 ['left_eye_6', 'left_eye_5'], ['left_eye_5', 'left_eye_4'],
                 ['left_eyebrow_1', 'left_eyebrow_2'],
                 ['left_eyebrow_2', 'left_eyebrow_3'],
                 ['left_eyebrow_3', 'left_eyebrow_4'],
                 ['left_eyebrow_4', 'left_eyebrow_5']],
    'mouth':
    [['right_mouth_1', 'right_mouth_2'], ['right_mouth_2', 'right_mouth_3'],
     ['right_mouth_3', 'mouth_top'], ['mouth_top', 'left_mouth_3'],
     ['left_mouth_3', 'left_mouth_2'], ['left_mouth_2', 'left_mouth_1'],
     ['left_mouth_1', 'left_mouth_5'], ['left_mouth_5', 'left_mouth_4'],
     ['left_mouth_4', 'mouth_bottom'], ['mouth_bottom', 'right_mouth_4'],
     ['right_mouth_4', 'right_mouth_5'], ['right_mouth_5', 'right_mouth_1'],
     ['right_lip_1', 'right_lip_2'], ['right_lip_2', 'lip_top'],
     ['lip_top', 'left_lip_2'], ['left_lip_2', 'left_lip_1'],
     ['left_lip_1', 'left_lip_3'], ['left_lip_3', 'lip_bottom'],
     ['lip_bottom', 'right_lip_3'], ['right_lip_3', 'right_lip_1']],
    'nose': [
        ['nosebridge_1', 'nosebridge_2'],
        ['nosebridge_2', 'nosebridge_3'],
        ['nosebridge_3', 'nosebridge_4'],
        ['right_nose_2', 'right_nose_1'],
        ['right_nose_1', 'nose_middle'],
        ['nose_middle', 'left_nose_1'],
        ['left_nose_1', 'left_nose_2'],
    ]
}

SMPLX_LIMBS_INDEX = {}
for k in SMPLX_LIMBS:
    SMPLX_LIMBS_INDEX[k] = [[
        SMPLX_KEYPOINTS.index(limb[0]),
        SMPLX_KEYPOINTS.index(limb[1])
    ] for limb in SMPLX_LIMBS[k]]

SMPLX_PALETTE = {
    'left_eye': [[0, 0, 0]],
    'right_eye': [[0, 0, 0]],
    'nose': [[0, 0, 255]],
    'mouth': [[0, 255, 255]],
    'face': [[255, 0, 0]],
    'left_hand': [[0, 0, 0]],
    'right_hand': [[0, 0, 0]]
}
