""" TODO
1. save smplify stages config elsewhere
2. merge smplify opt config with normal optimizers
3. add camera
4. add 2D loss support
5. **use the convention tools**
6. add GMM prior
7. create GTA tools in zoehuman/

optional:
1. optimize how to get batch_size and num_frames
2. add default body model
3. use from losses.py for loss computation
4. add model inference for param init
5. check if SMPL layer is better
"""

import numpy as np
import torch
from configs.smplify.smplify import smplify_opt_config, smplify_stages
from configs.smplify.smplifyx import smplifyx_opt_config, smplifyx_stages

# TODO: placeholder
default_camera = {}


def unify_joint_mappings_smpl(dataset='smpl'):
    """Unify different joint definitions to SMPL.

    Output unified definition (use SMPL):
        [
        # smpl joints
        'pelvis',           #0
        'left_hip',         #1
        'right_hip',        #2
        'spine1',           #3
        'left_knee',        #4
        'right_knee',       #5
        'spine2',           #6
        'left_ankle',       #7
        'right_ankle',      #8
        'spine3',           #9
        'left_foot',        #10
        'right_foot',       #11
        'neck',             #12
        'left_collar',      #13
        'right_collar',     #14
        'head',             #15
        'left_shoulder',    #16
        'right_shoulder',   #17
        'left_elbow',       #18
        'right_elbow',      #19
        'left_wrist',       #20
        'right_wrist',      #21
        'left_hand',        #22
        'right_hand',       #23
        # additional keypoints
        'nose',             #24
        'left eye'          #25
        'right eye'         #26
        'left ear'          #27
        'right ear'         #28
    ]

    Args:
      dataset: `smpl` and `gta`.
    Returns:
      a list of indexes that maps the joints to a SMPL convention.
      -1 denotes the joint is missing.
    """
    if dataset == 'smpl':
        return np.array([*range(0, 29)], dtype=np.int32)
    elif dataset == 'gta':
        return np.array([
            14,  # 00 pelvis (SMPL)     - 14 spine3 (GTA)
            19,  # 01 left hip          - 19 left hip
            16,  # 02 right hip         - 16 right hip
            13,  # 03 spine1            - 13 spine2
            20,  # 04 left knee         - 20 left knee
            17,  # 05 right knee        - 17 right knee
            11,  # 06 spine2            - 11 spine0
            21,  # 07 left ankle        - 21 left ankle
            18,  # 08 right ankle       - 18 right ankle
            -1,  # 09 spine3            - no match
            24,  # 10 left foot         - 24 SKEL_L_Toe0
            49,  # 11 right foot        - 49 SKEL_R_Toe0
            2,  # 12 neck              - 02 neck
            -1,  # 13 left clavicle     - no match, 07 left clavicle
            # different convention
            -1,  # 14 right clavicle    - no match, 03 right clavicle
            # different convention
            1,  # 15 head              - 01 head center
            8,  # 16 left shoulder     - 08 left shoulder
            4,  # 17 right shoulder    - 04 right shoulder
            9,  # 18 left elbow        - 09 left elbow
            5,  # 19 right elbow       - 05 right elbow
            55,  # 20 left wrist        - 55 left wrist
            6,  # 21 right wrist       - 06 right wrist
            57,  # 22 left_hand         - 57 SKEL_L_Finger20
            # (left middle finger root)
            81,  # 23 right_hand        - 81 SKEL_R_Finger20
            # (right middle finger root)
            99,  # 24 nose              - 99 interpolated nose
            60,  # 25 right eye         - 54 right eye
            54,  # 26 left eye          - 60 left eye
            -1,  # 27 right ear         - no match
            -1,  # 28 left ear          - no match
        ])
    else:
        raise ValueError(f'{dataset} is not supported')


def unify_joint_mappings(dataset='smplx'):
    """Unify different joint definitions to SMPL-X.

    Output unified definition:
        [
        'pelvis',           #0
        'left_hip',         #1
        'right_hip',        #2
        'spine1',           #3
        'left_knee',        #4
        'right_knee',       #5
        'spine2',           #6
        'left_ankle',       #7
        'right_ankle',      #8
        'spine3',           #9
        'left_foot',        #10
        'right_foot',       #11
        'neck',             #12
        'left_collar',      #13
        'right_collar',     #14
        'head',             #15
        'left_shoulder',    #16
        'right_shoulder',   #17
        'left_elbow',       #18
        'right_elbow',      #19
        'left_wrist',       #20
        'right_wrist',      #21
        'jaw',              #22
        'left_eye_smplhf',  #23
        'right_eye_smplhf', #24
        'left_index1',      #25
        'left_index2',
        'left_index3',
        'left_middle1',
        'left_middle2',
        'left_middle3',
        'left_pinky1',
        'left_pinky2',
        'left_pinky3',
        'left_ring1',
        'left_ring2',
        'left_ring3',
        'left_thumb1',
        'left_thumb2',
        'left_thumb3',
        'right_index1',     #40
        'right_index2',
        'right_index3',
        'right_middle1',
        'right_middle2',
        'right_middle3',
        'right_pinky1',
        'right_pinky2',
        'right_pinky3',
        'right_ring1',
        'right_ring2',
        'right_ring3',
        'right_thumb1',
        'right_thumb2',
        'right_thumb3',
        'nose',             #55
        'right_eye',        #56
        'left_eye',         #57
        'right_ear',        #58
        'left_ear',         #59
        'left_big_toe',     #60
        'left_small_toe',   #61
        'left_heel',        #62
        'right_big_toe',    #63
        'right_small_toe',  #64
        'right_heel',       #65
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
    ]

    Args:
      dataset: `smplx`, `smpl` and `gta`.
    Returns:
      a list of indexes that maps the joints to a SMPL-X convention.
      -1 denotes the joint is missing.
    """
    if dataset == 'smplx':
        return np.array([*range(0, 127)], dtype=np.int32)
    elif dataset == 'smpl':
        return np.array([
            *range(0, 22),
            *[-1 for _ in range(3)],
            22,
            *[-1 for _ in range(14)],
            23,
            *[-1 for _ in range(86)],
        ],
                        dtype=np.int32)
    elif dataset == 'gta':
        # smplx - gta
        # return np.array([
        #     14,  # 00 pelvis - pelvis
        #     19,  # 01 left hip - 19 left hip
        #     16,  # 02 right hip - 16 right hip
        #     13,  # 03 spine1 - spine2
        #     20,  # 04 left knee - left knee
        #     17,  # 05 right knee - right knee
        #     11,  # 06 spine2 - spine0
        #     21,  # 07 left ankle - left ankle
        #     18,  # 08 right ankle - right ankle
        #     -1,  # 09 spine3 - no match
        #     24,  # 10 left foot - SKEL_L_Toe0
        #     49,  # 11 right foot - SKEL_R_Toe0
        #     2,   # 12 neck - neck
        #     -1,   # 13 left clavicle - 07 left clavicle
        #     -1,   # 14 right clavicle - 03 right clavicle
        #     1,   # 15 head - head center
        #     8,   # 16 left shoulder - left shoulder
        #     4,   # 17 right shoulder - right shoulder
        #     9,   # 18 left elbow - left elbow
        #     5,   # 19 right elbow - right elbow
        #     10,  # 20 left wrist - left wrist
        #     6,   # 21 right wrist - right wrist
        #     -1,  # 22 jaw - no match
        #     *([-1] * 2),
        #     56,  # 25 left_index1
        #     32,  # 26 left_index2
        #     33,  # 27 left_index3
        #     57,  # 28 left_middle1,
        #     34,  # 29 left_middle2,
        #     35,  # 30 left_middle3,
        #     59,  # 31 left_pinky1,
        #     30,  # 32 left_pinky2,
        #     31,  # 33 left_pinky3,
        #     58,  # 34 left_ring1,
        #     28,  # 35 left_ring2,
        #     29,  # 36 left_ring3,
        #     -1,  # 37 left_thumb1,
        #     26,  # 38 left_thumb2,
        #     27,  # 39 left_thumb3,
        #     80,  # 40 right_index1,
        #     93,  # 41 right_index2,
        #     94,  # 42 right_index3,
        #     81,  # 43 right_middle1,
        #     95,  # 44 right_middle2,
        #     96,  # 45 right_middle3,
        #     83,  # 46 right_pinky1,
        #     91,  # 47 right_pinky2,
        #     92,  # 48 right_pinky3,
        #     82,  # 49 right_ring1,
        #     89,  # 50 right_ring2,
        #     90,  # 51 right_ring3,
        #     -1,  # 52 right_thumb1,
        #     87,  # 53 right_thumb2,
        #     88,  # 54 right_thumb3,
        #     99,  # 55 nose - interpolated nose
        #     60,  # 56 right eye - right eye
        #     54,  # 57 left eye - left eye
        #     -1,  # 58
        #     -1,  # 59
        #     -1,  # 60
        #     -1,  # 61
        #     -1,  # 62
        #     -1,  # 63
        #     -1,  # 64
        #     -1,  # 65
        #     -1,  # 66
        #     -1,  # 67
        #     -1,  # 68
        #     -1,  # 69
        #     -1,  # 70
        #     -1,  # 71
        #     -1,  # 72
        #     -1,  # 73
        #     -1,  # 74
        #     -1,  # 75
        #     *([-1] * 69)
        # ])
        return np.array([
            14,  # 00 pelvis - pelvis
            19,  # 01 left hip - 19 left hip
            16,  # 02 right hip - 16 right hip
            13,  # 03 spine1 - spine2
            20,  # 04 left knee - left knee
            17,  # 05 right knee - right knee
            11,  # 06 spine2 - spine0
            21,  # 07 left ankle - left ankle
            18,  # 08 right ankle - right ankle
            -1,  # 09 spine3 - no match
            24,  # 10 left foot - SKEL_L_Toe0
            49,  # 11 right foot - SKEL_R_Toe0
            2,  # 12 neck - neck
            -1,  # 13 left clavicle - 07 left clavicle
            -1,  # 14 right clavicle - 03 right clavicle
            1,  # 15 head - head center
            8,  # 16 left shoulder - left shoulder
            4,  # 17 right shoulder - right shoulder
            9,  # 18 left elbow - left elbow
            5,  # 19 right elbow - right elbow
            55,  # 20 left wrist - left wrist
            6,  # 21 right wrist - right wrist
            -1,  # 22 jaw - no match
            *([-1] * 2),
            56,  # 25 left_index1
            32,  # 26 left_index2
            33,  # 27 left_index3
            57,  # 28 left_middle1,
            34,  # 29 left_middle2,
            35,  # 30 left_middle3,
            59,  # 31 left_pinky1,
            30,  # 32 left_pinky2,
            31,  # 33 left_pinky3,
            58,  # 34 left_ring1,
            28,  # 35 left_ring2,
            29,  # 36 left_ring3,
            -1,  # 37 left_thumb1,
            26,  # 38 left_thumb2,
            27,  # 39 left_thumb3,
            80,  # 40 right_index1,
            93,  # 41 right_index2,
            94,  # 42 right_index3,
            81,  # 43 right_middle1,
            95,  # 44 right_middle2,
            96,  # 45 right_middle3,
            83,  # 46 right_pinky1,
            91,  # 47 right_pinky2,
            92,  # 48 right_pinky3,
            82,  # 49 right_ring1,
            89,  # 50 right_ring2,
            90,  # 51 right_ring3,
            -1,  # 52 right_thumb1,
            87,  # 53 right_thumb2,
            88,  # 54 right_thumb3,
            99,  # 55 nose - interpolated nose
            60,  # 56 right eye - right eye
            54,  # 57 left eye - left eye
            -1,  # 58
            -1,  # 59
            -1,  # 60
            -1,  # 61
            -1,  # 62
            -1,  # 63
            -1,  # 64
            -1,  # 65
            -1,  # 66
            -1,  # 67
            -1,  # 68
            -1,  # 69
            -1,  # 70
            -1,  # 71
            -1,  # 72
            -1,  # 73
            -1,  # 74
            -1,  # 75
            *([-1] * 51)
        ])
    else:
        raise ValueError(f'{dataset} is not supported')


def valid_mask(dataset):
    joint_mapping = unify_joint_mappings(dataset)
    mask = np.ones_like(joint_mapping)
    mask[joint_mapping == -1] = 0.
    return mask


def valid_mask_smpl(dataset):
    joint_mapping = unify_joint_mappings_smpl(dataset)
    mask = np.ones_like(joint_mapping)
    mask[joint_mapping == -1] = 0.
    return mask


def get_joint_names(dataset='smplx'):
    if dataset == 'smplx':
        return [
            'pelvis',
            'left_hip',
            'right_hip',
            'spine1',
            'left_knee',
            'right_knee',
            'spine2',
            'left_ankle',
            'right_ankle',
            'spine3',
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
            'left_eye_smplhf',
            'right_eye_smplhf',
            'left_index1',
            'left_index2',
            'left_index3',
            'left_middle1',
            'left_middle2',
            'left_middle3',
            'left_pinky1',
            'left_pinky2',
            'left_pinky3',
            'left_ring1',
            'left_ring2',
            'left_ring3',
            'left_thumb1',
            'left_thumb2',
            'left_thumb3',
            'right_index1',
            'right_index2',
            'right_index3',
            'right_middle1',
            'right_middle2',
            'right_middle3',
            'right_pinky1',
            'right_pinky2',
            'right_pinky3',
            'right_ring1',
            'right_ring2',
            'right_ring3',
            'right_thumb1',
            'right_thumb2',
            'right_thumb3',
            'nose',
            'right_eye',
            'left_eye',
            'right_ear',
            'left_ear',
            'left_big_toe',
            'left_small_toe',
            'left_heel',
            'right_big_toe',
            'right_small_toe',
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
        ]

    if dataset == 'smpl':
        return [
            'hips',  # 0
            'leftUpLeg',  # 1
            'rightUpLeg',  # 2
            'spine',  # 3
            'leftLeg',  # 4
            'rightLeg',  # 5
            'spine1',  # 6
            'leftFoot',  # 7
            'rightFoot',  # 8
            'spine2',  # 9
            'leftToeBase',  # 10
            'rightToeBase',  # 11
            'neck',  # 12
            'leftShoulder',  # 13
            'rightShoulder',  # 14
            'head',  # 15
            'leftArm',  # 16
            'rightArm',  # 17
            'leftForeArm',  # 18
            'rightForeArm',  # 19
            'leftHand',  # 20
            'rightHand',  # 21
            'leftHandIndex1',  # 22
            'rightHandIndex1',  # 23
        ]
    elif dataset == 'gta':
        return [
            'head_top',  # 00, extrapolate neck-head_center
            'head_center',  # 01
            'neck',  # 02
            'right_clavicle',  # 03
            'right_shoulder',  # 04
            'right_elbow',  # 05
            'right_wrist',  # 06
            'left_clavicle',  # 07
            'left_shoulder',  # 08
            'left_elbow',  # 09
            'left_wrist',  # 10
            'spine0',  # 11
            'spine1',  # 12
            'spine2',  # 13
            'spine3',  # 14
            'spine4',  # 15
            'right_hip',  # 16
            'right_knee',  # 17
            'right_ankle',  # 18
            'left_hip',  # 19
            'left_knee',  # 20
            'left_ankle',  # 21
            'SKEL_ROOT',  # 22
            'FB_R_Brow_Out_000',  # 23
            'SKEL_L_Toe0',  # 24
            'MH_R_Elbow',  # 25
            'SKEL_L_Finger01',  # 26
            'SKEL_L_Finger02',  # 27
            'SKEL_L_Finger31',  # 28
            'SKEL_L_Finger32',  # 29
            'SKEL_L_Finger41',  # 30
            'SKEL_L_Finger42',  # 31
            'SKEL_L_Finger11',  # 32
            'SKEL_L_Finger12',  # 33
            'SKEL_L_Finger21',  # 34
            'SKEL_L_Finger22',  # 35
            'RB_L_ArmRoll',  # 36
            'IK_R_Hand',  # 37
            'RB_R_ThighRoll',  # 38
            'FB_R_Lip_Corner_000',  # 39
            'SKEL_Pelvis',  # 40
            'IK_Head',  # 41
            'MH_R_Knee',  # 42
            'FB_LowerLipRoot_000',  # 43
            'FB_R_Lip_Top_000',  # 44
            'FB_R_CheekBone_000',  # 45
            'FB_UpperLipRoot_000',  # 46
            'FB_L_Lip_Top_000',  # 47
            'FB_LowerLip_000',  # 48
            'SKEL_R_Toe0',  # 49
            'FB_L_CheekBone_000',  # 50
            'MH_L_Elbow',  # 51
            'RB_L_ThighRoll',  # 52
            'PH_R_Foot',  # 53
            'FB_L_Eye_000',  # 54
            'SKEL_L_Finger00',  # 55
            'SKEL_L_Finger10',  # 56
            'SKEL_L_Finger20',  # 57
            'SKEL_L_Finger30',  # 58
            'SKEL_L_Finger40',  # 59
            'FB_R_Eye_000',  # 60
            'PH_R_Hand',  # 61
            'FB_L_Lip_Corner_000',  # 62
            'IK_R_Foot',  # 63
            'RB_Neck_1',  # 64
            'IK_L_Hand',  # 65
            'RB_R_ArmRoll',  # 66
            'FB_Brow_Centre_000',  # 67
            'FB_R_Lid_Upper_000',  # 68
            'RB_R_ForeArmRoll',  # 69
            'FB_L_Lid_Upper_000',  # 70
            'MH_L_Knee',  # 71
            'FB_Jaw_000',  # 72
            'FB_L_Lip_Bot_000',  # 73
            'FB_Tongue_000',  # 74
            'FB_R_Lip_Bot_000',  # 75
            'IK_Root',  # 76
            'PH_L_Foot',  # 77
            'FB_L_Brow_Out_000',  # 78
            'SKEL_R_Finger00',  # 79
            'SKEL_R_Finger10',  # 80
            'SKEL_R_Finger20',  # 81
            'SKEL_R_Finger30',  # 82
            'SKEL_R_Finger40',  # 83
            'PH_L_Hand',  # 84
            'RB_L_ForeArmRoll',  # 85
            'FB_UpperLip_000',  # 86
            'SKEL_R_Finger01',  # 87
            'SKEL_R_Finger02',  # 88
            'SKEL_R_Finger31',  # 89
            'SKEL_R_Finger32',  # 90
            'SKEL_R_Finger41',  # 91
            'SKEL_R_Finger42',  # 92
            'SKEL_R_Finger11',  # 93
            'SKEL_R_Finger12',  # 94
            'SKEL_R_Finger21',  # 95
            'SKEL_R_Finger22',  # 96
            'FACIAL_facialRoot',  # 97
            'IK_L_Foot',  # 98
            'interpolated_nose'  # 99, interpolate
            # FB_R_CheekBone_000-FB_L_CheekBone_000
        ]
    else:
        raise ValueError(f'{dataset} is not supported')


def gmof(x, sigma):
    """Geman-McClure error function."""
    x_squared = x**2
    sigma_squared = sigma**2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def build_optimizer(opt_params, opt_config):
    optimizer_type = opt_config.pop('type')

    if optimizer_type == 'lbfgs':
        optimizer = torch.optim.LBFGS(opt_params, **opt_config)
    else:
        raise NotImplementedError

    opt_config['type'] = optimizer_type
    return optimizer


def angle_prior(pose, spine=False):
    """Angle prior that penalizes unnatural bending of the knees and elbows."""
    # We subtract 3 because pose does not include the global rotation of
    # the model
    angle_loss = torch.exp(
        pose[:, [55 - 3, 58 - 3, 12 - 3, 15 - 3]] *
        torch.tensor([1., -1., -1, -1.], device=pose.device))**2
    if spine:  # limit rotation of 3 spines
        spine_poses = pose[:, [
            9 - 3, 10 - 3, 11 - 3, 18 - 3, 19 - 3, 20 - 3, 27 - 3, 28 - 3, 29 -
            3
        ]]
        spine_loss = torch.exp(torch.abs(spine_poses))**2
        angle_loss = torch.cat([angle_loss, spine_loss], axis=1)
    return angle_loss


class SMPLify(object):
    """Re-implementation of SMPLify with extended features."""

    def __init__(self,
                 body_model=None,
                 keypoints_2d_weight=1.0,
                 keypoints_3d_weight=1.0,
                 use_one_betas_per_video=False,
                 num_epochs=20,
                 camera=default_camera,
                 stage_config=smplify_stages,
                 opt_config=smplify_opt_config,
                 device=torch.device('cuda'),
                 verbose=False):

        self.keypoints_2d_weight = keypoints_2d_weight
        self.keypoints_3d_weight = keypoints_3d_weight
        self.use_one_betas_per_video = use_one_betas_per_video
        self.num_epochs = num_epochs
        self.stage_config = stage_config
        self.opt_config = opt_config
        self.camera = camera
        self.device = device
        self.body_model = body_model.to(self.device)

    def __call__(self,
                 keypoints_2d=None,
                 keypoints_conf_2d=None,
                 keypoints_3d=None,
                 keypoints_conf_3d=None,
                 init_global_orient=None,
                 init_transl=None,
                 init_body_pose=None,
                 init_betas=None,
                 batch_size=None,
                 num_videos=None):

        assert keypoints_2d is not None or keypoints_3d is not None, \
            'Neither of 2D nor 3D keypoints ground truth is provided.'
        if batch_size is None:
            batch_size = keypoints_2d.shape[
                0] if keypoints_2d is not None else keypoints_3d.shape[0]
        if num_videos is None:
            num_videos = batch_size
        assert batch_size % num_videos == 0

        global_orient = init_global_orient if init_global_orient is not None \
            else self.body_model.global_orient
        transl = init_transl if init_transl is not None \
            else self.body_model.transl
        body_pose = init_body_pose if init_body_pose is not None \
            else self.body_model.body_pose

        if init_betas is not None:
            betas = init_betas
        elif self.use_one_betas_per_video:
            betas = torch.zeros(
                num_videos, self.body_model.betas.shape[-1]).to(self.device)
        else:
            betas = self.body_model.betas

        for i in range(self.num_epochs):
            for stage_name, stage_config in self.stage_config.items():
                self._optimize_stage(
                    global_orient=global_orient,
                    transl=transl,
                    body_pose=body_pose,
                    betas=betas,
                    keypoints_2d=keypoints_2d,
                    keypoints_conf_2d=keypoints_conf_2d,
                    keypoints_3d=keypoints_3d,
                    keypoints_conf_3d=keypoints_conf_3d,
                    **stage_config,
                )

        return {
            'global_orient': global_orient,
            'transl': transl,
            'body_pose': body_pose,
            'betas': betas
        }

    def _set_param(self, fit_param, param, opt_param):
        if fit_param:
            param.require_grads = True
            opt_param.append(param)
        else:
            param.require_grads = False

    def _expand_betas(self, pose, betas):
        batch_size = pose.shape[0]
        num_video = betas.shape[0]
        if batch_size == num_video:
            return betas

        video_size = batch_size // num_video
        betas_ext = torch.zeros(
            batch_size, betas.shape[-1], device=betas.device)
        for i in range(num_video):
            betas_ext[i * video_size:(i + 1) * video_size] = betas[i]

        return betas_ext

    def _optimize_stage(self,
                        global_orient,
                        transl,
                        body_pose,
                        betas,
                        fit_global_orient=True,
                        fit_transl=True,
                        fit_body_pose=True,
                        fit_betas=True,
                        keypoints_2d=None,
                        keypoints_conf_2d=None,
                        keypoints_2d_weight=1.0,
                        keypoints_3d=None,
                        keypoints_conf_3d=None,
                        keypoints_3d_weight=1.0,
                        shape_prior_weight=1.0,
                        angle_prior_weight=1.0,
                        num_iter=1):

        opt_param = []
        self._set_param(fit_global_orient, global_orient, opt_param)
        self._set_param(fit_transl, transl, opt_param)
        self._set_param(fit_body_pose, body_pose, opt_param)
        self._set_param(fit_betas, betas, opt_param)

        optimizer = build_optimizer(opt_param, self.opt_config)

        for iter_idx in range(num_iter):

            def closure():
                # body_pose_fixed = use_reference_spine(body_pose,
                # init_body_pose)

                optimizer.zero_grad()
                betas_ext = self._arrange_betas(body_pose, betas)

                smpl_output = self.body_model(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    betas=betas_ext,
                    transl=transl)

                model_joints = smpl_output.joints

                mapping_target = unify_joint_mappings_smpl(dataset='smpl')
                model_joints = model_joints[:, mapping_target, :]

                loss_dict = self._compute_loss(
                    model_joints,
                    keypoints_2d=keypoints_2d,
                    keypoints_conf_2d=keypoints_conf_2d,
                    keypoints_2d_weight=keypoints_2d_weight,
                    keypoints_3d=keypoints_3d,
                    keypoints_conf_3d=keypoints_conf_3d,
                    keypoints_3d_weight=keypoints_3d_weight,
                    shape_prior_weight=shape_prior_weight,
                    angle_prior_weight=angle_prior_weight,
                    body_pose=body_pose,
                    betas=betas_ext)

                loss = loss_dict['total_loss']
                loss.backward()
                return loss

            optimizer.step(closure)

    def _compute_loss(self,
                      model_joints,
                      keypoints_2d=None,
                      keypoints_conf_2d=None,
                      keypoints_2d_weight=1.0,
                      keypoints_3d=None,
                      keypoints_conf_3d=None,
                      keypoints_3d_weight=1.0,
                      shape_prior_weight=1.0,
                      angle_prior_weight=1.0,
                      body_pose=None,
                      betas=None):

        total_loss = 0

        # 2D keypoint loss
        if keypoints_2d is not None:
            projected_joints = self.camera(model_joints)
            reprojection_error = gmof(
                projected_joints - keypoints_2d, sigma=100)
            joints_weights = torch.ones_like(
                keypoints_conf_2d) * keypoints_2d_weight
            reprojection_weight = (joints_weights * keypoints_conf_2d)**2
            reprojection_loss = reprojection_weight * reprojection_error.sum(
                dim=-1)
            total_loss = total_loss + reprojection_loss.sum(
                dim=-1) * keypoints_2d_weight

        # 3D keypoint loss
        # TODO: proper sigma for keypoints3d
        if keypoints_3d is not None:
            joint_diff_3d = gmof(model_joints - keypoints_3d, sigma=100)
            joints_weights = torch.ones_like(
                keypoints_conf_3d) * keypoints_3d_weight
            joint_loss_3d_weight = (joints_weights * keypoints_conf_3d)**2
            joint_loss_3d = joint_loss_3d_weight * joint_diff_3d.sum(dim=-1)
            total_loss = total_loss + joint_loss_3d.sum(
                dim=-1) * keypoints_3d_weight

        # Regularizer to prevent betas from taking large values
        if betas is not None:
            shape_prior_loss = (shape_prior_weight**2) * \
                               (betas**2).sum(dim=-1)
            total_loss = total_loss + shape_prior_loss

        # Angle prior for knees and elbows
        # TODO: temp disable angle_prior_loss
        # angle_prior_loss = (angle_prior_weight ** 2) * \
        # angle_prior(body_pose, spine=True).sum(dim=-1)
        # total_loss = total_loss + angle_prior_loss

        # Smooth body
        # TODO: temp disable body_pose_loss
        # theta = body_pose.reshape(body_pose.shape[0], -1, 3)
        # rot_6d = matrix_to_rotation_6d(axis_angle_to_matrix(theta))
        # rot_6d_diff = rot_6d[1:] - rot_6d[:-1]
        # smooth_body_loss = rot_6d_diff.abs().sum(dim=-1)
        # smooth_body_loss = torch.cat(
        #     [torch.zeros(1, smooth_body_loss.shape[1],
        #                  device=body_pose.device,
        #                  dtype=smooth_body_loss.dtype),
        #      smooth_body_loss]
        # ).mean(dim=-1)

        return {
            'total_loss': total_loss.sum(),
        }

    def _arrange_betas(self, pose, betas):
        batch_size = pose.shape[0]
        num_video = betas.shape[0]

        video_size = batch_size // num_video
        betas_ext = torch.zeros(
            batch_size, betas.shape[-1], device=betas.device)
        for i in range(num_video):
            betas_ext[i * video_size:(i + 1) * video_size] = betas[i]

        return betas_ext


class SMPLifyX(SMPLify):
    """Re-implementation of SMPLify-X with extended features."""

    def __init__(self,
                 body_model=None,
                 keypoints_2d_weight=1.0,
                 keypoints_3d_weight=1.0,
                 use_one_betas_per_video=False,
                 num_epochs=20,
                 camera=default_camera,
                 stage_config=smplifyx_stages,
                 opt_config=smplifyx_opt_config,
                 device=torch.device('cuda'),
                 verbose=False):
        super(SMPLifyX, self).__init__(
            body_model=body_model,
            keypoints_2d_weight=keypoints_2d_weight,
            keypoints_3d_weight=keypoints_3d_weight,
            use_one_betas_per_video=use_one_betas_per_video,
            num_epochs=num_epochs,
            camera=camera,
            stage_config=stage_config,
            opt_config=opt_config,
            device=device,
            verbose=verbose)

    def __call__(self,
                 keypoints_2d=None,
                 keypoints_conf_2d=1.0,
                 keypoints_3d=None,
                 keypoints_conf_3d=1.0,
                 init_global_orient=None,
                 init_transl=None,
                 init_body_pose=None,
                 init_betas=None,
                 init_left_hand_pose=None,
                 init_right_hand_pose=None,
                 init_expression=None,
                 init_jaw_pose=None,
                 init_leye_pose=None,
                 init_reye_pose=None,
                 batch_size=None,
                 num_videos=None):

        assert keypoints_2d is not None or keypoints_3d is not None, \
            'Neither of 2D nor 3D keypoints ground truth is provided.'
        if batch_size is None:
            batch_size = keypoints_2d.shape[
                0] if keypoints_2d is not None else keypoints_3d.shape[0]
        if num_videos is None:
            num_videos = batch_size
        assert batch_size % num_videos == 0

        global_orient = init_global_orient if init_global_orient is not None \
            else self.body_model.global_orient
        transl = init_transl if init_transl is not None \
            else self.body_model.transl
        body_pose = init_body_pose if init_body_pose is not None \
            else self.body_model.body_pose

        left_hand_pose = init_left_hand_pose \
            if init_left_hand_pose is not None \
            else self.body_model.left_hand_pose
        right_hand_pose = init_right_hand_pose \
            if init_right_hand_pose is not None \
            else self.body_model.right_hand_pose
        expression = init_expression \
            if init_expression is not None \
            else self.body_model.expression
        jaw_pose = init_jaw_pose \
            if init_jaw_pose is not None \
            else self.body_model.jaw_pose
        leye_pose = init_leye_pose \
            if init_leye_pose is not None \
            else self.body_model.leye_pose
        reye_pose = init_reye_pose \
            if init_reye_pose is not None \
            else self.body_model.reye_pose

        if init_betas is not None:
            betas = init_betas
        elif self.use_one_betas_per_video:
            betas = torch.zeros(
                num_videos, self.body_model.betas.shape[-1]).to(self.device)
        else:
            betas = self.body_model.betas

        for i in range(self.num_epochs):
            for stage_name, stage_config in self.stage_config.items():
                self._optimize_stage(
                    global_orient=global_orient,
                    transl=transl,
                    body_pose=body_pose,
                    betas=betas,
                    left_hand_pose=left_hand_pose,
                    right_hand_pose=right_hand_pose,
                    expression=expression,
                    jaw_pose=jaw_pose,
                    leye_pose=leye_pose,
                    reye_pose=reye_pose,
                    keypoints_2d=keypoints_2d,
                    keypoints_conf_2d=keypoints_conf_2d,
                    keypoints_3d=keypoints_3d,
                    keypoints_conf_3d=keypoints_conf_3d,
                    **stage_config,
                )

        return {
            'global_orient': global_orient,
            'transl': transl,
            'body_pose': body_pose,
            'betas': betas,
            'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose,
            'expression': expression,
            'jaw_pose': jaw_pose,
            'leye_pose': leye_pose,
            'reye_pose': reye_pose
        }

    def _optimize_stage(self,
                        global_orient,
                        transl,
                        body_pose,
                        betas,
                        left_hand_pose,
                        right_hand_pose,
                        expression,
                        jaw_pose,
                        leye_pose,
                        reye_pose,
                        fit_global_orient=True,
                        fit_transl=True,
                        fit_body_pose=True,
                        fit_betas=True,
                        fit_left_hand_pose=True,
                        fit_right_hand_pose=True,
                        fit_expression=True,
                        fit_jaw_pose=True,
                        fit_leye_pose=True,
                        fit_reye_pose=True,
                        keypoints_2d=None,
                        keypoints_conf_2d=None,
                        keypoints_2d_weight=1.0,
                        keypoints_3d=None,
                        keypoints_conf_3d=None,
                        keypoints_3d_weight=1.0,
                        shape_prior_weight=1.0,
                        angle_prior_weight=1.0,
                        num_iter=1):

        opt_param = []
        self._set_param(fit_global_orient, global_orient, opt_param)
        self._set_param(fit_transl, transl, opt_param)
        self._set_param(fit_body_pose, body_pose, opt_param)
        self._set_param(fit_betas, betas, opt_param)
        self._set_param(fit_left_hand_pose, left_hand_pose, opt_param)
        self._set_param(fit_right_hand_pose, right_hand_pose, opt_param)
        self._set_param(fit_expression, expression, opt_param)
        self._set_param(fit_jaw_pose, jaw_pose, opt_param)
        self._set_param(fit_leye_pose, leye_pose, opt_param)
        self._set_param(fit_reye_pose, reye_pose, opt_param)

        optimizer = build_optimizer(opt_param, self.opt_config)

        for iter_idx in range(num_iter):

            def closure():
                # body_pose_fixed = use_reference_spine(body_pose,
                # init_body_pose)

                optimizer.zero_grad()
                betas_ext = self._arrange_betas(body_pose, betas)

                output = self.body_model(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    betas=betas_ext,
                    transl=transl,
                    left_hand_pose=left_hand_pose,
                    right_hand_pose=right_hand_pose,
                    expression=expression,
                    jaw_pose=jaw_pose,
                    leye_pose=leye_pose,
                    reye_pose=reye_pose)

                model_joints = output.joints

                mapping_target = unify_joint_mappings(dataset='smplx')
                model_joints = model_joints[:, mapping_target, :]

                loss_dict = self._compute_loss(
                    model_joints,
                    keypoints_2d=keypoints_2d,
                    keypoints_conf_2d=keypoints_conf_2d,
                    keypoints_2d_weight=keypoints_2d_weight,
                    keypoints_3d=keypoints_3d,
                    keypoints_conf_3d=keypoints_conf_3d,
                    keypoints_3d_weight=keypoints_3d_weight,
                    shape_prior_weight=shape_prior_weight,
                    angle_prior_weight=angle_prior_weight,
                    body_pose=body_pose,
                    betas=betas_ext)

                loss = loss_dict['total_loss']
                loss.backward()
                return loss

            optimizer.step(closure)
