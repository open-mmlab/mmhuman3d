# ORIGINAL_NAMES = [
#     'head_top',             # 00, extrapolate 02-01
#     'head_center',          # 01
#     'neck',                 # 02
#     'right_clavicle',       # 03
#     'right_shoulder',       # 04
#     'right_elbow',          # 05
#     'right_wrist',          # 06
#     'left_clavicle',        # 07
#     'left_shoulder',        # 08
#     'left_elbow',           # 09
#     'left_wrist',           # 10
#     'spine0',               # 11
#     'spine1',               # 12
#     'spine2',               # 13
#     'spine3',               # 14
#     'spine4',               # 15
#     'right_hip',            # 16
#     'right_knee',           # 17
#     'right_ankle',          # 18
#     'left_hip',             # 19
#     'left_knee',            # 20
#     'left_ankle',           # 21
#     'SKEL_ROOT',            # 22
#     'FB_R_Brow_Out_000',    # 23
#     'SKEL_L_Toe0',          # 24
#     'MH_R_Elbow',           # 25
#     'SKEL_L_Finger01',      # 26
#     'SKEL_L_Finger02',      # 27
#     'SKEL_L_Finger31',      # 28
#     'SKEL_L_Finger32',      # 29
#     'SKEL_L_Finger41',      # 30
#     'SKEL_L_Finger42',      # 31
#     'SKEL_L_Finger11',      # 32
#     'SKEL_L_Finger12',      # 33
#     'SKEL_L_Finger21',      # 34
#     'SKEL_L_Finger22',      # 35
#     'RB_L_ArmRoll',         # 36
#     'IK_R_Hand',            # 37
#     'RB_R_ThighRoll',       # 38
#     'FB_R_Lip_Corner_000',  # 39
#     'SKEL_Pelvis',          # 40
#     'IK_Head',              # 41
#     'MH_R_Knee',            # 42
#     'FB_LowerLipRoot_000',  # 43
#     'FB_R_Lip_Top_000',     # 44
#     'FB_R_CheekBone_000',   # 45
#     'FB_UpperLipRoot_000',  # 46
#     'FB_L_Lip_Top_000',     # 47
#     'FB_LowerLip_000',      # 48
#     'SKEL_R_Toe0',          # 49
#     'FB_L_CheekBone_000',   # 50
#     'MH_L_Elbow',           # 51
#     'RB_L_ThighRoll',       # 52
#     'PH_R_Foot',            # 53
#     'FB_L_Eye_000',         # 54
#     'SKEL_L_Finger00',      # 55
#     'SKEL_L_Finger10',      # 56
#     'SKEL_L_Finger20',      # 57
#     'SKEL_L_Finger30',      # 58
#     'SKEL_L_Finger40',      # 59
#     'FB_R_Eye_000',         # 60
#     'PH_R_Hand',            # 61
#     'FB_L_Lip_Corner_000',  # 62
#     'IK_R_Foot',            # 63
#     'RB_Neck_1',            # 64
#     'IK_L_Hand',            # 65
#     'RB_R_ArmRoll',         # 66
#     'FB_Brow_Centre_000',   # 67
#     'FB_R_Lid_Upper_000',   # 68
#     'RB_R_ForeArmRoll',     # 69
#     'FB_L_Lid_Upper_000',   # 70
#     'MH_L_Knee',            # 71
#     'FB_Jaw_000',           # 72
#     'FB_L_Lip_Bot_000',     # 73
#     'FB_Tongue_000',        # 74
#     'FB_R_Lip_Bot_000',     # 75
#     'IK_Root',              # 76
#     'PH_L_Foot',            # 77
#     'FB_L_Brow_Out_000',    # 78
#     'SKEL_R_Finger00',      # 79
#     'SKEL_R_Finger10',      # 80
#     'SKEL_R_Finger20',      # 81
#     'SKEL_R_Finger30',      # 82
#     'SKEL_R_Finger40',      # 83
#     'PH_L_Hand',            # 84
#     'RB_L_ForeArmRoll',     # 85
#     'FB_UpperLip_000',      # 86
#     'SKEL_R_Finger01',      # 87
#     'SKEL_R_Finger02',      # 88
#     'SKEL_R_Finger31',      # 89
#     'SKEL_R_Finger32',      # 90
#     'SKEL_R_Finger41',      # 91
#     'SKEL_R_Finger42',      # 92
#     'SKEL_R_Finger11',      # 93
#     'SKEL_R_Finger12',      # 94
#     'SKEL_R_Finger21',      # 95
#     'SKEL_R_Finger22',      # 96
#     'FACIAL_facialRoot',    # 97
#     'IK_L_Foot',            # 98
#     'interpolated_nose'     # 99, mid-point of 45-50
# ]

GTA_KEYPOINTS = [
    'gta_head_top',  # 00
    'head',  # 01 - head_center
    'neck',  # 02 - neck
    'gta_right_clavicle',  # 03
    'right_shoulder',  # 04  - right_shoulder
    'right_elbow',  # 05  - right_elbow
    'right_wrist',  # 06  - right_wrist
    'gta_left_clavicle',  # 07
    'left_shoulder',  # 08  - left_shoulder
    'left_elbow',  # 09  - left_elbow
    'left_wrist',  # 10  - left_wrist
    'spine_2',  # 11  - spine0
    'gta_spine1',  # 12
    'spine_1',  # 13  - spine2
    'pelvis',  # 14  - pelvis
    'gta_spine4',  # 15
    'right_hip',  # 16  - right_hip
    'right_knee',  # 17  - right_knee
    'right_ankle',  # 18  - right_ankle
    'left_hip',  # 19  - left_hip
    'left_knee',  # 20  - left_knee
    'left_ankle',  # 21  - left_ankle
    'gta_SKEL_ROOT',  # 22
    'gta_FB_R_Brow_Out_000',  # 23
    'left_foot',  # 24  - SKEL_L_Toe0
    'gta_MH_R_Elbow',  # 25
    'left_thumb_2',  # 26  - SKEL_L_Finger01
    'left_thumb_3',  # 27  - SKEL_L_Finger02
    'left_ring_2',  # 28  - SKEL_L_Finger31
    'left_ring_3',  # 29  - SKEL_L_Finger32
    'left_pinky_2',  # 30  - SKEL_L_Finger41
    'left_pinky_3',  # 31  - SKEL_L_Finger42
    'left_index_2',  # 32  - SKEL_L_Finger11
    'left_index_3',  # 33  - SKEL_L_Finger12
    'left_middle_2',  # 34  - SKEL_L_Finger21
    'left_middle_3',  # 35  - SKEL_L_Finger22
    'gta_RB_L_ArmRoll',  # 36
    'gta_IK_R_Hand',  # 37
    'gta_RB_R_ThighRoll',  # 38
    'gta_FB_R_Lip_Corner_000',  # 39
    'gta_SKEL_Pelvis',  # 40
    'gta_IK_Head',  # 41
    'gta_MH_R_Knee',  # 42
    'gta_FB_LowerLipRoot_000',  # 43
    'gta_FB_R_Lip_Top_000',  # 44
    'gta_FB_R_CheekBone_000',  # 45
    'gta_FB_UpperLipRoot_000',  # 46
    'gta_FB_L_Lip_Top_000',  # 47
    'gta_FB_LowerLip_000',  # 48
    'right_foot',  # 49  - SKEL_R_Toe0
    'gta_FB_L_CheekBone_000',  # 50
    'gta_MH_L_Elbow',  # 51
    'gta_RB_L_ThighRoll',  # 52
    'gta_PH_R_Foot',  # 53
    'left_eye',  # 54  - FB_L_Eye_000
    'gta_SKEL_L_Finger00',  # 55
    'left_index_1',  # 56  - SKEL_L_Finger10
    'left_middle_1',  # 57  - SKEL_L_Finger20
    'left_ring_1',  # 58  - SKEL_L_Finger30
    'left_pinky_1',  # 59  - SKEL_L_Finger40
    'right_eye',  # 60  - FB_R_Eye_000
    'gta_PH_R_Hand',  # 61
    'gta_FB_L_Lip_Corner_000',  # 62
    'gta_IK_R_Foot',  # 63
    'gta_RB_Neck_1',  # 64
    'gta_IK_L_Hand',  # 65
    'gta_RB_R_ArmRoll',  # 66
    'gta_FB_Brow_Centre_000',  # 67
    'gta_FB_R_Lid_Upper_000',  # 68
    'gta_RB_R_ForeArmRoll',  # 69
    'gta_FB_L_Lid_Upper_000',  # 70
    'gta_MH_L_Knee',  # 71
    'gta_FB_Jaw_000',  # 72
    'gta_FB_L_Lip_Bot_000',  # 73
    'gta_FB_Tongue_000',  # 74
    'gta_FB_R_Lip_Bot_000',  # 75
    'gta_IK_Root',  # 76
    'gta_PH_L_Foot',  # 77
    'gta_FB_L_Brow_Out_000',  # 78
    'gta_SKEL_R_Finger00',  # 79
    'right_index_1',  # 80  - SKEL_R_Finger10
    'right_middle_1',  # 81  - SKEL_R_Finger20
    'right_ring_1',  # 82  - SKEL_R_Finger30
    'right_pinky_1',  # 83  - SKEL_R_Finger40
    'gta_PH_L_Hand',  # 84
    'gta_RB_L_ForeArmRoll',  # 85
    'gta_FB_UpperLip_000',  # 86
    'right_thumb_2',  # 87  - SKEL_R_Finger01
    'right_thumb_3',  # 88  - SKEL_R_Finger02
    'right_ring_2',  # 89  - SKEL_R_Finger31
    'right_ring_3',  # 90  - SKEL_R_Finger32
    'right_pinky_2',  # 91  - SKEL_R_Finger41
    'right_pinky_3',  # 92  - SKEL_R_Finger42
    'right_index_2',  # 93  - SKEL_R_Finger11
    'right_index_3',  # 94  - SKEL_R_Finger12
    'right_middle_2',  # 95  - SKEL_R_Finger21
    'right_middle_3',  # 96  - SKEL_R_Finger22
    'gta_FACIAL_facialRoot',  # 97
    'gta_IK_L_Foot',  # 98
    'nose'  # 99  - interpolated nose
]
