smplifyx_stages = {
    'Stage 1': {
        'num_iter': 10,
        'fit_global_orient': True,
        'fit_transl': True,
        'fit_body_pose': False,
        'fit_betas': False,
        'fit_left_hand_pose': False,
        'fit_right_hand_pose': False,
        'fit_expression': False,
        'fit_jaw_pose': False,
        'fit_leye_pose': False,
        'fit_reye_pose': False,
        'joint_weights': {
            'body_weight': 5.0,
            'use_shoulder_hip_only': True,
            'hand_weight': 0.0,
            'face_weight': 0.0
        }
    },
    'Stage 2': {
        'num_iter': 5,
        'fit_global_orient': True,
        'fit_transl': True,
        'fit_body_pose': True,
        'fit_betas': True,
        'fit_left_hand_pose': False,
        'fit_right_hand_pose': False,
        'fit_expression': False,
        'fit_jaw_pose': False,
        'fit_leye_pose': False,
        'fit_reye_pose': False,
        'joint_weights': {
            'body_weight': 5.0,
            'use_shoulder_hip_only': False,
            'hand_weight': 0.0,
            'face_weight': 0.0
        }
    },
    'Stage 3': {
        'num_iter': 3,
        'fit_global_orient': True,
        'fit_transl': True,
        'fit_body_pose': True,
        'fit_betas': True,
        'fit_left_hand_pose': True,
        'fit_right_hand_pose': True,
        'fit_expression': False,
        'fit_jaw_pose': False,
        'fit_leye_pose': False,
        'fit_reye_pose': False,
        'joint_weights': {
            'body_weight': 10.0,
            'use_shoulder_hip_only': False,
            'hand_weight': 1.0,
            'face_weight': 1.0
        }
    }
}

smplifyx_opt_config = {
    'type': 'LBFGS',
    'max_iter': 20,
    'lr': 1e-2,
    'line_search_fn': 'strong_wolfe'
}

# smplifyx_opt_config = {
#     'type': 'Adam',
#     'lr': 1e-2,
#     'betas': (0.9, 0.999)
# }

keypoints2d_loss_config = {
    'type': 'KeypointMSELoss',
    'loss_weight': 0,
    'reduction': 'sum',
    'sigma': 100
}

keypoints3d_loss_config = {
    'type': 'KeypointMSELoss',
    'loss_weight': 10,
    'reduction': 'sum',
    'sigma': 100
}

shape_prior_loss_config = {
    'type': 'ShapePriorLoss',
    'loss_weight': 1,
    'reduction': 'sum'
}

joint_prior_loss_config = {
    'type': 'JointPriorLoss',
    'loss_weight': 20,
    'reduction': 'sum',
    'smooth_spine': True,
    'smooth_spine_loss_weight': 20,
    'use_full_body': True
}

smooth_loss_config = {
    'type': 'SmoothJointLoss',
    'loss_weight': 0,
    'reduction': 'sum'
}
