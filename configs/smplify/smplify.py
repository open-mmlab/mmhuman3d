smplify_stages = {
    'Stage 1': {
        'num_iter': 10,
        'fit_global_orient': True,
        'fit_transl': True,
        'fit_body_pose': False,
        'fit_betas': False,
        'joint_weights': {
            'body_weight': 5.0,
            'use_shoulder_hip_only': True,
        }
    },
    'Stage 2': {
        'num_iter': 10,
        'fit_global_orient': True,
        'fit_transl': True,
        'fit_body_pose': True,
        'fit_betas': True,
        'joint_weights': {
            'body_weight': 5.0,
            'use_shoulder_hip_only': False
        }
    }
}

smplify_opt_config = {
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

keypoints_2d_loss_config = {
    'type': 'KeypointMSELoss',
    'loss_weight': 0,
    'reduction': 'sum',
    'sigma': 100
}

keypoints_3d_loss_config = {
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
