smplify_stages = {
    'Stage 1': {
        'num_iter': 10,
    },
    'Stage 2': {
        'num_iter': 1,
    },
}

smplify_opt_config = {
    'type': 'LBFGS',
    'max_iter': 20,
    'lr': 1e-2,
    'line_search_fn': 'strong_wolfe'
}

keypoints_2d_loss_config = {
    'type': 'KeypointMSELoss',
    'loss_weight': 1.0,
    'reduction': 'sum',
    'sigma': 100
}

keypoints_3d_loss_config = {
    'type': 'KeypointMSELoss',
    'loss_weight': 1.0,
    'reduction': 'sum',
    'sigma': 100
}

shape_prior_loss_config = {'type': 'ShapePriorLoss', 'reduction': 'sum'}

joint_prior_loss_config = {'type': 'JointPriorLoss', 'reduction': 'sum'}
