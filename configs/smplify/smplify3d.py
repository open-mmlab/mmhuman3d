type = 'SMPLify'
verbose = True

body_model = dict(
    type='SMPL',
    gender='neutral',
    num_betas=10,
    keypoint_src='smpl_45',
    # keypoint_dst='smpl_45',
    keypoint_dst='smpl',
    model_path='data/body_models/smpl',
    batch_size=1)

stages = [
    # stage 00
    dict(
        num_iter=10,
        fit_global_orient=False,
        fit_transl=False,
        fit_body_pose=False,
        fit_betas=True,
        keypoints3d_weight=0.0,
        pose_reg_weight=0.0,
        limb_length_weight=1.0,
        shape_prior_weight=1.0,
    ),
    # stage 0
    # dict(
    #     warmup=True,
    #     num_iter=10,
    #     fit_global_orient=True,
    #     fit_transl=True,
    #     fit_body_pose=False,
    #     fit_betas=False,
    #     joint_weights=dict(
    #         body_weight=5.0,
    #         use_shoulder_hip_only=True,
    #     )),
    # stage 1
    dict(
        # num_iter=20,
        # num_iter=10,
        num_iter=50,
        fit_global_orient=True,
        fit_transl=True,
        fit_body_pose=False,
        fit_betas=False,
        keypoints3d_weight=1.0,
        pose_reg_weight=0.001,
        limb_length_weight=0.0,
        shape_prior_weight=0.0,
        joint_weights=dict(
            body_weight=5.0,
            use_shoulder_hip_only=True,
        )),
    # stage 2
    dict(
        # num_iter=10,
        # num_iter=30,
        num_iter=120,
        fit_global_orient=True,
        fit_transl=True,
        fit_body_pose=True,
        fit_betas=False,
        keypoints3d_weight=1.0,
        pose_reg_weight=0.001,
        limb_length_weight=0.0,
        shape_prior_weight=0.0,
        joint_weights=dict(body_weight=5.0, use_shoulder_hip_only=False))
]

# optimizer = dict(
#     type='LBFGS', max_iter=20, lr=1e-2, line_search_fn='strong_wolfe')

# optimizer = dict(
#     type='LBFGS', max_iter=10, lr=1.0, line_search_fn='strong_wolfe')

optimizer = dict(
    type='LBFGS', max_iter=20, lr=1.0, line_search_fn='strong_wolfe')

# keypoints2d_loss = dict(
#     type='KeypointMSELoss', loss_weight=1.0, reduction='sum', sigma=100)

keypoints3d_loss = dict(
    type='KeypointMSELoss', loss_weight=10, reduction='sum', sigma=100)

# keypoints3d_loss = dict(
#     type='MSELoss', loss_weight=1, reduction='sum')

shape_prior_loss = dict(type='ShapePriorLoss', loss_weight=1, reduction='sum')

limb_length_loss = dict(type='LimbLengthLoss', loss_weight=1, reduction='sum')

pose_reg_loss = dict(type='PoseRegLoss', loss_weight=0.001, reduction='sum')

# joint_prior_loss = dict(
#     type='JointPriorLoss',
#     loss_weight=20,
#     reduction='sum',
#     smooth_spine=True,
#     smooth_spine_loss_weight=20,
#     use_full_body=False)
#     # use_full_body=True)

# TODO
# smooth_loss = dict(type='SmoothJointLoss', loss_weight=1.0, reduction='sum')
# smooth_loss = dict(type='SmoothJointLoss', loss_weight=0., reduction='sum')

# pose_prior_loss = dict(
#     type='MaxMixturePrior',
#     prior_folder='data',
#     num_gaussians=8,
#     loss_weight=4.78**2,
#     reduction='sum')

# TODO
ignore_keypoints = [
    'neck_openpose', 'right_hip_openpose', 'left_hip_openpose',
    'right_hip_extra', 'left_hip_extra'
]

# camera = dict(
#     type='PerspectiveCameras',
#     convention='opencv',
#     in_ndc=False,
#     focal_length=5000,
#     image_size=(224, 224),
#     principal_point=(112, 112))
