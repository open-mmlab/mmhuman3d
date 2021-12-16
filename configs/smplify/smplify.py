type = 'SMPLify'

body_model = dict(
    type='SMPL',
    gender='neutral',
    num_betas=10,
    keypoint_src='smpl_45',
    keypoint_dst='smpl_45',
    model_path='data/body_models/smpl',
    batch_size=1)

stages = [
    # stage 1
    dict(
        num_iter=20,
        fit_global_orient=True,
        fit_transl=True,
        fit_body_pose=False,
        fit_betas=False,
        joint_weights=dict(
            body_weight=5.0,
            use_shoulder_hip_only=True,
        )),
    # stage 2
    dict(
        num_iter=10,
        fit_global_orient=True,
        fit_transl=True,
        fit_body_pose=True,
        fit_betas=True,
        joint_weights=dict(body_weight=5.0, use_shoulder_hip_only=False))
]

optimizer = dict(
    type='LBFGS', max_iter=20, lr=1e-2, line_search_fn='strong_wolfe')

keypoints2d_loss = dict(
    type='KeypointMSELoss', loss_weight=1.0, reduction='sum', sigma=100)

keypoints3d_loss = dict(
    type='KeypointMSELoss', loss_weight=10, reduction='sum', sigma=100)

shape_prior_loss = dict(type='ShapePriorLoss', loss_weight=1, reduction='sum')

joint_prior_loss = dict(
    type='JointPriorLoss',
    loss_weight=20,
    reduction='sum',
    smooth_spine=True,
    smooth_spine_loss_weight=20,
    use_full_body=True)

smooth_loss = dict(type='SmoothJointLoss', loss_weight=0, reduction='sum')

pose_prior_loss = dict(
    type='MaxMixturePrior',
    prior_folder='data',
    num_gaussians=8,
    loss_weight=4.78**2,
    reduction='sum')

ignore_keypoints = [
    'neck_openpose', 'right_hip_openpose', 'left_hip_openpose',
    'right_hip_extra', 'left_hip_extra'
]

camera = dict(
    type='PerspectiveCameras',
    convention='opencv',
    in_ndc=False,
    focal_length=5000,
    image_size=(224, 224),
    principal_point=(112, 112))
