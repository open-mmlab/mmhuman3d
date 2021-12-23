type = 'SMPLifyX'

body_model = dict(
    type='SMPLX',
    gender='neutral',
    num_betas=10,
    use_face_contour=True,
    keypoint_src='smplx',
    keypoint_dst='smplx',
    model_path='data/body_models/smplx',
    batch_size=1)

stages = [
    # stage 1
    dict(
        num_iter=10,
        fit_global_orient=True,
        fit_transl=True,
        fit_body_pose=False,
        fit_betas=False,
        fit_left_hand_pose=False,
        fit_right_hand_pose=False,
        fit_expression=False,
        fit_jaw_pose=False,
        fit_leye_pose=False,
        fit_reye_pose=False,
        joint_weights=dict(
            body_weight=5.0,
            use_shoulder_hip_only=True,
            hand_weight=0.0,
            face_weight=0.0)),
    # stage 2
    dict(
        num_iter=5,
        fit_global_orient=True,
        fit_transl=True,
        fit_body_pose=True,
        fit_betas=True,
        fit_left_hand_pose=False,
        fit_right_hand_pose=False,
        fit_expression=False,
        fit_jaw_pose=False,
        fit_leye_pose=False,
        fit_reye_pose=False,
        joint_weights=dict(
            body_weight=5.0,
            use_shoulder_hip_only=False,
            hand_weight=0.0,
            face_weight=0.0)),
    # stage 3
    dict(
        num_iter=3,
        fit_global_orient=True,
        fit_transl=True,
        fit_body_pose=True,
        fit_betas=True,
        fit_left_hand_pose=True,
        fit_right_hand_pose=True,
        fit_expression=False,
        fit_jaw_pose=False,
        fit_leye_pose=False,
        fit_reye_pose=False,
        joint_weights=dict(
            body_weight=10.0,
            use_shoulder_hip_only=False,
            hand_weight=1.0,
            face_weight=1.0))
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

camera = dict(
    type='PerspectiveCameras',
    convention='opencv',
    in_ndc=False,
    focal_length=5000,
    image_size=(224, 224),
    principal_point=(112, 112))
