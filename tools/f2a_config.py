img_res = 1000
texture_res = 1024

body_model = dict(
    type='SMPL',
    gender='neutral',
    num_betas=10,
    keypoint_src='smpl_45',
    keypoint_dst='smpl_45',
    model_path='data/body_models/smpl',
    batch_size=1)

stages = [
    dict(
        num_iter=500,
        batch_size=5,
        fit_displacement=True,
        plot_period=100,
        losses_config={
            "edge": {
                "weight": .1,
                "reduction": 'mean',
            },
            "normal": {
                "weight": .01,
                "reduction": 'mean',
            },
            "laplacian": {
                "weight": .01,
                "reduction": 'mean',
            },
            "min": {
                "weight": 10,
                "reduction": 'mean',
                "min_bound": 0
            },
            "max": {
                "weight": 10,
                "reduction": 'mean',
                "max_bound": 0.03,
            },
            "mse_background_intersection": {
                "weight": 20.0,
                "reduction": 'mean',
            },
            "mse_wrap_visible": {
                "weight": 20.0,
                "reduction": 'mean',
                "use_visible_mask": True
            },
            "kp2d_mse_loss": {
                "weight": 0,
                "reduction": 'mean',
            },
            "displacement_smooth": {
                "weight": 0.,
                "reduction": 'mean',
                "strides": [1],
                "resolution": (512, 512),
            },
            "normal_smooth": {
                "weight": 0.,
                "reduction": 'mean',
                "strides": [1],
                "resolution": (512, 512),
            },
        }),
    dict(
        num_iter=500,
        batch_size=5,
        plot_period=100,
        fit_background=True,
        losses_config={
            "mse_background_image": {
                "weight": .0,
                "reduction": 'mean',
            },
        },
    ),
    dict(
        num_iter=500,
        batch_size=5,
        plot_period=100,
        fit_texture=True,
        losses_config={
            "texture_mse": {
                "weight": 0.0,
                "reduction": 'mean',
            },
            "texture_smooth": {
                "weight": 0.0,
                "reduction": 'mean',
                "strides": [1],
            },
            "texture_min": {
                "weight": 0.0,
                "reduction": 'mean',
                "min_bound": 0,
            },
            "texture_max": {
                "weight": 0.0,
                "reduction": 'mean',
                "max_bound": 1,
            },
        },
    ),
]