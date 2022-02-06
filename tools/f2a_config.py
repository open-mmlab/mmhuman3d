type = 'flow2avatar'
img_res = 1000
texture_res = 1024
uv_res = 512

select_frame = dict(
    temporal_successive=True, interval_range=5, fix_interval=True)

body_model = dict(
    type='smpld',
    gender='neutral',
    num_betas=10,
    keypoint_src='smpl_45',
    keypoint_dst='smpl_45',
    texture_res=texture_res,
    create_displacement=True,
    create_texture=True,
    model_path='',
    batch_size=1)

renderer_rgb = dict(
    type='base',
    rasterizer=dict(
        image_size=img_res,
        blur_radius=0.0,
        faces_per_pixel=1,
        perspective_correct=False,
    ),
    shader=dict(type='SoftPhongShader'))

renderer_silhouette = dict(
    type='silhouette',
    rasterizer=dict(
        image_size=img_res,
        blur_radius=2e-5,
        bin_size=None,
        faces_per_pixel=50,
        perspective_correct=False,
    ),
    shader=dict(type='SilhouetteShader'))

renderer_flow = dict(
    type='opticalflow',
    rasterizer=dict(
        image_size=img_res,
        blur_radius=0.0,
        faces_per_pixel=1,
        perspective_correct=False,
    ),
    shader=dict(type='OpticalFlowShader'))

renderer_uv = dict(type='UVRenderer', resolution=uv_res, model_type='smpl')

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
                "weight": .005,
                "reduction": 'mean',
            },
            "laplacian": {
                "weight": .01,
                "reduction": 'mean',
            },
            "displacement_min": {
                "weight": 20,
                "reduction": 'mean',
                "min_bound": 0
            },
            "displacement_max": {
                "weight": 20,
                "reduction": 'mean',
                "max_bound": {
                    'FOOT': 0.015,
                    'HAND': 0.005,
                    'LEG': 0.025,
                    'ARM': 0.02,
                    'HEAD': 0.01,
                    'UPBODY': 0.025,
                    'DOWNBODY': 0.025
                }
            },
            "displacement_smooth": {
                "weight": 0.,
                "reduction": 'mean',
                "strides": [1],
                "resolution": uv_res,
            },
            "mse_silhouette_background": {
                "weight": 20.0,
                "reduction": 'mean',
            },
            "mse_flow_visible": {
                "weight": 20.0,
                "reduction": 'mean',
                "use_visible_mask": True
            },
            "kp2d_mse_loss": {
                "weight": 0,
                "reduction": 'mean',
            },
            "normal_smooth": {
                "weight": 0.,
                "reduction": 'mean',
                "strides": [1],
                "resolution": uv_res,
            },
        }),
    dict(
        num_iter=100,
        batch_size=5,
        fit_displacement=True,
        plot_period=50,
        losses_config={
            # "edge": {
            #     "weight": .1,
            #     "reduction": 'mean',
            # },
            # "normal": {
            #     "weight": .005,
            #     "reduction": 'mean',
            # },
            # "laplacian": {
            #     "weight": .01,
            #     "reduction": 'mean',
            # },
            "displacement_min": {
                "weight": 20,
                "reduction": 'mean',
                "min_bound": 0
            },
            "displacement_max": {
                "weight": 20,
                "reduction": 'mean',
                "max_bound": {
                    'FOOT': 0.015,
                    'HAND': 1e-5,
                    'LEG': 0.025,
                    'ARM': 0.02,
                    'HEAD': 1e-5,
                    'UPBODY': 0.025,
                    'DOWNBODY': 0.025
                }
            },
        },
    ),
    dict(
        num_iter=500,
        batch_size=1,
        plot_period=100,
        fit_background=True,
        losses_config={
            "mse_background_image": {
                "weight": 1.0,
                "reduction": 'mean',
            },
        },
    ),
    dict(
        num_iter=50,
        batch_size=5,
        plot_period=100,
        fit_texture=True,
        losses_config={
            "texture_mse": {
                "weight": 100.0,
                "reduction": 'mean',
            },
            "texture_smooth": {
                "weight": .0,
                "reduction": 'mean',
                "strides": [1],
            },
            "texture_min": {
                "weight": 10,
                "reduction": 'mean',
                "min_bound": 0.,
            },
            "texture_max": {
                "weight": 10,
                "reduction": 'mean',
                "max_bound": 1.0,
            },
        },
    ),
]

optimizer = dict(type='SGD', lr=1.0, momentum=0.9)
