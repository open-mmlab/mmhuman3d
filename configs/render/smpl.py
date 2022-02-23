base_directional_light = {
    'type': 'directional',
    'direction': [[1, 1, 1]],
    'ambient_color': [[0.5, 0.5, 0.5]],
    'diffuse_color': [[0.5, 0.5, 0.5]],
    'specular_color': [[0.5, 0.5, 0.5]],
}

base_point_light = {
    'type': 'point',
    'ambient_color': [[1, 1, 1]],
    'diffuse_color': [[0.3, 0.3, 0.3]],
    'specular_color': [[0.5, 0.5, 0.5]],
    'location': [[2.0, 2.0, -2.0]],
}

base_ambient_light = {
    'type': 'ambient',
    'ambient_color': [[1.0, 1.0, 1.0]],
}

base_material = {
    'ambient_color': [[1, 1, 1]],
    'diffuse_color': [[0.5, 0.5, 0.5]],
    'specular_color': [[0.15, 0.15, 0.15]],
    'shininess': 60.0,
}

silhouete_material = {
    'ambient_color': [[1.0, 1.0, 1.0]],
    'diffuse_color': [[0.0, 0.0, 0.0]],
    'specular_color': [[0.0, 0.0, 0.0]],
    'shininess': 1.0,
}

white_blend_params = {'background_color': (1.0, 1.0, 1.0)}

black_blend_params = {'background_color': (0.0, 0.0, 0.0)}

RENDER_CONFIGS = {
    # low quality
    'lq': {
        'type': 'mesh',
        'shader': {
            'type': 'hard_flat'
        },
        'lights': base_directional_light,
        'materials': base_material,
        'rasterizer': {
            'bin_size': 0,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'perspective_correct': False,
        },
        'blend_params': white_blend_params,
    },
    # medium quality
    'mq': {
        'type': 'mesh',
        'shader': {
            'type': 'soft_gouraud'
        },
        'lights': base_directional_light,
        'materials': base_material,
        'rasterizer': {
            'bin_size': 0,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'perspective_correct': False,
        },
        'blend_params': white_blend_params,
    },
    # high quality
    'hq': {
        'type': 'mesh',
        'shader': {
            'type': 'soft_phong'
        },
        'lights': base_directional_light,
        'materials': base_material,
        'rasterizer': {
            'bin_size': 0,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'perspective_correct': False,
        },
        'blend_params': white_blend_params,
    },
    'silhouette': {
        'type': 'silhouette',
        'lights': None,
        'materials': silhouete_material,
        'rasterizer': {
            'bin_size': 0,
            'blur_radius': 2e-5,
            'faces_per_pixel': 50,
            'perspective_correct': False,
        },
        'blend_params': black_blend_params,
    },
    'part_silhouette': {
        'type': 'segmentation',
        'material': base_material,
        'rasterizer': {
            'bin_size': 0,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'perspective_correct': False,
        },
        'blend_params': black_blend_params,
    },
    'depth': {
        'type': 'depth',
        'rasterizer': {
            'bin_size': 0,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'perspective_correct': False,
        },
        'blend_params': black_blend_params,
    },
    'normal': {
        'type': 'normal',
        'rasterizer': {
            'bin_size': 0,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'perspective_correct': False,
        },
        'blend_params': white_blend_params,
    },
    'pointcloud': {
        'type': 'pointcloud',
        'compositor': {
            'background_color': [
                1.0,
                1.0,
                1.0,
                0.0,
            ],
        },
        'rasterizer': {
            'points_per_pixel': 10,
            'radius': 0.003,
            'bin_size': None,
            'max_points_per_bin': None,
        }
    }
}
