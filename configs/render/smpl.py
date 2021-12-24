base_directional_light = {
    'type': 'directional',
    'direction': [[10.0, 10.0, 10.0]],
    'ambient_color': [[0.5, 0.5, 0.5]],
    'diffuse_color': [[0.5, 0.5, 0.5]],
    'specular_color': [[0.5, 0.5, 0.5]],
}

base_point_light = {
    'type': 'point',
    'ambient_color': [[0.5, 0.5, 0.5]],
    'diffuse_color': [[0.3, 0.3, 0.3]],
    'specular_color': [[0.5, 0.5, 0.5]],
    'location': [[2.0, 2.0, -2.0]],
}

base_ambient_light = {
    'type': 'directional',
    'ambient_color': [[1.0, 1.0, 1.0]],
    'diffuse_color': [[0, 0, 0]],
    'specular_color': [[0, 0, 0]],
    'direction': [[10.0, 10.0, -10.0]],
}

base_material = {
    'ambient_color': [[1, 1, 1]],
    'diffuse_color': [[0.5, 0.5, 0.5]],
    'specular_color': [[0.5, 0.5, 0.5]],
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
    'lq': {
        'renderer_type': 'base',
        'light': base_directional_light,
        'material': base_material,
        'raster': {
            'type': 'mesh',
            'resolution': [256, 256],
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'cull_to_frustum': True,
            'cull_backfaces': True,
        },
        'shader': {
            'type': 'flat',
        },
        'texture': {
            'type': 'vertex'
        },
        'blend': white_blend_params,
    },
    'mq': {
        'renderer_type': 'base',
        'light': base_directional_light,
        'material': base_material,
        'raster': {
            'type': 'mesh',
            'resolution': [512, 512],
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'cull_to_frustum': True,
            'cull_backfaces': True,
        },
        'shader': {
            'type': 'gouraud',
        },
        'texture': {
            'type': 'vertex'
        },
        'blend': white_blend_params,
    },
    'hq': {
        'renderer_type': 'base',
        'light': base_directional_light,
        'material': base_material,
        'raster': {
            'type': 'mesh',
            'resolution': [1024, 1024],
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'cull_to_frustum': False,
            'cull_backfaces': False,
        },
        'shader': {
            'type': 'phong',
        },
        'texture': {
            'type': 'vertex'
        },
        'blend': white_blend_params,
    },
    'silhouette': {
        'renderer_type': 'silhouette',
        'light': None,
        'material': silhouete_material,
        'raster': {
            'type': 'mesh',
            'resolution': [512, 512],
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'cull_to_frustum': False,
            'cull_backfaces': True,
        },
        'shader': {
            'type': 'silhouette',
        },
        'texture': {
            'type': 'vertex'
        },
        'blend': black_blend_params,
    },
    'part_silhouette': {
        'renderer_type': 'base',
        'light': None,
        'material': None,
        'raster': {
            'type': 'mesh',
            'resolution': [512, 512],
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'cull_to_frustum': False,
            'cull_backfaces': True,
        },
        'shader': {
            'type': 'nolight',
        },
        'texture': {
            'type': 'closet'
        },
        'blend': black_blend_params,
    },
    'depth': {
        'renderer_type': 'depth',
        'light': None,
        'material': None,
        'raster': {
            'type': 'mesh',
            'resolution': [512, 512],
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'cull_to_frustum': False,
            'cull_backfaces': False,
        },
        'shader': {
            'type': 'nolight',
        },
        'texture': {
            'type': 'vertex'
        },
        'blend': white_blend_params,
    },
    'normal': {
        'renderer_type': 'normal',
        'light': None,
        'material': None,
        'raster': {
            'type': 'mesh',
            'resolution': [512, 512],
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'cull_to_frustum': False,
            'cull_backfaces': False,
        },
        'shader': {
            'type': 'nolight',
        },
        'texture': {
            'type': 'vertex'
        },
        'blend': white_blend_params,
    },
    'pointcloud': {
        'renderer_type': 'pointcloud',
        'light': base_directional_light,
        'material': None,
        'raster': {
            'type': 'point',
            'resolution': [512, 512]
        },
        'shader': {
            'type': 'nolight',
        },
        'texture': None,
        'blend': white_blend_params,
        'bg_color': [
            1.0,
            1.0,
            1.0,
            0.0,
        ],
        'points_per_pixel': 10,
        'radius': 0.003
    }
}
