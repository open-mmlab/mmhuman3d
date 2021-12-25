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
        'shader_type': 'flat',
        'texture_type': 'vertex',
        'light': base_directional_light,
        'material': base_material,
        'raster': {
            'type': 'mesh',
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'cull_to_frustum': True,
            'cull_backfaces': True,
        },
        'blend': white_blend_params,
    },
    'mq': {
        'renderer_type': 'base',
        'shader_type': 'gouraud',
        'texture_type': 'vertex',
        'light': base_directional_light,
        'material': base_material,
        'raster': {
            'type': 'mesh',
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'cull_to_frustum': True,
            'cull_backfaces': True,
        },
        'blend': white_blend_params,
    },
    'hq': {
        'renderer_type': 'base',
        'shader_type': 'phong',
        'texture_type': 'vertex',
        'light': base_directional_light,
        'material': base_material,
        'raster': {
            'type': 'mesh',
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'cull_to_frustum': False,
            'cull_backfaces': False,
        },
        'blend': white_blend_params,
    },
    'silhouette': {
        'renderer_type': 'silhouette',
        'shader_type': 'silhouette',
        'texture_type': 'vertex',
        'material': silhouete_material,
        'raster': {
            'type': 'mesh',
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'cull_to_frustum': False,
            'cull_backfaces': True,
        },
        'blend': black_blend_params,
    },
    'part_silhouette': {
        'renderer_type': 'base',
        'shader_type': 'nolight',
        'texture_type': 'closet',
        'light': None,
        'material': None,
        'raster': {
            'type': 'mesh',
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'cull_to_frustum': False,
            'cull_backfaces': True,
        },
        'blend': black_blend_params,
    },
    'depth': {
        'renderer_type': 'depth',
        'shader_type': 'nolight',
        'texture_type': 'vertex',
        'light': None,
        'material': None,
        'raster': {
            'type': 'mesh',
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'cull_to_frustum': False,
            'cull_backfaces': False,
        },
        'blend': white_blend_params,
    },
    'normal': {
        'renderer_type': 'normal',
        'shader_type': 'nolight',
        'texture_type': 'vertex',
        'light': None,
        'material': None,
        'raster': {
            'type': 'mesh',
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'cull_to_frustum': False,
            'cull_backfaces': False,
        },
        'blend': white_blend_params,
    },
    'pointcloud': {
        'renderer_type': 'pointcloud',
        'shader_type': 'nolight',
        'texture_type': 'vertex',
        'light': None,
        'material': None,
        'blend': white_blend_params,
        'bg_color': [
            1.0,
            1.0,
            1.0,
            0.0,
        ],
        'raster': {
            'type': 'point'
        },
        'points_per_pixel': 10,
        'radius': 0.003
    }
}
