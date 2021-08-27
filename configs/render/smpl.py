base_directional_light = {
    'light_type': 'directional',
    'direction': [[1.0, 1.0, 3.0]],
    'ambient_color': [[0.5, 0.5, 0.5]],
    'diffuse_color': [[0.3, 0.3, 0.3]],
    'specular_color': [[0.2, 0.2, 0.2]],
}

base_point_light = {
    'light_type': 'point',
    'ambient_color': [[0.5, 0.5, 0.5]],
    'diffuse_color': [[0.3, 0.3, 0.3]],
    'specular_color': [[0.2, 0.2, 0.2]],
    'location': [[0.0, 0.0, 3.0]],
}

base_ambient_light = {
    'light_type': 'point',
    'ambient_color': [[1.0, 1.0, 1.0]],
    'diffuse_color': [[0, 0, 0]],
    'specular_color': [[0, 0, 0]],
    'location': [[0.0, 0.0, -3.0]],
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

base_camera = {
    'camera_type': 'weakpespective',
    'axis_sign': [0, 0, 0],
    'orbit_speed': None,
}

white_blend_params = {'background_color': (1.0, 1.0, 1.0)}

black_blend_params = {'background_color': (0.0, 0.0, 0.0)}

RENDER_CONFIGS = {
    'lq': {
        'light': base_directional_light,
        'camera': base_camera,
        'material': base_material,
        'raster': {
            'resolution': [256, 256],
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'cull_to_frustum': True,
            'cull_backfaces': True,
        },
        'shader': {
            'shader_type': 'flat',
        },
        'blend': white_blend_params,
    },
    'mq': {
        'light': base_directional_light,
        'camera': base_camera,
        'material': base_material,
        'raster': {
            'resolution': [512, 512],
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'cull_to_frustum': True,
            'cull_backfaces': True,
        },
        'shader': {
            'shader_type': 'gouraud',
        },
        'blend': white_blend_params,
    },
    'hq': {
        'light': base_directional_light,
        'camera': base_camera,
        'material': base_material,
        'raster': {
            'resolution': [1024, 1024],
            'blur_radius': 0,
            'faces_per_pixel': 1,
            'cull_to_frustum': False,
            'cull_backfaces': False,
        },
        'shader': {
            'shader_type': 'phong',
        },
        'blend': white_blend_params,
    },
    'silhouette': {
        'light': base_ambient_light,
        'camera': base_camera,
        'material': silhouete_material,
        'raster': {
            'resolution': None,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'cull_to_frustum': False,
            'cull_backfaces': True,
        },
        'shader': {
            'shader_type': 'silhouette',
        },
        'blend': black_blend_params,
    },
    'part_silhouette': {
        'light': base_ambient_light,
        'camera': base_camera,
        'material': silhouete_material,
        'raster': {
            'resolution': None,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'cull_to_frustum': False,
            'cull_backfaces': True,
        },
        'shader': {
            'shader_type': 'nolight',
        },
        'blend': black_blend_params,
    },
}
