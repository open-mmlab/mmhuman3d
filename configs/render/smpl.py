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
    'type': 'ambient',
    'ambient_color': [[1.0, 1.0, 1.0]],
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

empty_light = None

empty_material = {}

white_blend_params = {'background_color': (1.0, 1.0, 1.0)}

black_blend_params = {'background_color': (0.0, 0.0, 0.0)}

RENDER_CONFIGS = {
    'lq': {
        'renderer_type': 'base',
        'shader_type': 'flat',
        'texture_type': 'vertex',
        'raster_type': 'mesh',
        'light': base_directional_light,
        'material': base_material,
        'raster_settings': {
            'bin_size': 0,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'perspective_correct': False,
        },
        'blend': white_blend_params,
    },
    'mq': {
        'renderer_type': 'base',
        'shader_type': 'gouraud',
        'texture_type': 'vertex',
        'raster_type': 'mesh',
        'light': base_directional_light,
        'material': base_material,
        'raster_settings': {
            'bin_size': 0,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'perspective_correct': False,
        },
        'blend': white_blend_params,
    },
    'hq': {
        'renderer_type': 'base',
        'shader_type': 'phong',
        'texture_type': 'vertex',
        'raster_type': 'mesh',
        'light': base_directional_light,
        'material': base_material,
        'raster_settings': {
            'bin_size': 0,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'perspective_correct': False,
            'bin_size': 0,
        },
        'blend': white_blend_params,
    },
    'silhouette': {
        'renderer_type': 'silhouette',
        'shader_type': 'silhouette',
        'texture_type': 'vertex',
        'raster_type': 'mesh',
        'light': empty_light,
        'material': silhouete_material,
        'raster_settings': {
            'bin_size': 0,
            'blur_radius': 2e-5,
            'faces_per_pixel': 50,
            'perspective_correct': False,
        },
        'blend': black_blend_params,
    },
    'part_silhouette': {
        'renderer_type': 'segmentation',
        'shader_type': 'segmentation',
        'texture_type': 'nearest',
        'raster_type': 'mesh',
        'light': empty_light,
        'material': base_material,
        'raster_settings': {
            'bin_size': 0,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'perspective_correct': False,
        },
        'blend': black_blend_params,
    },
    'depth': {
        'renderer_type': 'depth',
        'shader_type': 'depth',
        'texture_type': 'vertex',
        'raster_type': 'mesh',
        'light': empty_light,
        'material': empty_material,
        'raster_settings': {
            'bin_size': 0,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'perspective_correct': False,
        },
        'blend': black_blend_params,
    },
    'normal': {
        'renderer_type': 'normal',
        'shader_type': 'normal',
        'texture_type': 'vertex',
        'raster_type': 'mesh',
        'light': empty_light,
        'material': empty_material,
        'raster_settings': {
            'bin_size': 0,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'perspective_correct': False,
        },
        'blend': white_blend_params,
    },
    'pointcloud': {
        'renderer_type': 'pointcloud',
        'shader_type': None,
        'raster_type': 'point',
        'light': empty_light,
        'material': empty_material,
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
