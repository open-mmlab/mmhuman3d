import open3d as o3d
from mmhuman3d.core import optimizer
from mmhuman3d.data.data_converters import build_data_converter
from mmhuman3d.data.datasets import HumanImageSMPLXDataset
from mmhuman3d.models.body_models.builder import build_body_model
from mmhuman3d.core.optimizer import build_optimizers
import torch
import cv2
import numpy as np

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
def test_ffhq_converter():
    cfg = dict(type='FFHQFlameConverter')
    data_converter = build_data_converter(cfg)
    data_converter.convert_by_mode(
        '/mnt/lustre/shizhelun/data/ffhq',
        '/mnt/lustre/shizhelun/mmhuman3d/preprocessed_npzs',
        'train',
    )
    import ipdb;ipdb.set_trace()

def test_ffhq_dataset():
    data_keys = [
        'has_smplx', 'has_keypoints3d', 'has_keypoints2d',
        'smplx_jaw_pose', 
        'smplx_global_orient', 'smplx_betas','keypoints2d',
        'keypoints3d', 'sample_idx', 'smplx_expression'
    ]
    train_dataset = HumanImageSMPLXDataset(
        data_prefix = '/mnt/lustre/shizhelun/data',
        pipeline= [
            dict(type = 'LoadImageFromFile'),
            dict(type = 'BBoxCenterJitter', factor = 0.2, dist = 'uniform'),
            dict(type = 'RandomHorizontalFlip', flip_prob = 0.5, convention = 'flame'), # hand = 0,head = body = 0.5
            dict(type = 'GetRandomScaleRotation', rot_factor = 30.0, scale_factor = 0.0, rot_prob = 0.6),
            dict(type = 'MeshAffine', img_res = 256), #hand = 224, body = head = 256
            dict(type = 'RandomChannelNoise', noise_factor = 0.4),
            dict(type = 'SimulateLowRes', 
                 dist = 'categorical', 
                 cat_factors = (1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 8.0),
                 # head = (1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 8.0)
                 # hand = (1.0, 1.2, 1.5, 2.0, 3.0, 4.0)
                 # body = (1.0,)
                 factor_min = 1.0,
                 factor_max = 1.0
                 ),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img','ori_img']),
            dict(type='ToTensor', keys=data_keys),

        ],
        dataset_name='ffhq',
        ann_file='ffhq_flame_train.npz',
        convention='flame',
        num_betas=100,
        num_expression=50
    )
    data = train_dataset[0]
    img = data['img'].clone()
    kps2d = data['keypoints2d']
    viz(img,kps2d,'vizimgs/head/test.png')
    flame_model_dict = dict(
        type='FLAME',
        num_expression_coeffs = 50,
        num_betas = 100,
        use_face_contour = True,
        model_path='/mnt/lustre/shizhelun/data/body_models/flame',
        keypoint_src='flame',
        keypoint_dst='human_data',
    )
    flame_model = build_body_model(flame_model_dict)
    flame_output = flame_model(
        global_orient = data['smplx_global_orient'].unsqueeze(0),
        jaw_pose = data['smplx_jaw_pose'].unsqueeze(0),
        betas = data['smplx_betas'].unsqueeze(0),
        expression = data['smplx_expression'].unsqueeze(0)
    )
    import ipdb;ipdb.set_trace()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(flame_output['vertices'][0].detach().numpy())
    mesh.triangles = o3d.utility.Vector3iVector(flame_model.faces)
    o3d.io.write_triangle_mesh("vizimgs/head/test.ply", mesh)
    # import ipdb;ipdb.set_trace()

def viz(img, kps2d, out_path):
    import numpy as np
    import cv2
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    img = img.clone()
    mean = torch.tensor(img_norm_cfg['mean'], dtype=img.dtype, device=img.device)
    std = torch.tensor(img_norm_cfg['std'], dtype=img.dtype, device=img.device)
    img.mul_(std[:,None,None]).add_(mean[:,None,None])
    img = img.permute(1,2,0)
    img = np.array(img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    for index, xy in enumerate(kps2d):
        cv2.circle(img,(int(xy[0]),int(xy[1])),1,(0,0,255),1)
        cv2.putText(img, str(index), (int(xy[0]),int(xy[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 1)
    cv2.imwrite(out_path,img)

def test_curated_fits_converter():
    cfg = dict(type= 'ExposeCuratedFitsConverter')
    data_converter = build_data_converter(cfg)
    data_converter.convert_by_mode(
        '/mnt/lustre/shizhelun/data/ExPose',
        '/mnt/lustre/shizhelun/data/preprocessed_datasets',
        'train',
    )


def test_curated_fits_dataset():
    data_keys = [
        'has_smplx', 'has_keypoints3d', 'has_keypoints2d',
        'smplx_jaw_pose', 'smplx_body_pose', 'smplx_right_hand_pose', 'smplx_left_hand_pose', 
        'smplx_global_orient', 'smplx_betas','keypoints2d',
        'keypoints3d', 'sample_idx', 'smplx_expression'
    ]
    train_dataset = HumanImageSMPLXDataset(
        data_prefix = 'data',
        pipeline= [
            dict(type = 'LoadImageFromFile'),
            dict(type = 'BBoxCenterJitter', factor = 0.2, dist = 'uniform'),
            dict(type = 'RandomHorizontalFlip', flip_prob = 0.5, convention = 'smplx'), # hand = 0,head = body = 0.5
            dict(type = 'GetRandomScaleRotation', rot_factor = 30.0, scale_factor = 0.2, rot_prob = 1.0),
            dict(type = 'Rotation'),
            dict(type = 'MeshAffine', img_res = 256), #hand = 224, body = head = 256
            dict(type = 'RandomChannelNoise', noise_factor = 0.4),
            dict(type = 'SimulateLowRes', 
                 dist = 'categorical', 
                 cat_factors = (1.0,),
                 # head = (1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 8.0)
                 # hand = (1.0, 1.2, 1.5, 2.0, 3.0, 4.0)
                 # body = (1.0,)
                 factor_min = 1.0,
                 factor_max = 1.0
                 ),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img','ori_img']),
            dict(type='ToTensor', keys=data_keys),

        ],
        dataset_name='',
        ann_file='curated_fits_train.npz',
        convention='smplx',
        num_betas=10,
        num_expression=10
    )
    data = train_dataset[0]
    import ipdb;ipdb.set_trace()
    img = data['img'].clone()
    kps2d = data['keypoints2d']
    viz(img,kps2d,'vizimgs/body/test.png')
    smplx_model_dict = dict(
        type='SMPLX',
        num_expression_coeffs = 10,
        num_betas = 10,
        flat_hand_mean = True,
        use_face_contour = True,
        use_pca = False,
        model_path='/mnt/lustre/shizhelun/data/body_models/smplxv1.0/smplx',
        keypoint_src='smplx',
        keypoint_dst='human_data',

    )
    smplx_model = build_body_model(smplx_model_dict)
    smplx_output = smplx_model(
        global_orient = data['smplx_global_orient'].unsqueeze(0),
        jaw_pose = data['smplx_jaw_pose'].unsqueeze(0),
        body_pose = data['smplx_body_pose'].unsqueeze(0),
        right_hand_pose = data['smplx_right_hand_pose'].unsqueeze(0),
        left_hand_pose = data['smplx_left_hand_pose'].unsqueeze(0),
        betas = data['smplx_betas'].unsqueeze(0),
        expression = data['smplx_expression'].unsqueeze(0)
    )
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(smplx_output['vertices'][0].detach().numpy())
    mesh.triangles = o3d.utility.Vector3iVector(smplx_model.faces)
    o3d.io.write_triangle_mesh("vizimgs/body/test.ply", mesh)
    import ipdb;ipdb.set_trace()

def test_expose_spin_smplx_converter():
    cfg = dict(type='ExposeSPINSMPLXConverter')
    data_converter = build_data_converter(cfg)
    data_converter.convert_by_mode(
        '/mnt/lustre/shizhelun/data/spin_in_smplx',
        '/mnt/lustre/shizhelun/data/preprocessed_datasets',
        'train',
    )
    import ipdb;ipdb.set_trace()

def test_expose_spin_smplx_dataset():
    data_keys = [
        'has_smplx', 'has_keypoints3d', 'has_keypoints2d',
        'smplx_jaw_pose', 'smplx_body_pose', 'smplx_right_hand_pose', 'smplx_left_hand_pose', 
        'smplx_global_orient', 'smplx_betas','keypoints2d',
        'keypoints3d', 'sample_idx', 'smplx_expression'
    ]
    train_dataset = HumanImageSMPLXDataset(
        data_prefix = '/mnt/lustre/shizhelun/data',
        pipeline= [
            dict(type = 'LoadImageFromFile'),
            dict(type = 'BBoxCenterJitter', factor = 0.2, dist = 'uniform'),
            dict(type = 'RandomHorizontalFlip', flip_prob = 0.5, convention = 'smplx'), # hand = 0,head = body = 0.5
            dict(type = 'GetRandomScaleRotation', rot_factor = 30.0, scale_factor = 0.2, rot_prob = 0.6),
            dict(type = 'MeshAffine', img_res = 256), #hand = 224, body = head = 256
            dict(type = 'RandomChannelNoise', noise_factor = 0.4),
            dict(type = 'SimulateLowRes', 
                 dist = 'categorical', 
                 cat_factors = (1.0,),
                 # head = (1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 8.0)
                 # hand = (1.0, 1.2, 1.5, 2.0, 3.0, 4.0)
                 # body = (1.0,)
                 factor_min = 1.0,
                 factor_max = 1.0
                 ),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img','ori_img']),
            dict(type='ToTensor', keys=data_keys),

        ],
        dataset_name='',
        ann_file='spin_smplx_train.npz',
        convention='smplx',
        num_betas=10,
        num_expression=10
    )
    data = train_dataset[0]
    img = data['img'].clone()
    kps2d = data['keypoints2d']
    viz(img,kps2d,'vizimgs/body/test.png')
    import ipdb;ipdb.set_trace()
    smplx_model_dict = dict(
        type='SMPLX',
        num_expression_coeffs = 10,
        num_betas = 10,
        use_face_contour = True,
        use_pca = False,
        model_path='/mnt/lustre/shizhelun/data/body_models/smplxv1.0/smplx',
        keypoint_src='smplx',
        keypoint_dst='human_data',

    )
    smplx_model = build_body_model(smplx_model_dict)
    smplx_output = smplx_model(
        global_orient = data['smplx_global_orient'].unsqueeze(0),
        jaw_pose = data['smplx_jaw_pose'].unsqueeze(0),
        body_pose = data['smplx_body_pose'].unsqueeze(0),
        right_hand_pose = data['smplx_right_hand_pose'].unsqueeze(0),
        left_hand_pose = data['smplx_left_hand_pose'].unsqueeze(0),
        betas = data['smplx_betas'].unsqueeze(0),
        expression = data['smplx_expression'].unsqueeze(0)
    )
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(smplx_output['vertices'][0].detach().numpy())
    mesh.triangles = o3d.utility.Vector3iVector(smplx_model.faces)
    o3d.io.write_triangle_mesh("vizimgs/body/test.ply", mesh)
    
def test_freihand_converter():
    cfg = dict(type='FreihandConverter')
    data_converter = build_data_converter(cfg)
    data_converter.convert_by_mode(
        '/mnt/lustre/shizhelun/data/datasets/FreiHand',
        '/mnt/lustre/shizhelun/data/preprocessed_datasets',
        'train',
        'data/body_models/all_means.pkl'
    )

def test_freihand_dataset():
    data_keys = [
        'has_smplx', 'has_keypoints3d', 'has_keypoints2d', 'smplx_right_hand_pose',
        'smplx_global_orient', 'smplx_betas','keypoints2d',
        'keypoints3d', 'sample_idx'
    ]
    train_dataset = HumanImageSMPLXDataset(
        data_prefix = '/mnt/lustre/shizhelun/data',
        pipeline= [
            dict(type = 'LoadImageFromFile'),
            dict(type = 'BBoxCenterJitter', factor = 0.2, dist = 'uniform'),
            dict(type = 'RandomHorizontalFlip', flip_prob = 0.0, convention = 'smplx'), # hand = 0,head = body = 0.5
            dict(type = 'GetRandomScaleRotation', rot_factor = 30.0, scale_factor = 0.2, rot_prob = 0.6),
            dict(type = 'MeshAffine', img_res = 224), #hand = 224, body = head = 256
            dict(type = 'RandomChannelNoise', noise_factor = 0.4),
            dict(type = 'SimulateLowRes', 
                 dist = 'categorical', 
                 cat_factors = (1.0, 1.2, 1.5, 2.0, 3.0, 4.0),
                 # head = (1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 8.0)
                 # hand = (1.0, 1.2, 1.5, 2.0, 3.0, 4.0)
                 # body = (1.0,)
                 factor_min = 1.0,
                 factor_max = 1.0
                 ),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img','ori_img']),
            dict(type='ToTensor', keys=data_keys),

        ],
        dataset_name='FreiHand',
        ann_file='freihand_train.npz',
        convention='mano',
        num_betas=10,
        num_expression=10
    )
    data = train_dataset[0]
    img = data['img'].clone()
    kps2d = data['keypoints2d']
    viz(img,kps2d,'vizimgs/hand/test.png')
    mano_model_dict = dict(
        type='mano',
        model_path='/mnt/lustre/shizhelun/data/body_models/mano/mano_v1_2/models/MANO_RIGHT.pkl',
        num_pca_comps = 45,
        flat_hand_mean = True,
        keypoint_src = 'mano',
        keypoint_dst = 'mano',
    )

    mano_model = build_body_model(mano_model_dict)
    mano_output = mano_model(
        global_orient = data['smplx_global_orient'],
        right_hand_pose = data['smplx_right_hand_pose'].reshape(1,-1),
        betas = data['smplx_betas'].unsqueeze(0),
    )
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(mano_output['vertices'][0].detach().cpu().numpy())
    mesh.triangles = o3d.utility.Vector3iVector(mano_model.faces)
    o3d.io.write_triangle_mesh("vizimgs/hand/rep_verts.ply", mesh)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(mano_output['joints'][0].detach().cpu().numpy())
    o3d.io.write_triangle_mesh("vizimgs/hand/rep_joints.ply", mesh)
    import ipdb;ipdb.set_trace()


def viz_results():
    data = np.load('result.npy', allow_pickle= True).item()
    pred_vertices = data['pred_vertices']
    smplx_model_dict = dict(
        type='SMPLXLayer',
        num_expression_coeffs = 10,
        num_betas = 10,
        use_face_contour = True,
        use_pca = False,
        flat_hand_mean = True,
        model_path='data/body_models/smplx',
        keypoint_src='smplx',
        keypoint_dst='smplx',
    )
    mano_dict = dict(
        type='MANOLayer',
        num_expression_coeffs = 10,
        num_betas = 10,
        use_pca = False,
        flat_hand_mean = True,
        model_path='data/body_models/mano',
        keypoint_src='mano',
        keypoint_dst='mano',
    )
    flame_dict=dict(
        type='flamelayer',
        num_expression_coeffs = 50,
        num_betas = 100,
        use_pca = False,
        use_face_contour = True,
        model_path='data/body_models/flame',
        keypoint_src='flame',
        keypoint_dst='flame',
    )
    smplx_model = build_body_model(smplx_model_dict)
    mano_model = build_body_model(mano_dict)
    flame_model = build_body_model(flame_dict)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(pred_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(smplx_model.faces)
    o3d.io.write_triangle_mesh("vizimgs/body/body.ply", mesh)
    # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(pred_vertices[48])
    # mesh.triangles = o3d.utility.Vector3iVector(mano_model.faces)
    # o3d.io.write_triangle_mesh("left_handfromhand.ply", mesh)

def output_hand_mean(hand_mean):
    mano_dict = dict(
        type='MANOLayer',
        num_expression_coeffs = 10,
        num_betas = 10,
        use_pca = False,
        flat_hand_mean = True,
        model_path='data/body_models/mano',
        keypoint_src='mano',
        keypoint_dst='mano',
    )
    mano_model = build_body_model(mano_dict).cuda()
    from mmhuman3d.utils.geometry import rot6d_to_rotmat
    global_orient = rot6d_to_rotmat(hand_mean[:,:6]).reshape(-1,1,3,3)
    hand_pose = rot6d_to_rotmat(hand_mean[:,6:96].reshape(-1,6)).reshape(-1,15,3,3)
    mano_output = mano_model(global_orient = global_orient, right_hand_pose = hand_pose)
    pred_vertices = mano_output['vertices'].detach().cpu().numpy()
    import numpy as np
    np.save('result.npy',dict(pred_vertices = pred_vertices))

def output_hand_prediction(hand_predictions):
    mano_dict = dict(
        type='MANOLayer',
        num_expression_coeffs = 10,
        num_betas = 10,
        use_pca = False,
        flat_hand_mean = True,
        model_path='data/body_models/mano',
        keypoint_src='mano',
        keypoint_dst='mano',
    )
    mano_model = build_body_model(mano_dict).cuda()
    pred_param = hand_predictions['pred_param']
    mano_output = mano_model(global_orient = pred_param['global_orient'], right_hand_pose = pred_param['right_hand_pose'])
    pred_vertices = mano_output['vertices'].detach().cpu().numpy()
    import numpy as np
    np.save('result.npy',dict(pred_vertices = pred_vertices))

def output_body_prediction(body_predictions):
    pred_param = body_predictions['pred_param']
    smplx_model_dict = dict(
        type='SMPLXLayer',
        num_expression_coeffs = 10,
        num_betas = 10,
        use_face_contour = True,
        use_pca = False,
        flat_hand_mean = True,
        model_path='data/body_models/smplx',
        keypoint_src='smplx',
        keypoint_dst='smplx',
    )
    smplx_model = build_body_model(smplx_model_dict).cuda()
    pred_output = smplx_model(**pred_param)
    pred_vertices = pred_output['vertices'].detach().cpu().numpy()
    import numpy as np
    np.save('result.npy',dict(pred_vertices = pred_vertices))

def output_face_mean(face_mean):
    flame_dict=dict(
        type='flamelayer',
        num_expression_coeffs = 50,
        num_betas = 100,
        use_pca = False,
        use_face_contour = True,
        model_path='data/body_models/flame',
        keypoint_src='flame',
        keypoint_dst='flame',
    )
    flame_model = build_body_model(flame_dict).cuda()
    from mmhuman3d.utils.geometry import rot6d_to_rotmat
    global_orient = rot6d_to_rotmat(face_mean[:,:6]).reshape(-1,1,3,3)
    jaw_pose = rot6d_to_rotmat(face_mean[:,6:12].reshape(-1,6)).reshape(-1,1,3,3)
    flame_output = flame_model(global_orient = global_orient, jaw_pose = jaw_pose)
    pred_vertices = flame_output['vertices'].detach().cpu().numpy()
    import numpy as np
    np.save('result.npy',dict(pred_vertices = pred_vertices))

if __name__ == '__main__':
    test_curated_fits_dataset()