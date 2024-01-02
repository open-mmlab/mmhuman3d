import glob
import os
import pdb
import random
import json
import pickle

import cv2
import numpy as np
import torch
from tqdm import tqdm

from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_idx,
    get_keypoint_idxs_by_part,
)
from mmhuman3d.models.body_models.utils import transform_to_camera_frame
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.body_models.builder import build_body_model
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


import pdb


@DATA_CONVERTERS.register_module()
class DecoConverter(BaseModeConverter):

    ACCEPTED_MODES = ['train', 'test']

    def __init__(self, modes=[], *args, **kwargs):

        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.misc_config = dict(
            bbox_source='keypoints2d_smpl',
            smpl_source='original',
            cam_param_type='prespective',
            kps3d_root_aligned=False,
            has_gender=True,
            contact_label=['smpl_vertex', 'polygon_2d'],
        )
        
        self.smpl_shape = {
            'body_pose': (-1, 69),
            'betas': (-1, 10),
            'global_orient': (-1, 3),
            'transl': (-1, 3),} 
        
        super(DecoConverter, self).__init__(modes)


    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:
        print('Converting Deco dataset...')

        # build smpl model
        smpl_model = build_body_model(
            dict(
                type='SMPL',
                keypoint_src='smpl_45',
                keypoint_dst='smpl_45',
                model_path='data/body_models/smpl',
                gender='neutral',
                num_betas=10,
                use_pca=False,
                batch_size=1)).to(self.device)

        # use HumanData to store data
        human_data = HumanData()

        # initialize HumanData
        smpl_ = {}
        for key in self.smpl_shape.keys():
            smpl_[key] = []
        bboxs_ = {}
        for key in ['bbox_xywh']:
            bboxs_[key] = []
        image_path_ = []
        keypoints2d_smpl_, keypoints3d_smpl = [], []
        meta_ = {}
        for meta_key in ['principal_point', 'focal_length', 'height', 'width']:
            meta_[meta_key] = []
        contact_ = {}
        for contact_key in self.misc_config['contact_label']:
            contact_[contact_key] = []

        # load annotations
        if mode == 'test':
            anno_path = os.path.join(dataset_path, 'hot_dca_test.npz')
        else:
            anno_path = os.path.join(dataset_path, 'hot_dca_trainval.npz')
        anno = dict(np.load(anno_path, allow_pickle=True))

        
        seed, size = '231228', '9999'
        random.seed(int(seed))        
        
        # build smpl and get kps3d
        body_pose = np.array(anno['pose'])[:, 3:]
        global_orient = np.array(anno['pose'])[:, :3]
        betas = np.array(anno['shape'])
        transl = np.array(anno['transl'])

        output = smpl_model(
            body_pose=torch.tensor(body_pose).float().to(self.device),
            global_orient=torch.tensor(global_orient).float().to(self.device),
            betas=torch.tensor(betas).float().to(self.device),
            transl=torch.tensor(transl).float().to(self.device),
        )
        smpl_joints = output['joints']
        kps3d = smpl_joints.detach().cpu().numpy()

        # iterate over all images
        for iid, imgn in enumerate(tqdm(anno['imgname'])):

            imgp = imgn.replace('datasets/', '')

            # get height and width
            image_path = os.path.join(dataset_path, imgp)
            image = cv2.imread(image_path)
            height, width = image.shape[:2]

            # get cam params
            intrinsics = np.array(anno['cam_k'][iid])
            focal_length = [intrinsics[0, 0], intrinsics[1, 1]]
            principal_point = [intrinsics[0, 2], intrinsics[1, 2]]

            # build camera
            camera = build_cameras(
                dict(
                    type='PerspectiveCameras',
                    convention='opencv',
                    in_ndc=False,
                    focal_length=focal_length,
                    image_size=(width, height),
                    principal_point=principal_point)).to(self.device)

            kps2d = camera.transform_points_screen(smpl_joints[iid])[..., :2].detach().cpu().numpy()

            # test write kps2d
            # img = cv2.imread(f'{image_path}')
            # for kp in kps2d:
            #     if  0 < kp[0] < width and 0 < kp[1] < height: 
            #         cv2.circle(img, (int(kp[0]), int(kp[1])), 3, (0,0,255), 1)
            #     pass
            # # write image
            # os.makedirs(f'{out_path}', exist_ok=True)
            # cv2.imwrite(f'{out_path}/{iid}.jpg', img)

            bbox_xywh = self._keypoints_to_scaled_bbox(kps2d, scale=1.25)


            # image path
            image_path_.append(image_path)

            # bbox
            bboxs_['bbox_xywh'].append(bbox_xywh)

            # keypoints
            keypoints2d_smpl_.append(kps2d)
            keypoints3d_smpl.append(kps3d[iid])

            # smpl params
            smpl_['body_pose'].append(body_pose[iid])
            smpl_['betas'].append(betas[iid])
            smpl_['global_orient'].append(global_orient[iid])
            smpl_['transl'].append(transl[iid])

            # meta
            meta_['principal_point'].append(principal_point)
            meta_['focal_length'].append(focal_length)
            meta_['height'].append(height)
            meta_['width'].append(width)

            # contact
            contact_['smpl_vertex'].append(anno['contact_label'][iid])
            contact_['polygon_2d'].append(anno['polygon_2d_contact'][iid])
       
        size_i = min(int(size), len(image_path_))

        # meta
        human_data['meta'] = meta_

        # image path
        human_data['image_path'] = image_path_

        # bbox
        for bbox_name in bboxs_.keys():
            bbox_ = np.array(bboxs_[bbox_name]).reshape(-1, 4)
            bbox_conf = np.ones((bbox_.shape[0], 1))
            bbox_ = np.concatenate([bbox_, bbox_conf], axis=1)
            human_data[bbox_name] = bbox_

        # smpl
        for key in smpl_.keys():
            smpl_[key] = np.concatenate(
                smpl_[key], axis=0).reshape(self.smpl_shape[key])
        human_data['smpl'] = smpl_
        
        # contact
        human_data['contact'] = contact_

        # keypoints2d_smpl
        keypoints2d_smpl = np.concatenate(
            keypoints2d_smpl_, axis=0).reshape(-1, 45, 2)
        keypoints2d_smpl_conf = np.ones([keypoints2d_smpl.shape[0], 45, 1])
        keypoints2d_smpl = np.concatenate(
            [keypoints2d_smpl, keypoints2d_smpl_conf], axis=-1)
        keypoints2d_smpl, keypoints2d_smpl_mask = \
                convert_kps(keypoints2d_smpl, src='smpl_45', dst='human_data')
        human_data['keypoints2d_smpl'] = keypoints2d_smpl
        human_data['keypoints2d_smpl_mask'] = keypoints2d_smpl_mask

        # keypoints3d_smpl
        keypoints3d_smpl = np.concatenate(
            keypoints3d_smpl, axis=0).reshape(-1, 45, 3)
        keypoints3d_smpl_conf = np.ones([keypoints3d_smpl.shape[0], 45, 1])
        keypoints3d_smpl = np.concatenate(
            [keypoints3d_smpl, keypoints3d_smpl_conf], axis=-1)
        keypoints3d_smpl, keypoints3d_smpl_mask = \
                convert_kps(keypoints3d_smpl, src='smpl_45', dst='human_data')
        human_data['keypoints3d_smpl'] = keypoints3d_smpl
        human_data['keypoints3d_smpl_mask'] = keypoints3d_smpl_mask

        # misc
        human_data['misc'] = self.misc_config
        human_data['config'] = f'deco_{mode}'

        # save
        human_data.compress_keypoints_by_mask()
        os.makedirs(out_path, exist_ok=True)
        out_file = os.path.join(
            out_path,
            f'deco_{mode}_{seed}_{"{:04d}".format(size_i)}.npz')
        human_data.dump(out_file)







