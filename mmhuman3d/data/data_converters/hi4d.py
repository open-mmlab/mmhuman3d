import glob
import os
import pdb
import random
import json

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

from tools.utils.convert_contact_label import get_contact_region_label_from_smpl_vertex 
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class Hi4dConverter(BaseModeConverter):

    ACCEPTED_MODES = ['train']

    def __init__(self, modes=[], *args, **kwargs):

        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.misc_config = dict(
            bbox_source='keypoints2d_smpl',
            smpl_source='original',
            cam_param_type='perspective',
            bbox_body_scale=1.2,
            bbox_facehand_scale=1.0,
            kps3d_root_aligned=False,
            has_gender=True,
            contact_label=['smpl_vertex_correspondence', 'has_contact', 'contact_region'],
        )

        self.smpl_shape = {
            'body_pose': (-1, 69),
            'betas': (-1, 10),
            'global_orient': (-1, 3),
            'transl': (-1, 3),} 
        super(Hi4dConverter, self).__init__(modes)

    def _keypoints_to_scaled_bbox_bfh(self,
                                      keypoints,
                                      occ=None,
                                      body_scale=1.0,
                                      fh_scale=1.0,
                                      convention='smplx'):
        '''Obtain scaled bbox in xyxy format given keypoints
        Args:
            keypoints (np.ndarray): Keypoints
            scale (float): Bounding Box scale
        Returns:
            bbox_xyxy (np.ndarray): Bounding box in xyxy format
        '''
        bboxs = []

        # supported kps.shape: (1, n, k) or (n, k), k = 2 or 3
        if keypoints.ndim == 3:
            keypoints = keypoints[0]
        if keypoints.shape[-1] != 2:
            keypoints = keypoints[:, :2]

        for body_part in ['body', 'head', 'left_hand', 'right_hand']:
            if body_part == 'body':
                scale = body_scale
                kps = keypoints
            else:
                scale = fh_scale
                kp_id = get_keypoint_idxs_by_part(
                    body_part, convention=convention)
                kps = keypoints[kp_id]

            if not occ is None:
                occ_p = occ[kp_id]
                if np.sum(occ_p) / len(kp_id) >= 0.1:
                    conf = 0
                    # print(f'{body_part} occluded, occlusion: {np.sum(occ_p) / len(kp_id)}, skip')
                else:
                    # print(f'{body_part} good, {np.sum(self_occ_p + occ_p) / len(kp_id)}')
                    conf = 1
            else:
                conf = 1
            if body_part == 'body':
                conf = 1

            xmin, ymin = np.amin(kps, axis=0)
            xmax, ymax = np.amax(kps, axis=0)

            width = (xmax - xmin) * scale
            height = (ymax - ymin) * scale

            x_center = 0.5 * (xmax + xmin)
            y_center = 0.5 * (ymax + ymin)
            xmin = x_center - 0.5 * width
            xmax = x_center + 0.5 * width
            ymin = y_center - 0.5 * height
            ymax = y_center + 0.5 * height

            bbox = np.stack([xmin, ymin, xmax, ymax, conf],
                            axis=0).astype(np.float32)
            bboxs.append(bbox)

        return bboxs

    def convert_by_mode(self, dataset_path: str, out_path: str,
                    mode: str) -> dict:
        print('Converting HI4D dataset...')

        # build gendered smpl model
        gendered_smpl = {}
        for gender in ['male', 'female', 'neutral']:
            gendered_smpl[gender] = build_body_model(
                dict(
                    type='SMPL',
                    keypoint_src='smpl_45',
                    keypoint_dst='smpl_45',
                    model_path='data/body_models/smpl',
                    gender=gender,
                    num_betas=10,
                    use_pca=False,
                    batch_size=1)).to(self.device)
            
        # initialize HumanData
        smpl_ = {}
        for key in self.smpl_shape.keys():
            smpl_[key] = []
        bboxs_ = {}
        for key in ['bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh', 'rhand_bbox_xywh']:
            bboxs_[key] = []
        image_path_ = []
        keypoints2d_smpl_, keypoints3d_smpl = [], []
        meta_ = {}
        for meta_key in ['principal_point', 'focal_length', 'gender', 
                         'track_id', 'frame_id', 'seq', 'height', 'width',]:
            meta_[meta_key] = []
        contact_ = {}
        for contact_key in self.misc_config['contact_label']:
            contact_[contact_key] = []

        # parse seqs
        seqs = os.listdir(f'{dataset_path}/data')
        seqs = [s for s in seqs if '.' not in s]
        seqs = [s for s in seqs if 'output' not in s]

        seed, size = '240205', '999'
        size_i = min(int(size), len(seqs))
        random_ids = np.random.RandomState(seed=int(seed)).permutation(999999)
        used_id_num = 0

        for seq in tqdm(seqs, desc='Seqs', position=0, leave=False):
            
            # load cameras ['ids', 'intrinsics', 'extrinsics', 'dist_coeffs']
            cam_path = os.path.join(dataset_path, 'data', seq, 'cameras', 'rgb_cameras.npz')
            cam_param = dict(np.load(cam_path, allow_pickle=True))
            cids = cam_param['ids']

            # load meta
            meta_path = os.path.join(dataset_path, 'data', seq, 'meta.npz')
            meta = dict(np.load(meta_path, allow_pickle=True))

            # frame idxs
            frame_idxs = [a for a in range(meta['start'], meta['end']+1)]

            for cidx, cid in enumerate(cids):
                
                # load camera parameters
                intrinsics = cam_param['intrinsics'][cidx]
                extrinsics = np.eye(4)
                extrinsics[:3, :] = cam_param['extrinsics'][cidx]
                dist_coeffs = cam_param['dist_coeffs'][cidx]

                # build intrinsics camera
                focal_length = [intrinsics[0, 0], intrinsics[1, 1]]
                principal_point = [intrinsics[0, 2], intrinsics[1, 2]]

                # get height and width
                fid0 = frame_idxs[0]
                imgp0 = os.path.join(dataset_path, 'data', seq, 'images', str(cid), f'{fid0:06d}.jpg')
                img0 = cv2.imread(imgp0)
                height, width = img0.shape[:2]

                camera = build_cameras(
                    dict(
                        type='PerspectiveCameras',
                        convention='opencv',
                        in_ndc=False,
                        focal_length=focal_length,
                        image_size=(width, height),
                        principal_point=principal_point)).to(self.device)

                for fid in tqdm(frame_idxs, desc=f'Camera views {cidx+1}/{len(cids)}', position=1, leave=False):

                    # prepare image
                    imgp = os.path.join(dataset_path, 'data', seq, 'images', str(cid), f'{fid:06d}.jpg')
                    image_path = imgp.replace(f'{dataset_path}/', '')
                    image = cv2.imread(imgp)

                    # get height and width
                    height, width = image.shape[:2]

                    # get load smpl
                    smplp = os.path.join(dataset_path, 'data', seq, 'smpl',  f'{fid:06d}.npz')                    
                    smpl_param = dict(np.load(smplp, allow_pickle=True))

                    # load smpl params
                    betas = smpl_param['betas']
                    ped_num = betas.shape[0]

                    # load mesh segmentation
                    # segp = os.path.join(dataset_path, seq, 'seg', 'mesh_seg_mask', f'mesh-f{fid:05d}.npz')
                    # if os.path.exists(segp):
                    #     seg_param = dict(np.load(segp, allow_pickle=True))
                    
                    # write track dict
                    pids = [pid for pid in range(ped_num)]
                    track_dict = {}
                    for pid in pids:
                        track_dict[pid] = random_ids[used_id_num]
                        used_id_num += 1

                    for pid in pids:

                        track_id = track_dict[pid]
                        gender = meta['genders'][pid]

                        # get smpl output
                        body_model_param_tensor = {key: torch.tensor(
                            np.array(smpl_param[key][pid:pid+1, ...].reshape(self.smpl_shape[key])),
                                    device=self.device, dtype=torch.float32)
                                    for key in self.smpl_shape.keys()}
                        output = gendered_smpl[gender](**body_model_param_tensor)

                        smpl_joints = output['joints']
                        kps3d = smpl_joints.detach().cpu().numpy() 

                        # get pelvis world and transl
                        pelvis_world = kps3d[:, get_keypoint_idx('pelvis', 'smpl'), :]
                        transl = smpl_param['transl'][pid:pid+1, ...]
                        global_orient = smpl_param['global_orient'][pid:pid+1, ...]

                        # transform smpl to camera frame
                        global_orient_c, transl_c = transform_to_camera_frame(
                           global_orient, transl, pelvis_world, extrinsics) 

                        body_model_param_tensor['transl'] = torch.tensor(
                            np.array(transl_c.reshape(1, -1)),  device=self.device, dtype=torch.float32)
                        body_model_param_tensor['global_orient'] = torch.tensor(
                            np.array(global_orient_c.reshape(1, -1)),  device=self.device, dtype=torch.float32)
                        
                        # for key in body_model_param_tensor.keys():
                        #     print(key, body_model_param_tensor[key].shape)
                        # pdb.set_trace()
                        output = gendered_smpl[gender](**body_model_param_tensor, return_verts=True, 
                                                       return_full_pose=True, return_pose=True, 
                                                       return_shape=True, return_joints=True)

                        # project 3d keypoints to 2d
                        kps3d_c = output['joints']
                        kps2d = camera.transform_points_screen(kps3d_c)[..., :2].detach().cpu().numpy()

                        # get bbox
                        # get bbox from 2d keypoints
                        bboxs = self._keypoints_to_scaled_bbox_bfh(
                            kps2d,
                            body_scale=self.misc_config['bbox_body_scale'],
                            fh_scale=self.misc_config['bbox_facehand_scale'],
                            convention='smpl_45')
                        ## convert xyxy to xywh
                        for i, bbox_name in enumerate([
                                'bbox_xywh', 'face_bbox_xywh',
                                'lhand_bbox_xywh', 'rhand_bbox_xywh'
                        ]):
                            xmin, ymin, xmax, ymax, conf = bboxs[i]
                            bbox = np.array([
                                max(0, xmin),
                                max(0, ymin),
                                min(width, xmax),
                                min(height, ymax)
                            ])
                            bbox_xywh = self._xyxy2xywh(bbox)  # list of len 4
                            bbox_xywh.append(conf)  # (5,)
                            bboxs_[bbox_name].append(bbox_xywh)

                        # append image path
                        image_path_.append(image_path)

                        # append kps
                        keypoints2d_smpl_.append(kps2d)
                        keypoints3d_smpl.append(kps3d_c.detach().cpu().numpy().reshape(-1, 3))

                        # # append bbox
                        # bboxs_['bbox_xywh'].append(bbox_xywh)

                        # append smpl params
                        for key in self.smpl_shape.keys():
                            smpl_[key].append(body_model_param_tensor[key].detach().cpu().numpy()
                                              .reshape(self.smpl_shape[key]))
                            
                        # append meta
                        meta_['principal_point'].append(principal_point)
                        meta_['focal_length'].append(focal_length)
                        meta_['gender'].append(gender)
                        meta_['track_id'].append(track_id)
                        meta_['frame_id'].append(fid)
                        meta_['seq'].append(seq)
                        meta_['height'].append(height)
                        meta_['width'].append(width)

                        # append contact
                        if fid in meta['contact_ids']:
                            contact_['has_contact'].append(1)
                        else:
                            contact_['has_contact'].append(0)
                        contact_['smpl_vertex_correspondence'].append(smpl_param['contact'][pid])
                        # pdb.set_trace()

                        contactR = get_contact_region_label_from_smpl_vertex(
                            smpl_param['contact'][pid].reshape(-1, 1))
                        contact_['contact_region'].append(contactR)
                        
                        # test render contact
                        smplx_vert2region = np.load('tools/utils/smplx_vert2region.npy')
                        smpl_vert2region = np.load('tools/utils/smpl_vert2region.npy')

                        vertices = camera.transform_points_screen(output['vertices'])[..., :2].detach().cpu().numpy()[0]
                        image = cv2.imread(imgp)
                        for vi, v in enumerate(smpl_vert2region):
                            if np.sum(smpl_vert2region[vi] * contactR) > 0.5:
                                cv2.circle(image, (int(vertices[vi][0]), int(vertices[vi][1])), 1, (255,0,0), 2)
                            else:
                                cv2.circle(image, (int(vertices[vi][0]), int(vertices[vi][1])), 1, (0,0,255), 2)
                        cv2.imwrite(f'{out_path}/{os.path.basename(imgp)}', image)
                        continue
                    continue
                # pdb.set_trace()




                        # pdb.set_trace()

                        # write kps2d on image
                    #     for kp in kps2d[0]:
                    #         if  0 < kp[0] < width and 0 < kp[1] < height: 
                    #             cv2.circle(image, (int(kp[0]), int(kp[1])), 3, (0,0,255), 2)
                    # cv2.imwrite(f'{out_path}/{os.path.basename(imgp)}', image)
                    # pdb.set_trace()
        
        # meta
        human_data = HumanData()
        human_data['meta'] = meta_

        # contact
        human_data['contact'] = contact_

        # image path
        human_data['image_path'] = image_path_

        # bbox
        for bbox_name in bboxs_.keys():
            bbox_ = np.array(bboxs_[bbox_name]).reshape(-1, 5)
            # bbox_conf = np.ones((bbox_.shape[0], 1))
            # bbox_ = np.concatenate([bbox_, bbox_conf], axis=1)
            human_data[bbox_name] = bbox_

        # smpl
        for key in smpl_.keys():
            smpl_[key] = np.concatenate(
                smpl_[key], axis=0).reshape(self.smpl_shape[key])    
        human_data['smpl'] = smpl_

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
        human_data['config'] = f'hi4d_{mode}'

        # save
        human_data.compress_keypoints_by_mask()
        os.makedirs(out_path, exist_ok=True)
        out_file = os.path.join(
            out_path,
            f'hi4d_{mode}_{seed}_{"{:03d}".format(size_i)}.npz')
        human_data.dump(out_file)