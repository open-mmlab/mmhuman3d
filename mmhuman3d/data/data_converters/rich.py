import os
import pickle
import glob
import random
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

import logging
import json
import mmcv
import cv2
import torch
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.data.data_structures.multi_human_data import MultiHumanData
from mmhuman3d.models.body_models.builder import build_body_model
from mmhuman3d.core.visualization.visualize_smpl import render_smpl
from mmhuman3d.core.visualization.visualize_keypoints2d import visualize_kp2d
from mmhuman3d.utils.demo_utils import xyxy2xywh
from mmhuman3d.models.body_models.utils import batch_transform_to_camera_frame
from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_idx,
    get_keypoint_idxs_by_part,
)
from tools.utils.convert_contact_label import get_contact_label_from_smpl_vertex
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS

import pdb

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:,0] / cam_coord[:,2] * f[0] + c[0]
    y = cam_coord[:,1] / cam_coord[:,2] * f[1] + c[1]
    z = cam_coord[:,2]
    return np.stack((x,y,z),1)


def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord


@DATA_CONVERTERS.register_module()
class RichConverter(BaseModeConverter):
    ACCEPTED_MODES = ['train', 'test', 'val']
    def __init__(self, modes: List = []) -> None:

        super(RichConverter, self).__init__(modes)

        self.device = torch.device('cuda:0')
        self.misc_config = dict(
            bbox_body_scale=1.2,
            bbox_facehand_scale=1.0,
            bbox_source='keypoints2d_original',
            flat_hand_mean=False,
            cam_param_type='prespective',
            cam_param_source='original',
            smplx_source='original',
            contact_label=['smpl_vertex', 'part_segmentation'],
            part_segmentation=['left_foot', 'right_foot'],
        )

        self.smplx_shape = {
            'betas': (-1, 10),
            'transl': (-1, 3),
            'global_orient': (-1, 3),
            'body_pose': (-1, 21, 3),
            'left_hand_pose': (-1, 15, 3),
            'right_hand_pose': (-1, 15, 3),
            'leye_pose': (-1, 3),
            'reye_pose': (-1, 3),
            'jaw_pose': (-1, 3),
            'expression': (-1, 10)
        }


    def extract_cam_param_xml(self, xml_path='', dtype=torch.float32):
    
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_path)

        extrinsics_mat = [float(s) for s in tree.find('./CameraMatrix/data').text.split()]
        intrinsics_mat = [float(s) for s in tree.find('./Intrinsics/data').text.split()]
        distortion_vec = [float(s) for s in tree.find('./Distortion/data').text.split()]

        focal_length_x = intrinsics_mat[0]
        focal_length_y = intrinsics_mat[4]
        center = torch.tensor([[intrinsics_mat[2], intrinsics_mat[5]]], dtype=dtype)
        
        rotation = torch.tensor([[extrinsics_mat[0], extrinsics_mat[1], extrinsics_mat[2]], 
                                [extrinsics_mat[4], extrinsics_mat[5], extrinsics_mat[6]], 
                                [extrinsics_mat[8], extrinsics_mat[9], extrinsics_mat[10]]], dtype=dtype)

        translation = torch.tensor([[extrinsics_mat[3], extrinsics_mat[7], extrinsics_mat[11]]], dtype=dtype)

        # t = -Rc --> c = -R^Tt
        cam_center = [  -extrinsics_mat[0]*extrinsics_mat[3] - extrinsics_mat[4]*extrinsics_mat[7] - extrinsics_mat[8]*extrinsics_mat[11],
                        -extrinsics_mat[1]*extrinsics_mat[3] - extrinsics_mat[5]*extrinsics_mat[7] - extrinsics_mat[9]*extrinsics_mat[11], 
                        -extrinsics_mat[2]*extrinsics_mat[3] - extrinsics_mat[6]*extrinsics_mat[7] - extrinsics_mat[10]*extrinsics_mat[11]]

        principal_point = center.detach().cpu().numpy()

        k1 = torch.tensor([distortion_vec[0]], dtype=dtype)
        k2 = torch.tensor([distortion_vec[1]], dtype=dtype)

        focal_length = [focal_length_x, focal_length_y]
        R = rotation.detach().cpu().numpy()
        T = translation.detach().cpu().numpy()

        return focal_length, principal_point, R, T, cam_center, k1, k2
    
    def _revert_smplx_hands_pca(self, param_dict, num_pca_comps, 
        gender='neutral', gendered_model=None):
        # egobody 12
        hl_pca = param_dict['left_hand_pose']
        hr_pca = param_dict['right_hand_pose']

        if gendered_model != None:
            smplx_model = gendered_model
        else:
            smplx_model = dict(np.load(f'data/body_models/smplx/SMPLX_{gender.upper()}.npz', allow_pickle=True))

        hl = smplx_model['hands_componentsl'] # [45, 45]
        hr = smplx_model['hands_componentsr'] # 45, 45

        hl_pca = np.concatenate((hl_pca, np.zeros((len(hl_pca), 45 - num_pca_comps))), axis=1) # [1,45]
        hr_pca = np.concatenate((hr_pca, np.zeros((len(hr_pca), 45 - num_pca_comps))), axis=1)

        hl_reverted = np.einsum('ij, jk -> ik', hl_pca, hl).astype(np.float32) #[1,45] * [45, 45] = [1, 45]
        hr_reverted = np.einsum('ij, jk -> ik', hr_pca, hr).astype(np.float32)

        param_dict['left_hand_pose'] = hl_reverted
        param_dict['right_hand_pose'] = hr_reverted

        return param_dict

    
    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:
        print('Converting RICH dataset...')

        # use HumanData to store all data
        human_data = HumanData()

        # parse seqs
        seqs = os.listdir(os.path.join(dataset_path, mode))

        # init seed and size
        seed, size = '240126', '99'
        size_i = min(int(size), len(seqs))
        random.seed(int(seed))
        np.set_printoptions(suppress=True)
        random_ids = np.random.RandomState(seed=int(seed)).permutation(999999)
        used_id_num = 0
        
        seqs = seqs[:size_i]

        # build body model
        gendered_smplx = {}
        for gender in ['male', 'female', 'neutral']:
            gendered_smplx[gender] = build_body_model(
                dict(
                    type='SMPLX',
                    keypoint_src='smplx',
                    keypoint_dst='smplx',
                    model_path='data/body_models/smplx',
                    gender=gender,
                    num_betas=10,
                    use_face_contour=True,
                    flat_hand_mean=self.misc_config['flat_hand_mean'],
                    use_pca=False,
                    batch_size=1)).to(self.device)
            
        # build body model for pca revert
        gendered_smplx_rev = {} 
        for gender in ['male', 'female', 'neutral']:
            gendered_smplx_rev[gender] = dict(np.load(f'data/body_models/smplx/SMPLX_{gender.upper()}.npz', allow_pickle=True))

        # load gender mapping
        gender_mapping = json.load(open(f'{dataset_path}/gender.json','r'))

        # initialize output for human_data
        smplx_ = {}
        for key in self.smplx_shape.keys():
            smplx_[key] = []
        keypoints2d_, keypoints3d_, kps2d_orig_ = [], [], []
        bboxs_ = {}
        for bbox_name in [
                'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
                'rhand_bbox_xywh'
        ]:
            bboxs_[bbox_name] = []
        meta_ = {}
        for meta_name in ['principal_point', 'focal_length', 'height', 'width', 'RT',
                          'sequence_name', 'track_id', 'gender']:
            meta_[meta_name] = []
        image_path_ = []
        contact_ = {}
        for contact_key in self.misc_config['contact_label']:
            contact_[contact_key] = []

        # get sub ids from seq
        seqs_ids = []
        for seq in seqs:
            try:
                scene_name, sub_id, _ = seq.split('_')
                
                seqs_ids.append([seq, sub_id])
            except:
                # Pavallion_003_018_tossball
                scene_name, *sub_id, _ = seq.split('_')
                for sid in sub_id:
                    seqs_ids.append([seq, sid])

        for sid, [seq, sub_id] in enumerate(tqdm(seqs_ids, desc='seqs', position=0, leave=False)):
            
            # get camera ids
            cam_ids = os.listdir(os.path.join(dataset_path, mode, seq))
            cam_ids = [cam_id for cam_id in cam_ids if cam_id.startswith('cam')]
            cam_params = {}

            # get gender
            gender = gender_mapping[f'{int(sub_id)}']

            # load annotations
            # anno_path = os.path.join(dataset_path, mode, seq, cam_id, 'annotation.pkl')
            anno_ps = sorted(glob.glob(os.path.join(dataset_path, f'{mode}_body', seq, '*', f'{int(sub_id):03d}.pkl')))

            # load width and height
            fid = os.path.basename(os.path.dirname(anno_ps[0]))
            # print(anno_ps[0], fid)
            cid = int(cam_ids[0][-2:])
            image_sample = cv2.imread(os.path.join(dataset_path, mode, seq, cam_ids[0], f'{int(fid):05d}_{int(cid):02d}.jpeg'))
            height, width = image_sample.shape[:2]

            # load cam param
            camera_dict = {}
            for cam_id in cam_ids:
                cam_param = {}
                cid = int(cam_id[-2:])
                cam_param_path = f'{dataset_path}/scan_calibration/{seq.rsplit("_")[0]}/calibration/{int(cid):03d}.xml'
                if not os.path.exists(cam_param_path):
                    del cam_ids[cam_ids.index(cam_id)]
                    continue
                focal_length, principal_point, R, T, cam_center, k1, k2 = \
                        self.extract_cam_param_xml(cam_param_path)
                cam_param['focal_length'] = focal_length
                cam_param['principal_point'] = principal_point
                cam_param['R'] = np.array(R)
                cam_param['T'] = np.array(T)
                cam_params[cam_id]=cam_param

                # build cameras
                camera_dict[cam_id] = build_cameras(
                        dict(
                            type='PerspectiveCameras',
                            convention='opencv',
                            in_ndc=False,
                            focal_length=focal_length,
                            image_size=(width, height),
                            principal_point=principal_point)).to(self.device)
                
            # get track id dict
            track_id_dict = {}
            for cam_id in cam_ids:
                track_id_dict[cam_id] = random_ids[used_id_num]
                used_id_num += 1

            # parpare smplx params and get fid
            fids = []
            smplx_params = {}
            contacts_dict = {}
            for key in self.misc_config['contact_label']:
                contacts_dict[key] = []
            for key in self.smplx_shape.keys():
                smplx_params[key] = []
            for anno_p in tqdm(anno_ps, desc='Preparing annos', position=1, leave=False):

                # load annotation
                anno = pickle.load(open(anno_p, 'rb'))

                # load contact label
                contact_p = anno_p.replace(f'{mode}_body', f'{mode}_hsc')
                if not os.path.exists(contact_p):
                    print(contact_p, 'not exists')
                    continue

                # load smplx params
                smplx_param = {}
                for key in self.smplx_shape.keys():
                    smplx_param[key] = anno[key]

                smplx_param = self._revert_smplx_hands_pca(
                    smplx_param, num_pca_comps=12, gender=gender, gendered_model=gendered_smplx_rev[gender])
                
                # append to smplx_params
                for key in self.smplx_shape.keys():
                    smplx_params[key].append(smplx_param[key])

                fid = os.path.basename(os.path.dirname(anno_p))
                fids.append(fid)

                # load contact
                contact = pickle.load(open(contact_p, 'rb'))
                contact_label_dict = get_contact_label_from_smpl_vertex(contact['contact'].reshape(-1, 1))

                contact_temp = []
                for key in self.misc_config['part_segmentation']:
                    contact_temp.append(contact_label_dict[key])
                contacts_dict['part_segmentation'].append(contact_temp)
                contacts_dict['smpl_vertex'].append(contact['contact'].reshape(-1, 1))

            # prepare smplx params
            smplx_param_tensor = {}
            for key in self.smplx_shape.keys():
                smplx_params[key] = np.concatenate(smplx_params[key], axis=0)
                smplx_param_tensor[key] = torch.tensor(smplx_params[key]).float().to(self.device)

            output = gendered_smplx[gender](**smplx_param_tensor)

            kps3d = output['joints'].detach().cpu().numpy()
            pelvis_world = kps3d[:, get_keypoint_idx('pelvis', 'smplx'), :]

            for cam_id in cam_ids:
                
                cid = int(cam_id[-2:])

                # get RT and transform to camera space
                R = cam_params[cam_id]['R']
                T = cam_params[cam_id]['T']
                extrinsics = np.eye(4)
                extrinsics[:3, :3] = R
                extrinsics[:3, 3] = T

                global_orient, transl = batch_transform_to_camera_frame(
                    global_orient=smplx_params['global_orient'].reshape(-1, 3),
                    transl=smplx_params['transl'].reshape(-1, 3),
                    pelvis=pelvis_world.reshape(-1, 3),
                    extrinsic=extrinsics)

                # update smplx params
                smplx_params_copy = smplx_params.copy()
                smplx_params_copy['global_orient'] = global_orient.reshape(-1, 3)
                smplx_params_copy['transl'] = transl.reshape(-1, 3)

                # prepare tensor
                smplx_param_tensor = {}
                for key in self.smplx_shape.keys():
                    smplx_param_tensor[key] = torch.tensor(smplx_params_copy[key]
                                                            .reshape(self.smplx_shape[key])).float().to(self.device)

                output = gendered_smplx[gender](**smplx_param_tensor)
                kps3d_c = output['joints']

                # get kps2d
                keypoints_2d_xyd = camera_dict[cam_id].transform_points_screen(kps3d_c)
                kps2d = keypoints_2d_xyd[..., :2].detach().cpu().numpy()
                kps3d_c = kps3d_c.detach().cpu().numpy()

                # pdb.set_trace()
                # img = cv2.imread(imgp)
                # for jid, j in enumerate(kps2d[0]):
                #     cv2.circle(img, (int(j[0]), int(j[1])), 5, (0, 0, 255), -1)
                #         # cv2.putText(img, str(jid), (int(j[0]), int(j[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                #     # for bbox in bbo_:
                #     #     cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0, 255, 0), 2) 

                #     os.makedirs(f'{out_path}', exist_ok=True)
                # cv2.imwrite(f'{out_path}/{os.path.basename(seq)}_{cid}_{fid}.jpg', img)

                for fidx, fid in enumerate(fids):

                    imgp = os.path.join(dataset_path, mode, seq, cam_id, f'{int(fid):05d}_{int(cid):02d}.jpeg')
                    if not os.path.exists(imgp):
                        print(imgp, 'not exists')
                        continue
                    image_path = imgp.replace(f'{dataset_path}/', '')
                    # append image path
                    image_path_.append(image_path)

                    # get bbox from 2d keypoints
                    bboxs = self._keypoints_to_scaled_bbox_bfh(
                        kps2d[fidx],
                        body_scale=self.misc_config['bbox_body_scale'],
                        fh_scale=self.misc_config['bbox_facehand_scale'])
                    for i, bbox_name in enumerate([
                            'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
                            'rhand_bbox_xywh'
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

                    # append keypoints
                    keypoints2d_.append(kps2d[fidx].reshape(1, -1, 2))
                    keypoints3d_.append(kps3d_c[fidx].reshape(1, -1, 3))

                    # append smplx params
                    for key in self.smplx_shape.keys():
                        smplx_[key].append(smplx_params_copy[key][fidx])

                    # append meta
                    meta_['principal_point'].append(cam_params[cam_id]['principal_point'][0])
                    meta_['focal_length'].append(cam_params[cam_id]['focal_length'])
                    meta_['height'].append(height)
                    meta_['width'].append(width)
                    meta_['RT'].append(extrinsics)
                    meta_['sequence_name'].append(f'{seq}_{cid}_{sub_id}')
                    meta_['track_id'].append(track_id_dict[cam_id])
                    meta_['gender'].append(gender)

                    # append contact
                    for key in self.misc_config['contact_label']:
                        contact_[key].append(contacts_dict[key][fidx])
                            
                    # pdb.set_trace()        

        # get size
        size_i = min(int(size), len(seqs))

        # save keypoints 2d smplx
        keypoints2d = np.concatenate(keypoints2d_, axis=0)
        keypoints2d_conf = np.ones([keypoints2d.shape[0], 144, 1])
        keypoints2d = np.concatenate([keypoints2d, keypoints2d_conf], axis=-1)
        keypoints2d, keypoints2d_mask = convert_kps(
            keypoints2d, src='smplx', dst='human_data')
        human_data['keypoints2d_smplx'] = keypoints2d
        human_data['keypoints2d_smplx_mask'] = keypoints2d_mask

        # save keypoints 3d smplx
        keypoints3d = np.concatenate(keypoints3d_, axis=0)
        keypoints3d_conf = np.ones([keypoints3d.shape[0], 144, 1])
        keypoints3d = np.concatenate([keypoints3d, keypoints3d_conf], axis=-1)
        keypoints3d, keypoints3d_mask = convert_kps(
            keypoints3d, src='smplx', dst='human_data')
        human_data['keypoints3d_smplx'] = keypoints3d
        human_data['keypoints3d_smplx_mask'] = keypoints3d_mask

        # pdb.set_trace()
        # save bbox
        for bbox_name in [
                'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
                'rhand_bbox_xywh'
        ]:
            bbox_xywh_ = np.array(bboxs_[bbox_name]).reshape((-1, 5))
            human_data[bbox_name] = bbox_xywh_

        # save smplx
        for key in smplx_.keys():
            smplx_[key] = np.concatenate(
                smplx_[key], axis=0).reshape(self.smplx_shape[key])
        human_data['smplx'] = smplx_

        # save image path
        human_data['image_path'] = image_path_

        # save contact
        human_data['contact'] = contact_

        # save meta and misc
        human_data['config'] = 'rich'
        human_data['misc'] = self.misc_config
        human_data['meta'] = meta_

        os.makedirs(out_path, exist_ok=True)
        out_file = os.path.join(
            # out_path, f'moyo_{self.misc_config["flat_hand_mean"]}.npz')
            out_path, f'rich_{mode}_{seed}_{"{:02d}".format(size_i)}.npz')
        human_data.dump(out_file)








        return


        for seq_names in seq_names_:
            # if os.path.exists(os.path.join(out_path, 'rich', f'rich_no_w_{seq_names}_{mode}.npz')):
            #     continue
            seq_names = [seq_names]
            human_data = HumanData()
            
            image_path_, bbox_xywh_, keypoints2d_smplx_, keypoints3d_smplx_= [], [], [], []
            face_bbox_xywh_, lhand_bbox_xywh_, rhand_bbox_xywh_ = [], [], []
            
            # smplx
            smplx = {}
            smplx_key = [
                'betas', 'transl', 'global_orient', 'body_pose', 'left_hand_pose', 
                'right_hand_pose', 'leye_pose', 'reye_pose', 'jaw_pose', 'expression'
                ]
            for key in smplx_key:
                smplx[key] = []

            meta = {}
            meta['focal_length'] = []
            meta['principal_point'] = []
            meta['height'] = []
            meta['width'] = []
            meta['gender'] = []
            meta['R'] = []
            meta['T'] = []
            

            
            gender_mapping = json.load(open(dataset_path+'/resource/gender.json','r'))
            
            LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
            DATE_FORMAT = '%Y%m%d %H:%M:%S'
            logging.basicConfig(level=logging.INFO, filename='myLog.log', filemode='w', format=LOGGING_FORMAT, datefmt=DATE_FORMAT)
            
            
            if seq_names[0].count('_') == 3:
                # Pavallion_003_018_tossball
                seq_names = seq_names * 2
                
            for i, seq_name in enumerate(seq_names):
                

                try:
                    scene_name, *sub_id, _ = seq_name.split('_')
                    gender = gender_mapping[f'{int(sub_id)}']
                except:
                    # Pavallion_003_018_tossball
                    scene_name, *sub_id, _ = seq_name.split('_')
                
                    gender = gender_mapping[f'{int(sub_id[i])}']
                    
                
                
                smplx_params_folder = os.path.join(data_folder, seq_name)
                frame_ids = sorted(os.listdir(smplx_params_folder))
                cam_views = os.listdir(os.path.join(image_folder, seq_name))
                
                cam_params = {}

                for cam_view in cam_views:
                    cam_id = cam_view.split('_')[-1]
                    cam_param = {}
                    try:
                        cam_param_path = os.path.join(f'{dataset_path}/scan_calibration', scene_name, 'calibration', f'{int(cam_id):03d}.xml')
                        focal_length_x, focal_length_y, center, R, T, cam_center, k1, k2 = \
                            extract_cam_param_xml(cam_param_path)
                        cam_param['focal_length_x'] = focal_length_x
                        cam_param['focal_length_y'] = focal_length_y
                        cam_param['center'] = np.array(center)[0]
                        cam_param['R'] = np.array(R)
                        cam_param['T'] = np.array(T)
                        cam_params[cam_id]=cam_param
                    except:
                        logging.error('Catch an exception.', exc_info=True)
                        continue

                for frame_id in frame_ids:
                    smplx_params_fn = \
                        os.path.join(smplx_params_folder, frame_id, f'{sub_id[i]}.pkl')
                    body_params = pickle.load(open(smplx_params_fn,'rb'))
                    body_params = self._revert_smplx_hands_pca(body_params, num_pca_comps=12, gender=gender)
                    
                    betas = body_params['betas'].reshape(10)
                    global_orient = body_params['global_orient'].reshape(3)

                    transl = body_params['transl'].reshape(3)
                    # l_hand  = body_params['left_hand_pose']
                    # body_params['left_hand_pose'] = body_params['right_hand_pose']
                    # body_params['right_hand_pose'] = l_hand
                    left_hand_pose = body_params['left_hand_pose'].reshape(15,3) 
                    right_hand_pose = body_params['right_hand_pose'].reshape(15,3)
                    jaw_pose = body_params['jaw_pose'].reshape(3)
                    leye_pose = body_params['leye_pose'].reshape(3)
                    reye_pose = body_params['reye_pose'].reshape(3)
                    expression = body_params['expression'].reshape(10)
                    body_pose = body_params['body_pose'].reshape(21, 3)
                    # pose_embedding = body_params['pose_embedding'].reshape(32)
                    body_params = {k: torch.from_numpy(v) for k, v in body_params.items()}
                    
                    if gender == 'male':
                        self.smplx = self.smplx_male
                        # smplx_res = self.smplx_male(**body_params)
                    elif gender == 'female':
                        self.smplx = self.smplx_female
                        # smplx_res = self.smplx_female(**body_params)
                    
                    smplx_res = self.smplx(**body_params)
                    

                    keypoints3d_smplx_world=smplx_res['joints'].detach().cpu().numpy()[0]
                    vertx_smplx_world=smplx_res['vertices'].detach().cpu().numpy()[0]
                    # transform to camera space

                    for cam_view in cam_views:
                        cam_id = cam_view.split('_')[-1]
                        image_path = os.path.join(seq_name, cam_view, f"{frame_id}_{cam_id}.jpg")
                        print(image_path)
                        if not os.path.exists(os.path.join(image_folder, image_path)):
                            continue
                        
                        try:
                            # cam_param_path = os.path.join(f'{dataset_path}/scan_calibration', scene_name, 'calibration', f'{int(cam_id):03d}.xml')
                            # focal_length_x, focal_length_y, center, R, T, cam_center, k1, k2 = \
                            #     extract_cam_param_xml(cam_param_path)
                            focal = np.array([
                                cam_params[cam_id]['focal_length_x'], 
                                cam_params[cam_id]['focal_length_y']
                                ])
                            princpt = cam_params[cam_id]['center']
                            R = cam_params[cam_id]['R']
                            T = cam_params[cam_id]['T']
                            # camera
                            K =  np.array(
                                [
                                    [focal[0], 0, princpt[0] ],
                                    [0, focal[1], princpt[1] ],
                                    [0, 0, 1 ],
                                ]) 
                            height, width = cv2.imread(os.path.join(image_folder, image_path)).shape[:2]
                            # height = 3008
                            # width = 4112
                        except:
                            logging.error('Catch an exception.', exc_info=True)
                            # mmcv.utils.print_log()
                            continue

                        # # apply R to global rotation
                        extrinsic = np.zeros([4,4])
                        extrinsic[3,3] = 1.0
                        extrinsic[:3,:3] = R
                        extrinsic[:3,3] = T
                        # smplx_res = self.smplx_female(**body_params)
                        global_orient_, transl_ = transform_to_camera_frame(
                            global_orient=global_orient,
                            transl=transl,
                            pelvis=keypoints3d_smplx_world[0],
                            extrinsic=extrinsic)
                    
                        # global_orient_ = aa_to_rotmat(global_orient)
                        # global_orient_ = rotmat_to_aa(np.dot(R,global_orient_))
                        # global_orient_ = torch.from_numpy(global_orient_)
                        # for vis
                        
                        # global_orient_lh = aa_to_rotmat(left_hand_pose[0])
                        # global_orient_lh = rotmat_to_aa(np.dot(R,global_orient_lh))
                        # left_hand_pose[0] = torch.from_numpy(global_orient_lh)
                        # global_orient_lr = aa_to_rotmat(right_hand_pose[0])
                        # global_orient_lr = rotmat_to_aa(np.dot(R,global_orient_lr))
                        # right_hand_pose[0] = torch.from_numpy(global_orient_lr)
                        body_params_ = body_params.copy()
                        body_params_['global_orient'] = torch.tensor(global_orient_).view(1, 3)
                        body_params_['transl'] = torch.tensor(transl_).view(1, 3)
                        smplx_res = self.smplx(**body_params_)
                        keypoints3d_smplx_cam=smplx_res['joints'].detach().cpu().numpy()[0]
                        keypoints3d_smplx_img = cam2pixel(keypoints3d_smplx_cam, focal, princpt)
                        keypoints2d_smplx = keypoints3d_smplx_img[:,:2]
                        bbox_xyxy_smplx = keypoints_to_scaled_bbox_bfh(
                        keypoints2d_smplx, vertices=smplx_res['vertices'].detach().cpu().numpy()[0], body_scale=1.2, fh_scale=1.2, convention='smplx')
                            
                        vis_smpl = False
                        if vis_smpl:
                            body_params_ = body_params.copy()
                            body_params_['global_orient'] = torch.tensor(global_orient_).view(1, 3)
                            body_params_['transl'] = torch.tensor(transl).view(1, 3)
                            smplx_res = self.smplx_female(**body_params_)
                            keypoints3d_smplx_cam=smplx_res['joints'].detach().cpu().numpy()[0]
                            # smplx_root_cam = keypoints3d_smplx_cam[0]
                        
                            # T_ = np.dot(np.array(R), smplx_root_cam) - smplx_root_cam  + np.array(T)
                            # keypoints3d_smplx_cam = keypoints3d_smplx_cam + T_
                            keypoints3d_smplx_img = cam2pixel(keypoints3d_smplx_cam, focal, princpt)
                            keypoints2d_smplx = keypoints3d_smplx_img[:,:2]
                            bbox_xyxy_smplx = keypoints_to_scaled_bbox_bfh(
                            keypoints2d_smplx, vertices=smplx_res['vertices'].detach().cpu().numpy()[0], body_scale=1, fh_scale=1, convention='smplx')
                            bbox_xywh_smplx = xyxy2xywh(np.array(bbox_xyxy_smplx))
                            
                            vertx_smplx_world=smplx_res['vertices'].detach().cpu().numpy()[0]
                            # vertx_smplx_cam = vertx_smplx_world + np.array(T_)
                            
                            image = cv2.imread(os.path.join(image_folder, image_path))
                            img = visualize_kp2d(keypoints2d_smplx[None], output_path='.', image_array=image.copy()[None], data_source='smplx',overwrite=True)
                            img = mmcv.imshow_bboxes(img[0].copy(),np.array(bbox_xyxy_smplx), show=False)
                            cv2.imwrite('test_smplx.png',img)
                            
                            render_smpl(
                                batch_size=1,
                                verts = vertx_smplx_world[None],
                                # expression=expression,
                                body_model=self.smplx_female,
                                output_path='.',
                                K=K,
                                R=None,
                                T=None,
                                image_array=img,
                                # end=10,
                                in_ndc=False,
                                convention='opencv',
                                projection='perspective',        
                                overwrite=True,
                                no_grad=True,
                                return_tensor=False,
                                # img_format=None,
                                device='cuda',
                                resolution=img.shape[:2],
                                render_choice='lq',          
                            )    
                                                
                        # world to cam
                        # keypoints3d_smplx_cam = np.dot(R, keypoints3d_smplx_world.T).T + np.array(T)
                        # vertx_smplx_cam = np.dot(R, vertx_smplx_world.T).T + np.array(T)
                        
                        # cam to img
                        # keypoints3d_smplx_img = cam2pixel(keypoints3d_smplx_cam, focal, princpt)
                        # keypoints2d_smplx = keypoints3d_smplx_img[:,:2]
                        # bbox_xyxy_smplx = keypoints_to_scaled_bbox_bfh(
                        #     keypoints2d_smplx, body_scale=1.2, fh_scale=1.2, convention='smplx')
                        bbox_xywh_smplx = xyxy2xywh(np.array(bbox_xyxy_smplx))
                        
                        
                        
                        image_path_.append(image_path)
                        bbox_xywh_.append(bbox_xywh_smplx[0])
                        face_bbox_xywh_.append(bbox_xywh_smplx[1]) 
                        lhand_bbox_xywh_.append(bbox_xywh_smplx[2])
                        rhand_bbox_xywh_.append(bbox_xywh_smplx[3])

                        keypoints2d_smplx_.append(keypoints2d_smplx)
                        keypoints3d_smplx_.append(keypoints3d_smplx_cam)
                        # keypoints2d_smpl_.append(keypoints2d_smpl)
                        # keypoints3d_smpl_.append(keypoints3d_smpl_cam)            
                        # keypoints2d_ori_.append(keypoints2d_ori)
                        # keypoints3d_ori_.append(keypoints3d_ori_cam)
                                        
                        # smplx
                        smplx['betas'].append(np.array(betas).reshape(-1))
                        smplx['global_orient'].append(np.array(global_orient_).reshape(-1))
                        smplx['body_pose'].append(np.array(body_pose).reshape(21,3))
                        smplx['transl'].append(np.array(transl_).reshape(-1))
                        smplx['left_hand_pose'].append(np.array(left_hand_pose).reshape(15,3))
                        smplx['right_hand_pose'].append(np.array(right_hand_pose).reshape(15,3))
                        smplx['leye_pose'].append(np.array(leye_pose).reshape(-1))
                        smplx['reye_pose'].append(np.array(reye_pose).reshape(-1))
                        smplx['jaw_pose'].append(np.array(jaw_pose).reshape(-1))
                        smplx['expression'].append(np.array(expression).reshape(-1))                   
                        
                        # meta
                        meta['height'].append(height)
                        meta['width'].append(width)
                        meta['focal_length'].append(focal)
                        meta['principal_point'].append(princpt)
                        meta['gender'].append(gender)
                        meta['R'].append(R)
                        meta['T'].append(T)                  
                        
                        if False:
                            from mmhuman3d.core.conventions.cameras.convert_convention import convert_world_view
                            R_, T_ = convert_world_view(R[None], T)
                            image = cv2.imread(os.path.join(image_folder, image_path))
                            img = visualize_kp2d(keypoints2d_smplx[None], output_path='.', image_array=image.copy()[None], data_source='smplx',overwrite=True)
                            img = mmcv.imshow_bboxes(img[0].copy(),np.array(bbox_xyxy_smplx), show=False)
                            cv2.imwrite('test_smplx.png',img)
                            pose = np.concatenate([np.array(body_params['global_orient']).reshape(1,-1), body_pose.reshape(1,-1), left_hand_pose.reshape(1,-1), right_hand_pose.reshape(1,-1), leye_pose.reshape(1,-1), reye_pose.reshape(1,-1), jaw_pose.reshape(1,-1)],-1)
                            img = cv2.imread(os.path.join(image_folder, image_path))
                            from mmhuman3d.core.conventions.cameras.convert_convention import convert_world_view
                            R_, T_ = convert_world_view(R[None], T)
                            render_smpl(
                                batch_size=1,
                                poses=pose,
                                betas=betas,
                                transl=transl,
                                # expression=expression,
                                body_model=self.smplx_female,
                                output_path='.',
                                K=K,
                                R=R_,
                                T=T_,
                                image_array=img,
                                # end=10,
                                in_ndc=False,
                                convention='opencv',
                                projection='perspective',        
                                overwrite=True,
                                no_grad=True,
                                return_tensor=False,
                                # img_format=None,
                                device='cuda',
                                resolution=img.shape[:2],
                                render_choice='hq',
                                kp3d=keypoints3d_smplx_cam,
                                plot_kps=True,          
                            )                          
                            # render_smpl(
                            #     batch_size=1,
                            #     verts = vertx_smplx_cam[None],
                            #     # expression=expression,
                            #     body_model=self.smplx_female,
                            #     output_path='.',
                            #     K=K,
                            #     R=R_,
                            #     T=T_,
                            #     image_array=img,
                            #     # end=10,
                            #     in_ndc=False,
                            #     convention='opencv',
                            #     projection='perspective',        
                            #     overwrite=True,
                            #     no_grad=True,
                            #     return_tensor=False,
                            #     # img_format=None,
                            #     device='cuda',
                            #     resolution=img.shape[:2],
                            #     render_choice='lq',          
                            # )    
                            # img = cv2.imread(os.path.join(image_folder, image_path))
                            render_smpl(
                                batch_size=1,
                                poses=pose,
                                betas=betas,
                                transl=transl,
                                # expression=expression,
                                body_model=self.smplx_female,
                                output_path='.',
                                K=K,
                                R=R_,
                                T=T_,
                                image_array=img,
                                # end=10,
                                in_ndc=False,
                                convention='opencv',
                                projection='perspective',        
                                overwrite=True,
                                no_grad=True,
                                return_tensor=False,
                                # img_format=None,
                                device='cuda',
                                resolution=img.shape[:2],
                                render_choice='lq',          
                            )                            
                            cam = CalibratedCamera(calib_path=cam_param_path)
                            j_2D = cam(smplx_res['joints']).detach().cpu().numpy()
                            image_path = os.path.join(image_folder, seq_name, cam_view, f"{frame_id}_{cam_id}.jpg")
                            img = cv2.imread(image_path)
                            for j in j_2D[0][:144]:
                                cv2.circle(img, (int(j[0]), int(j[1])), 6, (255, 0, 255), thickness=-1)
                            cv2.imwrite(f'tmp_{frame_id}.jpg',img)


            # smplx
            for k, v in smplx.items():
                smplx[k] = np.array(v)
            human_data['smplx'] = smplx
            
            # kp2d_smplx
            keypoints2d_smplx = np.array(keypoints2d_smplx_).reshape(-1, 144, 2)
            keypoints2d_smplx_conf = np.ones([keypoints2d_smplx.shape[0], 144, 1])
            keypoints2d_smplx = np.concatenate([keypoints2d_smplx, keypoints2d_smplx_conf], axis=-1)
            keypoints2d_smplx, keypoints2d_smplx_mask = \
                convert_kps(keypoints2d_smplx, src='smplx', dst='human_data')
            human_data['keypoints2d_smplx'] = keypoints2d_smplx
            human_data['keypoints2d_smplx_mask'] = keypoints2d_smplx_mask
            
            # kp3d_smplx
            keypoints3d_smplx = np.array(keypoints3d_smplx_).reshape(-1, 144, 3)
            keypoints3d_smplx_conf = np.ones([keypoints3d_smplx.shape[0], 144, 1])
            keypoints3d_smplx = np.concatenate([keypoints3d_smplx, keypoints3d_smplx_conf], axis=-1)
            keypoints3d_smplx, keypoints3d_smplx_mask = \
                    convert_kps(keypoints3d_smplx, src='smplx', dst='human_data')
            human_data['keypoints3d_smplx'] = keypoints3d_smplx
            human_data['keypoints3d_smplx_mask'] = keypoints3d_smplx_mask     

            human_data['image_path'] = image_path_
            # bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 5))
            human_data['bbox_xywh'] = np.array(bbox_xywh_)
            human_data['face_bbox_xywh'] = np.array(face_bbox_xywh_)
            human_data['lhand_bbox_xywh'] = np.array(lhand_bbox_xywh_)
            human_data['rhand_bbox_xywh'] = np.array(rhand_bbox_xywh_)
            human_data['config'] = 'rich'

            for k, v in meta.items():
                meta[k] = np.array(v)
            human_data['meta'] = meta
            
            human_data.compress_keypoints_by_mask()
            # store the data struct
            if not os.path.isdir(out_path):
                os.makedirs(out_path)
            out_file = os.path.join(out_path, 'rich', f'rich_no_w_{seq_name}_{mode}.npz')
            human_data.dump(out_file)

    def _revert_smplx_hands_pca(
        self, param_dict, num_pca_comps, 
        gender='neutral', gendered_model=None):
        # egobody 12
        hl_pca = param_dict['left_hand_pose']
        hr_pca = param_dict['right_hand_pose']

        if gendered_model != None:
            smplx_model = gendered_model
        else:
            smplx_model = dict(np.load(f'data/body_models/smplx/SMPLX_{gender.upper()}.npz', allow_pickle=True))

        hl = smplx_model['hands_componentsl'] # [45, 45]
        hr = smplx_model['hands_componentsr'] # 45, 45

        hl_pca = np.concatenate((hl_pca, np.zeros((len(hl_pca), 45 - num_pca_comps))), axis=1) # [1,45]
        hr_pca = np.concatenate((hr_pca, np.zeros((len(hr_pca), 45 - num_pca_comps))), axis=1)

        hl_reverted = np.einsum('ij, jk -> ik', hl_pca, hl).astype(np.float32) #[1,45] * [45, 45] = [1, 45]
        hr_reverted = np.einsum('ij, jk -> ik', hr_pca, hr).astype(np.float32)

        param_dict['left_hand_pose'] = hl_reverted
        param_dict['right_hand_pose'] = hr_reverted

        return param_dict
    


import torch
import torch.nn as nn
from smplx.lbs import transform_mat

def extract_cam_param_xml(xml_path='', dtype=torch.float32):
    
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)

    extrinsics_mat = [float(s) for s in tree.find('./CameraMatrix/data').text.split()]
    intrinsics_mat = [float(s) for s in tree.find('./Intrinsics/data').text.split()]
    distortion_vec = [float(s) for s in tree.find('./Distortion/data').text.split()]

    focal_length_x = intrinsics_mat[0]
    focal_length_y = intrinsics_mat[4]
    center = torch.tensor([[intrinsics_mat[2], intrinsics_mat[5]]], dtype=dtype)
    
    rotation = torch.tensor([[extrinsics_mat[0], extrinsics_mat[1], extrinsics_mat[2]], 
                            [extrinsics_mat[4], extrinsics_mat[5], extrinsics_mat[6]], 
                            [extrinsics_mat[8], extrinsics_mat[9], extrinsics_mat[10]]], dtype=dtype)

    translation = torch.tensor([[extrinsics_mat[3], extrinsics_mat[7], extrinsics_mat[11]]], dtype=dtype)

    # t = -Rc --> c = -R^Tt
    cam_center = [  -extrinsics_mat[0]*extrinsics_mat[3] - extrinsics_mat[4]*extrinsics_mat[7] - extrinsics_mat[8]*extrinsics_mat[11],
                    -extrinsics_mat[1]*extrinsics_mat[3] - extrinsics_mat[5]*extrinsics_mat[7] - extrinsics_mat[9]*extrinsics_mat[11], 
                    -extrinsics_mat[2]*extrinsics_mat[3] - extrinsics_mat[6]*extrinsics_mat[7] - extrinsics_mat[10]*extrinsics_mat[11]]

    cam_center = torch.tensor([cam_center], dtype=dtype)

    k1 = torch.tensor([distortion_vec[0]], dtype=dtype)
    k2 = torch.tensor([distortion_vec[1]], dtype=dtype)

    return focal_length_x, focal_length_y, center, rotation, translation, cam_center, k1, k2

class CalibratedCamera(nn.Module):

    def __init__(self, calib_path='', rotation=None, translation=None,
                 focal_length_x=None, focal_length_y=None, 
                 batch_size=1,
                 center=None, dtype=torch.float32, **kwargs):
        super(CalibratedCamera, self).__init__()
        self.batch_size = batch_size
        self.dtype = dtype
        self.calib_path = calib_path
        # Make a buffer so that PyTorch does not complain when creating
        # the camera matrix
        self.register_buffer('zero',
                             torch.zeros([batch_size], dtype=dtype))

        import os.path as osp
        if not osp.exists(calib_path):
            raise FileNotFoundError('Could''t find {}.'.format(calib_path))
        else:
            focal_length_x, focal_length_y, center, rotation, translation, cam_center, _, _ \
                    = extract_cam_param_xml(xml_path=calib_path, dtype=dtype)
        
        if focal_length_x is None or type(focal_length_x) == float:
            focal_length_x = torch.full(
                [batch_size],               
                focal_length_x,
                dtype=dtype)

        if focal_length_y is None or type(focal_length_y) == float:
            focal_length_y = torch.full(
                [batch_size],                
                focal_length_y,
                dtype=dtype)

        self.register_buffer('focal_length_x', focal_length_x)
        self.register_buffer('focal_length_y', focal_length_y)

        if center is None:
            center = torch.zeros([batch_size, 2], dtype=dtype)
        self.register_buffer('center', center)

        rotation = rotation.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        rotation = nn.Parameter(rotation, requires_grad=False)
    
        self.register_parameter('rotation', rotation)

        if translation is None:
            translation = torch.zeros([batch_size, 3], dtype=dtype)

        translation = translation.view(3, -1).repeat(batch_size, 1, 1).squeeze(dim=-1)
        translation = nn.Parameter(translation, requires_grad=False)
        self.register_parameter('translation', translation)
        
        cam_center = nn.Parameter(cam_center, requires_grad=False)
        self.register_parameter('cam_center', cam_center)

    def forward(self, points):
        device = points.device

        with torch.no_grad():
            camera_mat = torch.zeros([self.batch_size, 2, 2],
                                     dtype=self.dtype, device=points.device)
            camera_mat[:, 0, 0] = self.focal_length_x
            camera_mat[:, 1, 1] = self.focal_length_y

        camera_transform = transform_mat(self.rotation,
                                         self.translation.unsqueeze(dim=-1))
        homog_coord = torch.ones(list(points.shape)[:-1] + [1],
                                 dtype=points.dtype,
                                 device=device)
        # Convert the points to homogeneous coordinates
        points_h = torch.cat([points, homog_coord], dim=-1)

        projected_points = torch.einsum('bki,bji->bjk',
                                        [camera_transform, points_h])

        img_points = torch.div(projected_points[:, :, :2],
                               projected_points[:, :, 2].unsqueeze(dim=-1))
        img_points = torch.einsum('bki,bji->bjk', [camera_mat, img_points]) \
            + self.center.unsqueeze(dim=1)
        return img_points