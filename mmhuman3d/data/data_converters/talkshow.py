import os
from tqdm import tqdm
import torch
import numpy as np
import random
import os
from typing import List, Tuple
import torch
import numpy as np
from tqdm import tqdm
import cv2
import glob
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData

from mmhuman3d.models.body_models.builder import build_body_model
from smplx.lbs import transform_mat
from mmhuman3d.models.body_models.utils import batch_transform_to_camera_frame

import cv2
from mmhuman3d.core.conventions.keypoints_mapping import get_keypoint_idxs_by_part,get_keypoint_idx

from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS

import pdb

@DATA_CONVERTERS.register_module()
class TalkshowConverter(BaseModeConverter):

    ACCEPTED_MODES = ['train']

    def __init__(self, modes: List = []) -> None:

        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        super(TalkshowConverter, self).__init__(modes)


    def _cam2pixel(self, cam_coord, f, c):
        x = cam_coord[:,:,0] / cam_coord[:,:,2] * f + c[0]
        y = cam_coord[:,:,1] / cam_coord[:,:,2] * f + c[1]
        z = cam_coord[:,:,2]

        return np.stack((x,y,z),2)


    def _revert_smplx_hands_pca(self, param_dict, num_pca_comps, gender='neutral', gendered_model=None):
            
        hl_pca = param_dict['left_hand_pose']
        hr_pca = param_dict['right_hand_pose']

        if gendered_model != None:
            smplx_model = gendered_model
        else:
            smplx_model = dict(np.load(f'data/body_models/smplx/SMPLX_{gender.upper()}.npz', allow_pickle=True))

        hl = smplx_model['hands_componentsl'] # 45, 45
        hr = smplx_model['hands_componentsr'] # 45, 45

        hl_pca = np.concatenate((hl_pca, np.zeros((len(hl_pca), 45 - num_pca_comps))), axis=1)
        hr_pca = np.concatenate((hr_pca, np.zeros((len(hr_pca), 45 - num_pca_comps))), axis=1)
        # import pdb;pdb.set_trace()
        hl_reverted = np.einsum('ij, jk -> ik', hl_pca, hl).astype(np.float32)
        hr_reverted = np.einsum('ij, jk -> ik', hr_pca, hr).astype(np.float32)

        param_dict['left_hand_pose'] = hl_reverted
        param_dict['right_hand_pose'] = hr_reverted

        return param_dict

    def _split_list(self, full_list,shuffle=False,ratio=0.2):
        n_total = len(full_list)
        offset_0 = int(n_total * ratio)
        offset_1 = int(n_total * ratio * 2)
        if n_total==0 or offset_1<1:
            return [],full_list
        if shuffle:
            random.shuffle(full_list)
        sublist_0 = full_list[:offset_0]
        sublist_1 = full_list[offset_0:offset_1]
        sublist_2 = full_list[offset_1:]
        return sublist_0, sublist_1, sublist_2

    def _read_pkl(self, data):
        betas = np.array(data['betas']) #B 300

        jaw_pose = np.array(data['jaw_pose']) #B 3
        leye_pose = np.array(data['leye_pose']) #B 3
        reye_pose = np.array(data['reye_pose']) #B 3
        global_orient = np.array(data['global_orient']).squeeze()  #B 1 63
        body_pose = np.array(data['body_pose_axis']) #B 63
        left_hand_pose = np.array(data['left_hand_pose']) #B 12
        right_hand_pose = np.array(data['right_hand_pose']) #B 12

        full_body = np.concatenate(
            (jaw_pose, leye_pose, reye_pose, global_orient, body_pose, left_hand_pose, right_hand_pose), axis=1)

        expression = np.array(data['expression']) #B 100
        full_body = np.concatenate((full_body, expression), axis=1)

        if (full_body.shape[0] < 90) or (torch.isnan(torch.from_numpy(full_body)).sum() > 0):
            return 1
        else:
            return 0
        
    def _modify_pose(self, speaker, data):
        global_orient = data['global_orient'].copy()
        transl = data['transl'].copy()
        body_pose_axis  = data['body_pose_axis'].copy()
        global_orient[:,:] = global_orient[0,:]
        transl[:,:] = transl[0,:]
        bs = body_pose_axis.shape[0]
        if (
            speaker == "oliver" or
            speaker == "seth" or
            speaker == "chemistry"
        ):
            pose_type='sitting'
        else:
            pose_type='standing'


        if pose_type == 'standing':
            ref_pose = np.zeros(55 * 3)
        elif pose_type == 'sitting':
            ref_pose = np.array([
                0.0, 0.0, 0.0, -1.1826512813568115, 0.23866955935955048, 0.15146760642528534, -1.2604516744613647,
                -0.3160211145877838, -0.1603458970785141, 0.0, 0.0, 0.0, 1.1654603481292725, 0.0, 0.0,
                1.2521806955337524, 0.041598282754421234, -0.06312154978513718, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0])

        body_pose = body_pose_axis.reshape(bs, 63)
        for i in [1, 2, 4, 5, 7, 8, 10, 11]:
            body_pose[:, (i - 1) * 3 + 0] = ref_pose[(i) * 3 + 0]
            body_pose[:, (i - 1) * 3 + 1] = ref_pose[(i) * 3 + 1]
            body_pose[:, (i - 1) * 3 + 2] = ref_pose[(i) * 3 + 2]
        # data['transl']=transl
        # data['global_orient']=global_orient
        data['body_pose_axis']=body_pose
        return data
        
    def _keypoints_to_scaled_bbox_bfh(self, keypoints_input, occ=None, body_scale=1.0, fh_scale=1.0, convention='smplx',thr=0.005):
        '''Obtain scaled bbox in xyxy format given keypoints
        Args:
            keypoints (np.ndarray): Keypoints
            scale (float): Bounding Box scale
        Returns:
            bbox_xyxy (np.ndarray): Bounding box in xyxy format
        '''
        bboxs = []
        keypoints_conf=None
        # supported kps.shape: (1, n, k) or (n, k), k = 2 or 3
        if keypoints_input.ndim == 3:
            keypoints = keypoints_input[0]
        keypoints = keypoints_input[:, :2]
        if keypoints_input.shape[-1] != 2:
            
            keypoints_conf = keypoints_input[:, 2]
        
        for body_part in ['body', 'head', 'left_hand', 'right_hand']:
            if body_part == 'body':
                scale = body_scale
                kps = keypoints
                if keypoints_conf is not None:
                    kps_conf = keypoints_conf
                    if not (kps_conf>thr).sum()<=1:
                        kps = kps[kps_conf>thr]
            else:
                scale = fh_scale
                kp_id = get_keypoint_idxs_by_part(body_part, convention=convention)
                kps = keypoints[kp_id]
                if keypoints_conf is not None:
                    kps_conf = keypoints_conf[kp_id]
                    if not (kps_conf>thr).sum()<=1:
                        kps = kps[kps_conf>thr]
            if keypoints_conf is not None and body_part != 'body':

                occ_p = kps_conf<thr
            

                
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

            bbox = np.stack([xmin, ymin, xmax, ymax, conf], axis=0).astype(np.float32)
            bboxs.append(bbox)
        
        return bboxs
    
    def _world2cam(self, world_coord, R, t):
        cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
        return cam_coord


        # srun -u -p Zoetrope /mnt/lustre/share_data/weichen1/7zip a renbody.zip renbody/image/20220924/wangkexin_f/wangkexin_yf1_dz6/image/21/*.jpg -v20g


    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:
        # parse data
        speakers = ['seth','conan','oliver','chemistry']
        # speakers = ['conan']
        data_root = "/mnt/lustrenew/share_data/zoetrope/data/datasets/talkshow/ExpressiveWholeBodyDatasetReleaseV1.0"
        split = 'train'
        vid_root = '/mnt/lustrenew/share_data/zoetrope/data/datasets/talkshow/raw_videos/'
        img_root = '/mnt/lustre/share_data/weichen1/talkshow_frames'
        root = '/mnt/lustre/share_data/weichen1'

        # build smplx model
        smplx_model = build_body_model(
            dict(
                type='SMPLX',
                keypoint_src='smplx',
                keypoint_dst='smplx',
                model_path='data/body_models/smplx',
                gender='neutral',
                use_face_contour=True,
                flat_hand_mean=True,
                use_pca=False,
                num_betas= 300,
                num_expression_coeffs=100)).to(self.device)

        # use HumanData to store the data
        human_data = HumanData()

        # init
        image_path_, bbox_xywh_, keypoints2d_smplx_, keypoints3d_smplx_, \
                keypoints2d_smpl_, keypoints3d_smpl_, keypoints2d_ori_ \
        = [], [], [], [], [], [], []

        smplx = {}
        smplx_key = [
            'betas', 'transl', 'global_orient', 'body_pose', 'left_hand_pose', 
            'right_hand_pose','leye_pose','reye_pose','jaw_pose','expression'
            ]
        for key in smplx_key:
            smplx[key] = []

        meta = {}
        meta['focal_length'] = []
        meta['principal_point'] = []
        meta['height'] = []
        meta['width'] = []
        meta['R'] = []
        meta['T'] = []
        meta['gender'] = []
        _bboxs = {}

        misc = {}
        misc_key = [
            'kps3d_root_aligned','bbox_body_scale'
            'bbox_facehand_scale','bbox_soruce','cam_param_source','smplx_source'
        ]
        misc['kps3d_root_aligned'] = True
        misc['bbox_body_scale'] = 1.2
        misc['bbox_facehand_scale'] = 1
        misc['bbox_soruce'] = 'estimated smplx'
        misc['cam_param_source'] = 'estimated'
        misc['smplx_source'] = 'PIXIE'
        misc['smpl_source'] = 'PIXIE'
        for bbox_name in ['bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh', 'rhand_bbox_xywh']:
            _bboxs[bbox_name] = []

        for speaker_name in speakers:

            speaker_root = os.path.join(data_root, speaker_name)
            speaker_vid_root = os.path.join(vid_root,speaker_name,'videos')
            videos = [v for v in os.listdir(speaker_root)]
            speaker_img_root = os.path.join(img_root, speaker_name)
            if not os.path.exists(speaker_img_root):
                os.mkdir(speaker_img_root)
            haode = huaide = 0
            total_seqs = []
            # ipdb.set_trace()
            for vid in tqdm(videos, desc="Processing training data of {}......".format(speaker_name)):
            # for vid in videos:
                source_vid = vid
                vid_pth = os.path.join(speaker_root, source_vid,split)
                # vid_pth = os.path.join(speaker_root, source_vid, 'images/half', split)
                raw_vid = os.path.join(speaker_vid_root,source_vid)
                raw_img = os.path.join(speaker_img_root,source_vid)
                # if not os.path.exists(raw_img):
                #         os.mkdir(raw_img)
                try:
                    seqs = [s for s in os.listdir(vid_pth)]
                except:
                    continue
                # if len(seqs) == 0:
                #     shutil.rmtree(os.path.join(speaker_root, source_vid))
                    # None
                
                
                for s in tqdm(seqs):
                    quality = 0
                    total_seqs.append(os.path.join(vid_pth,s))
                    seq_root = os.path.join(vid_pth, s)
                    seq_img = os.path.join(raw_img,s)
                    # if not os.path.exists(seq_img):
                    #     os.mkdir(seq_img)

                    
                    key = seq_root  # correspond to clip******
                    
                    motion_fname = os.path.join(speaker_root, source_vid,split, s, '%s.pkl' % (s))
                    # print(motion_fname)
                    # import ipdb;ipdb.set_trace()
                    try:
                        data = np.load(motion_fname, allow_pickle=True)
                    except:
                        # shutil.rmtree(key)
                        huaide = huaide + 1
                        # continue
                    

                    # ipdb.set_trace()
                    
                    # data = pickle.load(f)
                    st,et = s.split('-')[1].replace('_',':'),s.split('-')[2].replace('_',':')

                    

                    # import ipdb;ipdb.set_trace()
                    # video_to_images(
                    #     vid_file=raw_vid,
                    #     start=st,
                    #     end=et,
                    #     img_folder=seq_img
                    # )
                    
                    frame_list = glob.glob(seq_img.replace(' ','_')+'/*')
                    frame_list.sort()
                    # print(frame_list)
                    if len(frame_list) != len(data['body_pose_axis']):
                        print('skippp',seq_img)
                        continue
                    # import ipdb;ipdb.set_trace()
                    data = self._revert_smplx_hands_pca(data,12)
                    self._modify_pose(speaker_name,data)
                    # ipdb.set_trace()
                    # import ipdb;ipdb.set_trace()
                    jaw_pose = torch.Tensor(data['jaw_pose']).to(self.device)

                    batch_size = len(jaw_pose)
                    betas = torch.Tensor(data['betas']).to(self.device).repeat(batch_size,1)
                    
                    leye_pose = torch.Tensor(data['leye_pose']).to(self.device)
                    reye_pose = torch.Tensor(data['reye_pose']).to(self.device)
                    global_orient = torch.Tensor(data['global_orient']).squeeze().to(self.device)
                    body_pose = torch.Tensor(data['body_pose_axis']).to(self.device)
                    left_hand_pose = torch.Tensor(data['left_hand_pose']).to(self.device).reshape(-1,15,3)
                    right_hand_pose = torch.Tensor(data['right_hand_pose']).to(self.device).reshape(-1,15,3)
                    transl = torch.Tensor(data['transl']).to(self.device)
                    expression = torch.Tensor(data['expression']).to(self.device) #B 100
                    # full_body = np.concatenate(
                    #     (jaw_pose, leye_pose, reye_pose, global_orient, body_pose, left_hand_pose, right_hand_pose), axis=1)
                    focal  = data['focal_length']
                    princpt = data['center']
                    # princpt
                    K = np.array([
                        [focal, 0, princpt[0]],
                        [0, focal, princpt[1]],
                        [0, 0, 1]])
                    T = data['camera_transl']
                    T[1] *= -1
                    T[0] *= -1
                    # import ipdb;ipdb.set_trace()
                    smplx_res = smplx_model(
                                    betas=betas, 
                                    body_pose=body_pose,  
                                    global_orient=global_orient, 
                                    transl=transl,
                                    left_hand_pose=left_hand_pose,
                                    right_hand_pose=right_hand_pose,
                                    jaw_pose=jaw_pose,
                                    reye_pose=reye_pose,
                                    leye_pose=leye_pose,
                                    expression=expression,

                                    pose2rot=True,
                                    return_full_pose=True)
                    rotation = torch.eye(3)[None].cpu()
                    # ipdb.set_trace()
                    camera_transform = transform_mat(rotation,torch.Tensor(T).unsqueeze(dim=0).unsqueeze(dim=-1).cpu())
                    # for b_idx in range(len(global_orient)):
                    # import ipdb;ipdb.set_trace()

                    new_smplx_global_orient, new_smplx_transl = batch_transform_to_camera_frame(data['global_orient'].squeeze(1), 
                                                                            data['transl'], 
                                                                            smplx_res['joints'][:,0,:].cpu().detach().numpy(), (camera_transform[0]).cpu().numpy())
                    # import ipdb;ipdb.set_trace()
                    smplx_res_new = smplx_model(
                                    betas=betas, 
                                    body_pose=body_pose,  
                                    global_orient=torch.Tensor(new_smplx_global_orient).to(self.device), 
                                    transl=torch.Tensor(new_smplx_transl).to(self.device), 
                                    left_hand_pose=left_hand_pose,
                                    right_hand_pose=right_hand_pose,
                                    jaw_pose=jaw_pose,
                                    reye_pose=reye_pose,
                                    leye_pose=leye_pose,
                                    expression=expression,

                                    pose2rot=True,
                                    return_full_pose=True)
                    
                    
                    smplx_keypoints3d = smplx_res_new['joints'].clone()
                    # import ipdb;ipdb.set_trace()
                    smplx_keypoints3d_img = self._cam2pixel(smplx_keypoints3d.cpu().detach()*1000, focal, torch.Tensor(princpt))
                    smplx_keypoints2d = smplx_keypoints3d_img[:,:,:2]
                    smplx_pelvis = get_keypoint_idx('pelvis','smplx')
                    smplx_keypoints3d_root = smplx_keypoints3d - smplx_keypoints3d[:,smplx_pelvis].unsqueeze(1)
                    
                    
                    for smpl_kp2d in smplx_keypoints2d:
                        bbox_tmp_ = {}
                        # import ipdb;ipdb.set_trace()
                        bbox_tmp_['bbox_xywh'],bbox_tmp_['face_bbox_xywh'], bbox_tmp_['lhand_bbox_xywh'], bbox_tmp_[
                            'rhand_bbox_xywh'] = self._keypoints_to_scaled_bbox_bfh(smpl_kp2d, body_scale=1.2, fh_scale=1,convention='smplx')
                        
                        for bbox_name in ['bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh', 'rhand_bbox_xywh']:
                            bbox = bbox_tmp_[bbox_name]
                            xmin, ymin, xmax, ymax = bbox[:4]
                            if bbox_name == 'bbox_xywh':
                                bbox_conf = 1
                            else:
                                bbox_conf = bbox[-1]
                                # if bbox_conf == 0:
                                #     print(f'{npzf}, {idx},{bbox_name} invalid')
                            # import pdb; pdb.set_trace()
                            bbox = np.array([max(0, xmin), max(0, ymin), min(data['width'], xmax), min(data['height'], ymax)])
                            bbox_xywh = self._xyxy2xywh(bbox)
                            bbox_xywh.append(bbox_conf)
                            # import ipdb;ipdb.set_trace()
                            import mmcv
                            bbox_xyxy = np.array(bbox_xywh[:4].copy())
                            # bbox_xyxy = np.array(ann['bbox'])
                            bbox_xyxy[2:4] += bbox_xyxy[:2]
                            self._cam2pixel(smplx_res_new['joints'].detach().cpu(),5000,[360.0, 640.0])             
                            # img_ = mmcv.imshow_bboxes((img*255).astype(np.uint8).copy(),bbox_xyxy[None],show=False)
                            # cv2.imwrite('test%s.png'%bbox_name,img_)
                            

                            _bboxs[bbox_name].append(bbox_xywh)
                    # ipdb.set_trace()
                    for img in frame_list:
                        # print(os.path.join(*img.split('/')[-4:]))
                        image_path_.append(os.path.join(*img.split('/')[-4:]))
                    keypoints2d_smplx_.append(smplx_keypoints2d)
                    keypoints3d_smplx_.append(smplx_keypoints3d_root.detach().cpu().numpy())
                    # jaw_pose = torch.Tensor(data['jaw_pose']).to(self.device)
                    # leye_pose = torch.Tensor(data['leye_pose']).to(self.device)
                    # reye_pose = torch.Tensor(data['reye_pose']).to(self.device)
                    # global_orient = torch.Tensor(data['global_orient']).squeeze().to(self.device)
                    # body_pose = torch.Tensor(data['body_pose_axis']).to(self.device)
                    # left_hand_pose = torch.Tensor(data['left_hand_pose']).to(self.device)
                    # right_hand_pose = torch.Tensor(data['right_hand_pose']).to(self.device)
                    # transl = torch.Tensor(data['transl']).to(self.device)
                    # expression = torch.Tensor(data['expression']).to(self.device) #B 100
                    # import ipdb;ipdb.set_trace()

                    meta['focal_length'].append(np.ones([batch_size])*focal)
                    meta['principal_point'].append(np.ones([batch_size,2])*princpt)
                    meta['height'].append(np.ones(batch_size)*data['height'])
                    meta['width'].append(np.ones(batch_size)*data['width'])


                    for k in smplx.keys():
                        if k == 'global_orient' or k == 'transl' or k=='betas':
                            continue
                        if k=='body_pose':
                            smplx[k].append(data['body_pose_axis'])
                            continue
                        smplx[k].append(data[k])
                    smplx['betas'].append(betas[:,:10].cpu().detach().numpy())
                    smplx['global_orient'].append(new_smplx_global_orient)
                    smplx['transl'].append(new_smplx_transl)
                # print('done')
                # break
            # print('done2')
            # break
        for k in smplx.keys():
            smplx[k] = np.concatenate(smplx[k],axis=0)
        for key in _bboxs.keys():
            
            bbox_ = np.array(_bboxs[key]).reshape((-1, 5))
            # bbox_ = np.hstack([bbox_, np.zeros([bbox_.shape[0], 1])])
            # import pdb; pdb.set_trace()
            human_data[key] = bbox_


        keypoints3d_smplx_ = np.concatenate(keypoints3d_smplx_, axis=0)
        keypoints3d_smplx_conf = np.ones([keypoints3d_smplx_.shape[0], 144, 1])
        keypoints3d_smplx_ = np.concatenate([keypoints3d_smplx_, keypoints3d_smplx_conf], axis=-1)
        keypoints3d_smplx_, keypoints3d_smplx_mask_ = \
                convert_kps(keypoints3d_smplx_, src='smplx', dst='human_data')

        keypoints2d_smplx_ = np.concatenate(keypoints2d_smplx_, axis=0)
        keypoints2d_smplx_conf = np.ones([keypoints2d_smplx_.shape[0], 144, 1])
        keypoints2d_smplx_ = np.concatenate([keypoints2d_smplx_, keypoints2d_smplx_conf], axis=-1)
        keypoints2d_smplx_, keypoints2d_smplx_mask_ = convert_kps(keypoints2d_smplx_, 'smplx', 'human_data')        

        for k in meta.keys():
            # print(meta[k])
            if k!= 'R' and k!= 'T' and k!='gender':
                meta[k] =np.concatenate(meta[k],axis=0)
        # import ipdb;ipdb.set_trace()
        # human_data['bbox_xywh'] = bbox_xywh_
        human_data['image_path'] = image_path_

        human_data['keypoints2d_smplx'] = keypoints2d_smplx_
        human_data['keypoints2d_smplx_mask'] = keypoints2d_smplx_mask_

        human_data['keypoints3d_smplx'] = keypoints3d_smplx_
        human_data['keypoints3d_smplx_mask'] = keypoints3d_smplx_mask_

        human_data['smplx'] = smplx
        human_data['config'] = 'talkshow'
        human_data.compress_keypoints_by_mask()
        human_data['meta'] = meta
        human_data['misc'] = misc
        # human_data['cam_param'] = cam_param_
            

        file_name = 'talkshow_smplx_conan.npz'
        human_data.dump(file_name)


        return 0













    # def moveto(list, file):
    #     for f in list:
    #         before, after = '/'.join(f.split('/')[:-1]), f.split('/')[-1]
    #         new_path = os.path.join(before, file)
    #         new_path = os.path.join(new_path, after)
    #         # os.makedirs(new_path)
    #         # os.path.isdir(new_path)
    #         # shutil.move(f, new_path)

    #         #转移到新目录
    #         shutil.copytree(f, new_path)
    #         #删除原train里的文件
    #         shutil.rmtree(f)
    #     return None

    # def video_to_images(
    #     vid_file,
    #     prefix='',
    #     start='00:00',
    #     end='00:00',
    #     img_folder=None,
    #     return_info=False,
    #     fps=30
    # ):
    #     '''
    #     From https://github.com/mkocabas/VIBE/blob/master/lib/utils/demo_utils.py

    #     fps will sample the video to this rate.
    #     '''
    #     if img_folder is None:
    #         img_folder = os.path.join('/tmp', os.path.basename(vid_file).replace('.', '_'))

    #     os.makedirs(img_folder, exist_ok=True)

    #     command = ['ffmpeg',
    #             '-i', vid_file,
    #             '-r', str(fps),
    #             '-ss', str(start),
    #             '-to', str(end),
    #             '-f', 'image2',
    #             '-v', 'error',
    #             f'{img_folder}/{prefix}%06d.jpg']
    #     print(f'Running \"{" ".join(command)}\"')
    #     subprocess.call(command)

    #     print(f'Images saved to \"{img_folder}\"')




