import argparse
import json
import os
import math

import numpy as np
import torch
from tqdm import tqdm

from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.models.body_models.builder import build_body_model

import pdb

def xyxy2xywh(bbox_xyxy):

    x1, y1, x2, y2 = bbox_xyxy
    return [x1, y1, x2 - x1, y2 - y1]


def process_vid(vid, smplx_model, anno_param, smplx_param):
    # vid = args.vid_p
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # prepare paths
    root_idx = vid.split(os.path.sep).index('ubody')
    anno_folder = os.path.sep.join(vid.split(
        os.path.sep)[:root_idx + 3]).replace('videos', 'annotations')

    seq = os.path.basename(vid)[:-4]
    image_base_path = os.path.sep.join(
        vid.split(os.path.sep)[root_idx + 1:root_idx + 3]).replace(
            'videos', 'images')

    preprocess_folder = os.path.sep.join(
        vid.split(os.path.sep)[:root_idx + 3]).replace('videos', 'preprocess')
    preprocess_file = os.path.join(preprocess_folder, f'{seq}.npz')

    if os.path.exists(preprocess_file):
        return
        # pass

    smplx_shape = {
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
    bbox_mapping = {
        'bbox_xywh': 'bbox',
        'face_bbox_xywh': 'face_box',
        'lhand_bbox_xywh': 'lefthand_box',
        'rhand_bbox_xywh': 'righthand_box'
    }
    smplx_mapping = {
        'betas': 'shape',
        'transl': 'trans',
        'global_orient': 'root_pose',
        'body_pose': 'body_pose',
        'left_hand_pose': 'lhand_pose',
        'right_hand_pose': 'rhand_pose',
        'jaw_pose': 'jaw_pose',
        'expression': 'expr'
    }

    param_dict = {}
    for key in [
            'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
            'rhand_bbox_xywh', 'betas', 'transl', 'global_orient', 'body_pose',
            'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'expression',
            'keypoints2d', 'keypoints3d', 'keypoints2d_ubody', 'image_path',
            'height', 'width', 'principal_point', 'focal_length',
            'iscrowd', 'num_keypoints', 'valid_label', 'lefthand_valid',
            'righthand_valid', 'face_valid',
    ]:
        param_dict[key] = []

    ids = [
        image_info['id'] for image_info in anno_param['images']
        if seq in image_info['file_name']
        and str(image_info['id']) in smplx_param.keys()
    ]
    idxs_match = [
        (idx, anno['image_id'], anno['id']) for idx, anno in enumerate(anno_param['annotations'])
        if int(anno['image_id']) in ids
    ]

    for (aid, iid, idx) in tqdm(
            idxs_match, desc=f'Video frams: {seq}', leave=False, position=1):
    # for (aid, iid, idx) in idxs_match:
        kp_param = anno_param['annotations'][aid]
        image_id = kp_param['image_id']
        image_info = anno_param['images'][iid]
        # generate image info
        image_path = os.path.join(image_base_path, image_info['file_name'])
        id = kp_param['id']
        if str(id) not in smplx_param.keys():
            continue
        if '/'.join( vid.split('/')[-3:-1]) not in image_path:
            print(f'Wrong image_path found in {vid}')
            continue

        height = image_info['height']
        width = image_info['width']

        # collect coco_wholebody keypoints
        body_kps = kp_param['keypoints']
        foot_kps = kp_param['foot_kpts']
        face_kps = kp_param['face_kpts']
        lhand_kps = kp_param['lefthand_kpts']
        rhand_kps = kp_param['righthand_kpts']

        keypoints_2d_ubody = np.array(body_kps + foot_kps + face_kps +
                                      lhand_kps + rhand_kps).reshape(-1, 3)

        # collect bbox
        for bbox_name in [
                'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
                'rhand_bbox_xywh'
        ]:
            xmin, ymin, w, h = kp_param[bbox_mapping[bbox_name]]
            bbox = np.array([
                max(0, xmin),
                max(0, ymin),
                min(width, xmin + w),
                min(height, ymin + h)
            ])
            bbox_xywh = xyxy2xywh(bbox)  # list of len 4
            if bbox_xywh[2] * bbox_xywh[3] > 0:
                bbox_xywh.append(1)  # (5,)
            else:
                bbox_xywh.append(0)
            param_dict[bbox_name].append(bbox_xywh)
        # import pdb; pdb.set_trace()
        # collect smplx
        smplx_frame_param = smplx_param[str(id)]['smplx_param']
        camera_frame_param = smplx_param[str(id)]['cam_param']

        # generate smplx keypoints
        smplx_temp = {}
        for key in smplx_mapping.keys():
            smplx_temp[key] = np.array(
                smplx_frame_param[smplx_mapping[key]],
                dtype=np.float32).reshape(smplx_shape[key])

        output = smplx_model(
            global_orient=torch.tensor(
                smplx_temp['global_orient'], device=device),
            body_pose=torch.tensor(smplx_temp['body_pose'], device=device),
            betas=torch.tensor(smplx_temp['betas'], device=device),
            transl=torch.tensor(smplx_temp['transl'], device=device),
            left_hand_pose=torch.tensor(
                smplx_temp['left_hand_pose'], device=device),
            right_hand_pose=torch.tensor(
                smplx_temp['right_hand_pose'], device=device),
            jaw_pose=torch.tensor(smplx_temp['jaw_pose'], device=device),
            expression=torch.tensor(smplx_temp['expression'], device=device),
            return_joints=True,
        )
        keypoints_3d = output['joints']

        # build camera
        focal_length = camera_frame_param['focal']
        principal_point = camera_frame_param['princpt']
        camera = build_cameras(
            dict(
                type='PerspectiveCameras',
                convention='opencv',
                in_ndc=False,
                focal_length=np.array(focal_length).reshape(-1, 2),
                image_size=(height, width),
                principal_point=np.array(principal_point).reshape(
                    -1, 2))).to(device)

        # perspective projection 3d to 2d keypoints
        keypoints_2d_xyd = camera.transform_points_screen(keypoints_3d)
        keypoints_2d = keypoints_2d_xyd[..., :2].detach().cpu().numpy()
        keypoints_3d = keypoints_3d.detach().cpu().numpy()

        # add image path
        param_dict['image_path'].append(image_path)

        # add keypoints
        param_dict['keypoints2d_ubody'].append(keypoints_2d_ubody)
        param_dict['keypoints2d'].append(keypoints_2d)
        param_dict['keypoints3d'].append(keypoints_3d)

        # add smplx param
        for key in smplx_temp:
            param_dict[key].append(smplx_temp[key])

        # append meta
        param_dict['height'].append(height)
        param_dict['width'].append(width)
        param_dict['focal_length'].append(focal_length)
        param_dict['principal_point'].append(principal_point)

        for key in ['iscrowd', 'num_keypoints', 'valid_label', 'lefthand_valid',
            'righthand_valid', 'face_valid']:
            param_dict[key].append(kp_param[key])
        # pdb.set_trace()

    # import pdb; pdb.set_trace()
    os.makedirs(preprocess_folder, exist_ok=True)
    np.savez(preprocess_file, **param_dict)

    # for vid in tqdm(vid_ps, desc=f'Scene: {scene}', leave=False, position=1):
    #     self.preprocess_ubody(vid)
    # root_idx = vid.split(os.path.sep).index('ubody')
    # anno_folder = os.path.sep.join(vid.split(os.path.sep)[:root_idx+3]).replace('videos', 'annotations')

    # seq = os.path.basename(vid)[:-4]
    # image_base_path = os.path.sep.join(vid.split(os.path.sep)[root_idx+1:root_idx+3]).replace('videos', 'images')

    # # load seq kp annotation
    # with open(os.path.join(anno_folder, 'keypoint_annotation.json')) as f:
    #     anno_param =json.load(f)
    # # load seq smplx annotation
    # with open(os.path.join(anno_folder, 'smplx_annotation.json')) as f:
    #     smplx_param =json.load(f)

    # ids = [image_info['id'] for image_info in anno_param['images']
    #         if seq in image_info['file_name'] and str(image_info['id']) in smplx_param.keys()]
    # idxs_anno = [idx for idx, anno in enumerate(anno_param['annotations']) if int(anno['id']) in ids]

    # for idx in tqdm(idxs_anno, desc=f'Video frams: {seq}', leave=False, position=2):
    #     kp_param = anno_param['annotations'][idx]
    #     id = kp_param['id']

    #     image_info = anno_param['images'][id]

    #     # generate image info
    #     image_path = os.path.join(image_base_path, image_info['file_name'])
    #     image_id = image_info['id']

    #     height = image_info['height']
    #     width = image_info['width']

    #     # collect coco_wholebody keypoints
    #     body_kps = kp_param['keypoints']
    #     foot_kps = kp_param['foot_kpts']
    #     face_kps = kp_param['face_kpts']
    #     lhand_kps = kp_param['lefthand_kpts']
    #     rhand_kps = kp_param['righthand_kpts']

    #     keypoints_2d_ubody = np.array(body_kps + foot_kps + face_kps + lhand_kps + rhand_kps).reshape(-1, 3)

    #     # collect bbox
    #     for bbox_name in ['bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh', 'rhand_bbox_xywh']:
    #         xmin, ymin, w, h = kp_param[self.bbox_mapping[bbox_name]]
    #         bbox = np.array([max(0, xmin), max(0, ymin), min(width, xmin+w), min(height, ymin+h)])
    #         bbox_xywh = self._xyxy2xywh(bbox)  # list of len 4
    #         if bbox_xywh[2] * bbox_xywh[3] > 0:
    #             bbox_xywh.append(1)  # (5,)
    #         else:
    #             bbox_xywh.append(0)
    #         bboxs_[bbox_name].append(bbox_xywh)

    #     # collect smplx
    #     smplx_frame_param = smplx_param[str(image_id)]['smplx_param']
    #     camera_frame_param = smplx_param[str(image_id)]['cam_param']

    #     # generate smplx keypoints
    #     smplx_temp = {}
    #     for key in self.smplx_mapping.keys():
    #         smplx_temp[key] = np.array(smplx_frame_param[self.smplx_mapping[key]],
    #                                     dtype=np.float32).reshape(self.smplx_shape[key])

    #     output = smplx_model(
    #         global_orient=torch.tensor(smplx_temp['global_orient'], device=self.device),
    #         body_pose=torch.tensor(smplx_temp['body_pose'], device=self.device),
    #         betas=torch.tensor(smplx_temp['betas'], device=self.device),
    #         transl=torch.tensor(smplx_temp['transl'], device=self.device),
    #         left_hand_pose=torch.tensor(smplx_temp['left_hand_pose'], device=self.device),
    #         right_hand_pose=torch.tensor(smplx_temp['right_hand_pose'], device=self.device),
    #         jaw_pose=torch.tensor(smplx_temp['jaw_pose'], device=self.device),
    #         expression=torch.tensor(smplx_temp['expression'], device=self.device),
    #         return_joints=True,
    #     )
    #     keypoints_3d = output['joints']

    #     # build camera
    #     focal_length = camera_frame_param['focal']
    #     principal_point = camera_frame_param['princpt']
    #     camera = build_cameras(
    #         dict(
    #             type='PerspectiveCameras',
    #             convention='opencv',
    #             in_ndc=False,
    #             focal_length=np.array(focal_length).reshape(-1, 2),
    #             image_size=(height, width),
    #             principal_point=np.array(principal_point).reshape(-1, 2))).to(self.device)

    #     # perspective projection 3d to 2d keypoints
    #     keypoints_2d_xyd = camera.transform_points_screen(keypoints_3d)
    #     keypoints_2d = keypoints_2d_xyd[..., :2].detach().cpu().numpy()
    #     keypoints_3d = keypoints_3d.detach().cpu().numpy()

    #     # add image path
    #     image_path_.append(image_path)

    #     # add keypoints
    #     keypoints2d_ubody_.append(keypoints_2d_ubody)
    #     keypoints2d_.append(keypoints_2d)
    #     keypoints3d_.append(keypoints_3d)

    #     # add smplx param
    #     for key in smplx_temp:
    #         smplx_[key].append(smplx_temp[key])

    #     # append meta
    #     meta_['height'].append(height)
    #     meta_['width'].append(width)
    #     meta_['focal_length'].append(focal_length)
    #     meta_['principal_point'].append(principal_point)


def process_vid_COCO(smplx_model, vscene, scene_split, batch, db, smplx_param, dst):
    # vid = args.vid_p
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    smplx_shape = {
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
    bbox_mapping = {
        'bbox_xywh': 'bbox',
        'face_bbox_xywh': 'face_box',
        'lhand_bbox_xywh': 'lefthand_box',
        'rhand_bbox_xywh': 'righthand_box'
    }
    smplx_mapping = {
        'betas': 'shape',
        'transl': 'trans',
        'global_orient': 'root_pose',
        'body_pose': 'body_pose',
        'left_hand_pose': 'lhand_pose',
        'right_hand_pose': 'rhand_pose',
        'jaw_pose': 'jaw_pose',
        'expression': 'expr'
    }

    param_dict = {}
    for key in [
            'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
            'rhand_bbox_xywh', 'betas', 'transl', 'global_orient', 'body_pose',
            'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'expression',
            'keypoints2d', 'keypoints3d', 'keypoints2d_ubody', 'image_path',
            'height', 'width', 'principal_point', 'focal_length',
            'iscrowd', 'num_keypoints', 'valid_label', 'lefthand_valid',
            'righthand_valid', 'face_valid',
    ]:
        param_dict[key] = []

    block_size = 10000
    blocks = math.ceil(len(db.anns.keys()) / block_size)
    aids = list(db.anns.keys())

    # prepare vid list 
    # vid_list = [vid_p.split('/')[-2] for vid_p in vid_ps_batch]
    dataset_path = os.path.sep.join(dst.split('/')[:-2])

    valid_count = 0
    for bid in range(blocks):

        param_dict = {}
        for key in [
                'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
                'rhand_bbox_xywh', 'betas', 'transl', 'global_orient', 'body_pose',
                'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'expression',
                'keypoints2d', 'keypoints3d', 'keypoints2d_ubody', 'image_path',
                'height', 'width', 'principal_point', 'focal_length',
                'iscrowd', 'num_keypoints', 'valid_label', 'lefthand_valid',
                'righthand_valid', 'face_valid',
        ]:
            param_dict[key] = []
        preprocess_file = os.path.join(dst, f'{vscene}_{bid}.npz')
        if os.path.exists(preprocess_file):
            continue

        for aid in tqdm(aids[block_size*bid: block_size*(bid+1)], 
                        desc=f'Processing scene {vscene}, block id {bid+1} / {blocks}', 
                        leave=False, position=1):

        # for aid in aids[block_size*bid: block_size*(bid+1)]:
    
            ann = db.anns[aid]
            image_info = db.loadImgs(ann['image_id'])[0]
            video_name = image_info['file_name'].split('/')[-2]

            if image_info['file_name'].startswith('/'):
                file_name = image_info['file_name'][1:]  # [1:] means delete '/'
            else:
                file_name = image_info['file_name']

            video_name = file_name.split('/')[-2]
            if 'Trim' in video_name:
                video_name = video_name.split('_Trim')[0]

            if batch == 'train':
                if video_name in scene_split:
                    continue
            elif batch == 'test':
                if video_name not in scene_split:
                    continue       

            imgp = os.path.join(dataset_path, 'images', 
                                vscene, image_info['file_name'])
            if ann['iscrowd'] or (ann['num_keypoints']==0): continue
            if ann['valid_label'] == 0: continue
            if not os.path.exists(imgp): 
                continue
            if str(aid) not in smplx_param: 
                continue
            
            kp_param = ann
            image_id = kp_param['image_id']
            # generate image info
            image_path = imgp.replace(f'{dataset_path}/', '')
            id = kp_param['id']
            # if str(id) not in smplx_param.keys():
            #     continue

            height = image_info['height']
            width = image_info['width']

            # collect coco_wholebody keypoints
            body_kps = kp_param['keypoints']
            foot_kps = kp_param['foot_kpts']
            face_kps = kp_param['face_kpts']
            lhand_kps = kp_param['lefthand_kpts']
            rhand_kps = kp_param['righthand_kpts']

            keypoints_2d_ubody = np.array(body_kps + foot_kps + face_kps +
                                        lhand_kps + rhand_kps).reshape(-1, 3)

            # collect bbox
            for bbox_name in [
                    'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
                    'rhand_bbox_xywh'
            ]:
                xmin, ymin, w, h = kp_param[bbox_mapping[bbox_name]]
                bbox = np.array([
                    max(0, xmin),
                    max(0, ymin),
                    min(width, xmin + w),
                    min(height, ymin + h)
                ])
                bbox_xywh = xyxy2xywh(bbox)  # list of len 4
                if bbox_xywh[2] * bbox_xywh[3] > 0:
                    bbox_xywh.append(1)  # (5,)
                else:
                    bbox_xywh.append(0)
                param_dict[bbox_name].append(bbox_xywh)
            # import pdb; pdb.set_trace()
            # collect smplx
            smplx_frame_param = smplx_param[str(id)]['smplx_param']
            camera_frame_param = smplx_param[str(id)]['cam_param']

            # generate smplx keypoints
            smplx_temp = {}
            for key in smplx_mapping.keys():
                smplx_temp[key] = np.array(
                    smplx_frame_param[smplx_mapping[key]],
                    dtype=np.float32).reshape(smplx_shape[key])

            output = smplx_model(
                global_orient=torch.tensor(
                    smplx_temp['global_orient'], device=device),
                body_pose=torch.tensor(smplx_temp['body_pose'], device=device),
                betas=torch.tensor(smplx_temp['betas'], device=device),
                transl=torch.tensor(smplx_temp['transl'], device=device),
                left_hand_pose=torch.tensor(
                    smplx_temp['left_hand_pose'], device=device),
                right_hand_pose=torch.tensor(
                    smplx_temp['right_hand_pose'], device=device),
                jaw_pose=torch.tensor(smplx_temp['jaw_pose'], device=device),
                expression=torch.tensor(smplx_temp['expression'], device=device),
                return_joints=True,
            )
            keypoints_3d = output['joints']

            # build camera
            focal_length = camera_frame_param['focal']
            principal_point = camera_frame_param['princpt']
            camera = build_cameras(
                dict(
                    type='PerspectiveCameras',
                    convention='opencv',
                    in_ndc=False,
                    focal_length=np.array(focal_length).reshape(-1, 2),
                    image_size=(height, width),
                    principal_point=np.array(principal_point).reshape(
                        -1, 2))).to(device)

            # perspective projection 3d to 2d keypoints
            keypoints_2d_xyd = camera.transform_points_screen(keypoints_3d)
            keypoints_2d = keypoints_2d_xyd[..., :2].detach().cpu().numpy()
            keypoints_3d = keypoints_3d.detach().cpu().numpy()

            # add image path
            param_dict['image_path'].append(image_path)

            # add keypoints
            param_dict['keypoints2d_ubody'].append(keypoints_2d_ubody)
            param_dict['keypoints2d'].append(keypoints_2d)
            param_dict['keypoints3d'].append(keypoints_3d)

            # add smplx param
            for key in smplx_temp:
                param_dict[key].append(smplx_temp[key])

            # append meta
            param_dict['height'].append(height)
            param_dict['width'].append(width)
            param_dict['focal_length'].append(focal_length)
            param_dict['principal_point'].append(principal_point)

            for key in ['iscrowd', 'num_keypoints', 'valid_label', 'lefthand_valid',
                'righthand_valid', 'face_valid']:
                param_dict[key].append(kp_param[key])
            valid_count += 1
            # pdb.set_trace()

        os.makedirs(os.path.dirname(preprocess_file), exist_ok=True)
        np.savez(preprocess_file, **param_dict)

    print(f'{vscene}: Valid count: {valid_count}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ubody preprocess')
    parser.add_argument(
        '--vid_p', type=str, required=True, help='path to the video')

    args = parser.parse_args()

    # build smplx model
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    smplx_model = build_body_model(
        dict(
            type='SMPLX',
            keypoint_src='smplx',
            keypoint_dst='smplx',
            model_path='data/body_models/smplx',
            gender='neutral',
            num_betas=10,
            use_face_contour=True,
            flat_hand_mean=True,
            use_pca=False,
            batch_size=1)).to(device)


    process_vid(args.vid_p, smplx_model)
