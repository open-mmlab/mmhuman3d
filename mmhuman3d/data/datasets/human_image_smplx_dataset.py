import json
import os
import os.path
from abc import ABCMeta
from collections import OrderedDict
from typing import Any, List, Optional, Union

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info

from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_num,
)
from mmhuman3d.core.evaluation import (
    fg_vertices_to_mesh_distance
)
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.data.data_structures.human_data_cache import (
    HumanDataCacheReader,
    HumanDataCacheWriter,
)
from .human_image_dataset import HumanImageDataset
from .builder import DATASETS


@DATASETS.register_module()
class HumanImageSMPLXDataset(HumanImageDataset):

    # metric
    ALLOWED_METRICS = {
        'mpjpe', 'pa-mpjpe', 'pve', '3dpck', 'pa-3dpck', '3dauc', 'pa-3dauc', '3DRMSE'
    }
    def __init__(self, 
                 data_prefix: str, 
                 pipeline: list, 
                 dataset_name: str, 
                 body_model: Optional[Union[dict, None]] = None, 
                 ann_file: Optional[Union[str, None]] = None, 
                 convention: Optional[str] = 'human_data', 
                 cache_data_path: Optional[Union[str, None]] = None, 
                 test_mode: Optional[bool] = False,
                 num_betas: Optional[int] = 10,
                 num_expression: Optional[int] = 10):
        super().__init__(data_prefix, pipeline, dataset_name, body_model, ann_file, convention, cache_data_path, test_mode)
        self.num_betas = num_betas
        self.num_expression = num_expression

    def prepare_raw_data(self, idx: int):
        """Get item from self.human_data."""
        # if self.cache_reader is not None:
        #     self.human_data = self.cache_reader.get_item(idx)
        #     idx = idx % self.cache_reader.slice_size
        # info = {}
        # info['img_prefix'] = None
        # image_path = self.human_data['image_path'][idx]
        # info['image_path'] = os.path.join(self.data_prefix, 'datasets', self.dataset_name, image_path)
        # info['dataset_name'] = self.dataset_name
        # info['sample_idx'] = idx
        # if 'bbox_xywh' in self.human_data:
        #     info['bbox_xywh'] = self.human_data['bbox_xywh'][idx]
        #     x, y, w, h, s = info['bbox_xywh']
        #     cx = x + w / 2
        #     cy = y + h / 2
        #     w = h = max(w, h)
        #     info['center'] = np.array([cx, cy])
        #     info['scale'] = np.array([w, h])
        # else:
        #     info['bbox_xywh'] = np.zeros((5))
        #     info['center'] = np.zeros((2))
        #     info['scale'] = np.zeros((2))
        # if 'keypoints2d' in self.human_data:
        #     info['keypoints2d'] = self.human_data['keypoints2d'][idx]
        #     info['has_keypoints2d'] = 1
        # else:
        #     info['keypoints2d'] = np.zeros((self.num_keypoints, 3))
        #     info['has_keypoints2d'] = 0
        # if 'keypoints3d' in self.human_data:
        #     info['keypoints3d'] = self.human_data['keypoints3d'][idx]
        #     info['has_keypoints3d'] = 1
        # else:
        #     info['keypoints3d'] = np.zeros((self.num_keypoints, 4))
        #     info['has_keypoints3d'] = 0
        info = super().prepare_raw_data(idx)
        if self.cache_reader is not None:
            self.human_data = self.cache_reader.get_item(idx)
            idx = idx % self.cache_reader.slice_size
            
        if 'smplx' in self.human_data:
            smplx_dict = self.human_data['smplx']
            info['has_smplx'] = 1
        else:
            smplx_dict = {}
            info['has_smplx'] = 0
        if 'global_orient' in smplx_dict:
            info['smplx_global_orient'] = smplx_dict['global_orient'][idx]
            info['has_smplx_global_orient'] = 1
        else:
            info['smplx_global_orient'] = np.zeros((3),dtype=np.float32)
            info['has_smplx_global_orient'] = 0

        if 'body_pose' in smplx_dict:
            info['smplx_body_pose'] = smplx_dict['body_pose'][idx]
            info['has_smplx_body_pose'] = 1
        else:
            info['smplx_body_pose'] = np.zeros((21,3),dtype=np.float32)
            info['has_smplx_body_pose'] = 0

        if 'right_hand_pose' in smplx_dict:
            info['smplx_right_hand_pose'] = smplx_dict['right_hand_pose'][idx]
            info['has_smplx_right_hand_pose'] = 1
        else:
            info['smplx_right_hand_pose'] = np.zeros((15,3),dtype=np.float32)
            info['has_smplx_right_hand_pose'] = 0

        if 'left_hand_pose' in smplx_dict:
            info['smplx_left_hand_pose'] = smplx_dict['left_hand_pose'][idx]
            info['has_smplx_left_hand_pose'] = 1
        else:
            info['smplx_left_hand_pose'] = np.zeros((15,3),dtype=np.float32)
            info['has_smplx_left_hand_pose'] = 0

        if 'jaw_pose' in smplx_dict:
            info['smplx_jaw_pose'] = smplx_dict['jaw_pose'][idx]
            info['has_smplx_jaw_pose'] = 1
        else:
            info['smplx_jaw_pose'] = np.zeros((3),dtype=np.float32)
            info['has_smplx_jaw_pose'] = 0

        if 'betas' in smplx_dict:
            info['smplx_betas'] = smplx_dict['betas'][idx]
            info['has_smplx_betas'] = 1
        else:
            info['smplx_betas'] = np.zeros((self.num_betas),dtype=np.float32)
            info['has_smplx_betas'] = 0

        if 'expression' in smplx_dict:
            info['smplx_expression'] = smplx_dict['expression'][idx]
            info['has_smplx_expression'] = 1
        else:
            info['smplx_expression'] = np.zeros((self.num_expression),dtype=np.float32)
            info['has_smplx_expression'] = 0
        


        return info

    def _parse_result(self, res, mode='keypoint'):
        if mode == 'vertice':
            # gt
            if 'vertices' in self.human_data: # stirling
                gt_vertices = self.human_data['vertices'].copy()
                gt_vertices = gt_vertices
            else:
                gt_param_dict = self.human_data['smplx'].copy()
                for key,value in gt_param_dict.items():
                    value = torch.FloatTensor(value)
                gt_output = self.body_model(**gt_param_dict)
                gt_vertices = gt_output['vertices'].detach().cpu().numpy() * 1000.

            gt_mask = np.ones(gt_vertices.shape[:-1])

            # pred
            pred_vertices = res['vertices'] * 1000.

            assert len(pred_vertices) == self.num_data
            
            return pred_vertices, gt_vertices, gt_mask
        elif mode == 'keypoint':
            pred_keypoints3d = res['keypoints']
            assert len(pred_keypoints3d) == self.num_data
            
            

            if self.dataset_name == 'pw3d' or self.dataset_name == '3DPW':
                betas = []
                body_pose = []
                global_orient = []
                gender = []
                smpl_dict = self.human_data['smpl']
                for idx in range(self.num_data):
                    betas.append(smpl_dict['betas'][idx])
                    body_pose.append(smpl_dict['body_pose'][idx])
                    global_orient.append(smpl_dict['global_orient'][idx])
                    if self.human_data['meta']['gender'][idx] == 'm':
                        gender.append(0)
                    else:
                        gender.append(1)
                betas = torch.FloatTensor(betas)
                body_pose = torch.FloatTensor(body_pose).view(-1, 69)
                global_orient = torch.FloatTensor(global_orient)
                gender = torch.Tensor(gender)
                gt_output = self.body_model(
                    betas=betas,
                    body_pose=body_pose,
                    global_orient=global_orient,
                    gender=gender)
                gt_keypoints3d = gt_output['joints'].detach().cpu().numpy()
                gt_keypoints3d_mask = np.ones((len(pred_keypoints3d), gt_keypoints3d.shape[1]))
            else:
                gt_keypoints3d = self.human_data['keypoints3d'][:,:,:3]
                gt_keypoints3d_mask = np.ones((len(pred_keypoints3d),gt_keypoints3d.shape[1]))

            if gt_keypoints3d.shape[1] == 17:
                # SMPLX_to_J14
                assert pred_keypoints3d.shape[1] == 14
                H36M_TO_J17 = [
                    6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9
                ]
                H36M_TO_J14 = H36M_TO_J17[:14]
                joint_mapper = H36M_TO_J14
                gt_keypoints3d = gt_keypoints3d[:, joint_mapper, :]
                pred_pelvis = pred_keypoints3d[:, [2,3],:].mean(axis=1, keepdims = True)
                gt_pelvis = gt_keypoints3d[:, [2,3],:].mean(axis=1, keepdims = True)
                gt_keypoints3d_mask = gt_keypoints3d_mask[:, joint_mapper]
                pred_keypoints3d = pred_keypoints3d - pred_pelvis
                gt_keypoints3d = gt_keypoints3d - gt_pelvis
            else:
                pass
            
            pred_keypoints3d = pred_keypoints3d * 1000
            if self.dataset_name != 'stirling':
                gt_keypoints3d = gt_keypoints3d * 1000
            gt_keypoints3d_mask = gt_keypoints3d_mask > 0

            return pred_keypoints3d, gt_keypoints3d, gt_keypoints3d_mask

    def _report_3d_rmse(self, res_file):
        '''compute the 3DRMSE between a predicted 3D face shape and the 3D ground truth scan
        '''
        pred_vertices, gt_vertices, _ = \
            self._parse_result(res_file, mode = 'vertice')
        pred_keypoints3d, gt_keypoints3d, _ = \
            self._parse_result(res_file, mode = 'keypoint')
        errors = []
        for pred_vertice, gt_vertice, pred_points, gt_points in zip(pred_vertices,gt_vertices,pred_keypoints3d,gt_keypoints3d):
            error = fg_vertices_to_mesh_distance(gt_vertice,gt_points,pred_vertice,self.body_model.faces, pred_points)
            errors.append(error)
        
        error = np.array(errors).mean()
        name_value_tuples = [('3DRMSE', error)]
        return name_value_tuples

    def evaluate(self,
                 outputs: list,
                 res_folder: str,
                 metric: Optional[Union[str, List[str]]] = 'pa-mpjpe',
                 **kwargs: dict):
        """Evaluate 3D keypoint results.

        Args:
            outputs (list): results from model inference.
            res_folder (str): path to store results.
            metric (Optional[Union[str, List(str)]]):
                the type of metric. Default: 'pa-mpjpe'
            kwargs (dict): other arguments.
        Returns:
            dict:
                A dict of all evaluation results.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        for metric in metrics:
            if metric not in self.ALLOWED_METRICS:
                raise KeyError(f'metric {metric} is not supported')

        # for keeping correctness during multi-gpu test, we sort all results
        res_dict = {}
        for out in outputs:
            target_id = out['image_idx']
            batch_size = len(out['keypoints_3d'])
            for i in range(batch_size):
                res_dict[int(target_id[i])] = dict(
                    keypoints=out['keypoints_3d'][i],
                    vertices=out['vertices'][i],
                )
        keypoints, vertices = [], []
        for i in range(self.num_data):
            keypoints.append(res_dict[i]['keypoints'])
            vertices.append(res_dict[i]['vertices'])
        keypoints = np.stack(keypoints)
        vertices = np.stack(vertices)
        res = dict(keypoints=keypoints, vertices=vertices)
        name_value_tuples = []
        for _metric in metrics:
            if _metric == 'mpjpe':
                _nv_tuples = self._report_mpjpe(res)
            elif _metric == 'pa-mpjpe':
                _nv_tuples = self._report_mpjpe(res, metric='pa-mpjpe')
            elif _metric == '3dpck':
                _nv_tuples = self._report_3d_pck(res)
            elif _metric == 'pa-3dpck':
                _nv_tuples = self._report_3d_pck(res, metric='pa-3dpck')
            elif _metric == '3dauc':
                _nv_tuples = self._report_3d_auc(res)
            elif _metric == 'pa-3dauc':
                _nv_tuples = self._report_3d_auc(res, metric='pa-3dauc')
            elif _metric == 'pve':
                _nv_tuples = self._report_pve(res)
            elif _metric == '3DRMSE':
                _nv_tuples = self._report_3d_rmse(res)
            else:
                raise NotImplementedError
            name_value_tuples.extend(_nv_tuples)

        name_value = OrderedDict(name_value_tuples)
        return name_value