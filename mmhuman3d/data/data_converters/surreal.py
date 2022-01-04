import os
from typing import List

import cv2
import numpy as np
import scipy.io as sio
from tqdm import tqdm

from mmhuman3d.core.cameras.camera_parameters import CameraParameter
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class SurrealConverter(BaseModeConverter):
    """SURREAL dataset `Learning from Synthetic Humans' CVPR`2017 More details
    can be found in the `paper.

    <https://arxiv.org/pdf/1701.01370.pdf>`__.

    Args:
        modes (list): 'val', 'test' or 'train' for accepted modes
        run (int): 0, 1, 2 for available runs
        extract_img (bool): Store True to extract images from video.
        Default: False.
    """
    ACCEPTED_MODES = ['val', 'train', 'test']

    def __init__(self,
                 modes: List = [],
                 run: int = 0,
                 extract_img: bool = False) -> None:
        super(SurrealConverter, self).__init__(modes)
        accepted_runs = [0, 1, 2]
        if run not in accepted_runs:
            raise ValueError('Input run not in accepted runs. \
                Use either 0 or 1 or 2')
        self.run = run
        self.extract_img = extract_img
        self.image_height = 240
        self.image_width = 320

    @staticmethod
    def get_intrinsic() -> np.ndarray:
        """Blender settings obtained from https://github.com/gulvarol/surreal/
        blob/45a90f3987b1347f0560daf1a69e22b2b7d0270c/datageneration/misc/
        smpl_relations/smpl_relations.py#L79 specific to SURREAL.

        Returns:
            K (np.ndarray): 3x3 intrinsic matrix
        """
        res_x_px = 320  # scn.render.resolution_x
        res_y_px = 240  # scn.render.resolution_y
        f_mm = 60  # cam_ob.data.lens
        sensor_w_mm = 32  # cam_ob.data.sensor_width
        # cam_ob.data.sensor_height
        sensor_h_mm = sensor_w_mm * res_y_px / res_x_px

        scale = 1  # scn.render.resolution_percentage/100
        skew = 0  # only use rectangular pixels
        pixel_aspect_ratio = 1

        # From similar triangles:
        # sensor_width_in_mm / resolution_x_inx_pix =
        # focal_length_x_in_mm / focal_length_x_in_pix
        fx_px = f_mm * res_x_px * scale / sensor_w_mm
        fy_px = f_mm * res_y_px * scale * pixel_aspect_ratio / sensor_h_mm

        # Center of the image
        u = res_x_px * scale / 2
        v = res_y_px * scale / 2

        # Intrinsic camera matrix
        K = np.array([[fx_px, skew, u], [0, fy_px, v], [0, 0, 1]])
        return K

    @staticmethod
    def get_extrinsic(T):
        """Obtained from https://github.com/gulvarol/surreal/blob/
        45a90f3987b1347f0560daf1a69e22b2b7d0270c/datageneration/misc/
        smpl_relations/smpl_relations.py#L109 specific to SURREAL.

        Args:
            T : translation vector from Blender (*cam_ob.location)

        Returns:
            RT: extrinsic matrix
        """
        # transpose of the first 3 columns of matrix_world in Blender
        # hard-coded for SURREAL images
        R_world2bcam = np.array([[0, 0, 1], [0, -1, 0], [-1, 0,
                                                         0]]).transpose()

        # Convert camera location to translation vector used in
        # coordinate changes
        T_world2bcam = -1 * np.dot(R_world2bcam, T)

        # convert Blender camera to computer vision camera
        R_bcam2cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        # transform matrix from world to computer vision camera
        R_world2cv = np.dot(R_bcam2cv, R_world2bcam)
        T_world2cv = np.dot(R_bcam2cv, T_world2bcam)

        RT = np.concatenate([R_world2cv, T_world2cv], axis=1)
        return RT, R_world2cv, T_world2cv.reshape(-1)

    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where raw images and
            annotations are stored.
            out_path (str): Path to directory to save preprocessed npz file
            mode (str): Mode in accepted modes

        Returns:
            dict:
                A dict containing keys image_path, bbox_xywh, keypoints2d,
                keypoints2d_mask, keypoints3d, keypoints3d_mask, video_path,
                smpl, meta, cam_param stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()

        # structs we use
        image_path_, bbox_xywh_, video_path_, keypoints2d_, keypoints3d_, \
            cam_param_ = [], [], [], [], [], []

        smpl = {}
        smpl['body_pose'] = []
        smpl['global_orient'] = []
        smpl['betas'] = []
        meta = {}
        meta['gender'] = []

        data_path = os.path.join(dataset_path,
                                 '{}/run{}'.format(mode, self.run))

        # go through all the .pkl files
        for seq_name in tqdm(os.listdir(data_path)):
            seq_path = os.path.join(data_path, seq_name)
            if not os.path.isdir(seq_path):
                continue
            seq_files = [
                os.path.join(seq_path, f) for f in os.listdir(seq_path)
                if (f.endswith('_info.mat') and not f.startswith('.'))
            ]
            for ann_file in seq_files:
                annotations = sio.loadmat(ann_file)
                ann_id = ann_file.split('/')[-1].split('_info.mat')[0]
                vid_path = os.path.join(seq_path, ann_id + '.mp4')
                gender = annotations['gender']
                kp2d = annotations['joints2D'].reshape(2, 24, -1)
                kp3d = annotations['joints3D'].reshape(3, 24, -1)
                pose = annotations['pose']
                beta = annotations['shape']
                K = self.get_intrinsic()
                _, R, T = self.get_extrinsic(annotations['camLoc'])
                camera = CameraParameter(
                    H=self.image_height, W=self.image_width)
                camera.set_KRT(K, R, T)
                parameter_dict = camera.to_dict()

                # image folder
                img_dir = os.path.join(seq_path, 'images_' + str(ann_id))
                rel_img_dir = img_dir.replace(dataset_path + '/', '')

                # extract frames from video file
                if self.extract_img:

                    # if doesn't exist
                    if not os.path.isdir(img_dir):
                        os.makedirs(img_dir)

                    vidcap = cv2.VideoCapture(vid_path)

                    # process video
                    frame = 0
                    while 1:
                        # extract all frames
                        success, image = vidcap.read()
                        if not success:
                            break
                        frame += 1
                        # image name
                        imgname = os.path.join(img_dir,
                                               'frame_%06d.jpg' % frame)
                        # save image
                        cv2.imwrite(imgname, image)

                num_frames = kp2d.shape[2]
                for idx in range(num_frames):
                    keypoints2d = kp2d[:, :, idx].T
                    keypoints3d = kp3d[:, :, idx].T

                    bbox_xyxy = [
                        min(keypoints2d[:, 0]),
                        min(keypoints2d[:, 1]),
                        max(keypoints2d[:, 0]),
                        max(keypoints2d[:, 1])
                    ]
                    bbox_xywh = self._bbox_expand(bbox_xyxy, scale_factor=1.2)

                    # add confidence column
                    keypoints2d = np.hstack([keypoints2d, np.ones((24, 1))])
                    keypoints3d = np.hstack([keypoints3d, np.ones([24, 1])])
                    image_path = os.path.join(rel_img_dir,
                                              'frame_%06d.jpg' % idx)

                    # store the data
                    image_path_.append(image_path)
                    video_path_.append(vid_path)
                    meta['gender'].append(int(gender[idx]))
                    keypoints2d_.append(keypoints2d)
                    keypoints3d_.append(keypoints3d)
                    bbox_xywh_.append(bbox_xywh)
                    smpl['body_pose'].append(pose[3:, idx].reshape((23, 3)))
                    smpl['global_orient'].append(pose[:3, idx])
                    smpl['betas'].append(beta[:, idx])
                    cam_param_.append(parameter_dict)

        # change list to np array
        smpl['body_pose'] = np.array(smpl['body_pose']).reshape((-1, 23, 3))
        smpl['global_orient'] = np.array(smpl['global_orient']).reshape(
            (-1, 3))
        smpl['betas'] = np.array(smpl['betas']).reshape((-1, 10))
        meta['gender'] = np.array(meta['gender']).reshape(-1)

        # convert keypoints
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 24, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'smpl', 'human_data')
        keypoints3d_ = np.array(keypoints3d_).reshape((-1, 24, 4))
        keypoints3d_, _ = convert_kps(keypoints3d_, 'smpl', 'human_data')

        human_data['image_path'] = image_path_
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['keypoints3d_mask'] = mask
        human_data['keypoints3d'] = keypoints3d_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['video_path'] = video_path_
        human_data['smpl'] = smpl
        human_data['meta'] = meta
        human_data['cam_param'] = cam_param_
        human_data['config'] = 'surreal'
        human_data.compress_keypoints_by_mask()

        # store data
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        file_name = 'surreal_{}_run{}.npz'.format(mode, self.run)
        out_file = os.path.join(out_path, file_name)
        human_data.dump(out_file)
