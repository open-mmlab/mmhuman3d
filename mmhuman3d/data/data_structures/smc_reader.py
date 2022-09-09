import json

import cv2
import h5py
import numpy as np
import torch
import tqdm

from mmhuman3d.models.body_models.builder import build_body_model
from mmhuman3d.models.body_models.utils import batch_transform_to_camera_frame


class SMCReader:

    def __init__(self, file_path, body_model=None):
        """Read SenseMocapFile endswith ".smc", see: https://github.com/open-
        mmlab/mmhuman3d/blob/main/docs/smc.md.

        Args:
            file_path (str):
                Path to an SMC file.
            body_model (nn.Module or dict):
                Only needed for SMPL transformation to device frame
                if nn.Module: a body_model instance
                if dict: a body_model config
        """
        self.smc = h5py.File(file_path, 'r')
        self.__calibration_dict__ = None
        self.action_id = self.smc.attrs['action_id']
        self.actor_id = self.smc.attrs['actor_id']
        self.datetime_str = self.smc.attrs['datetime_str']  # .decode()
        self.kinect_num_frames = self.smc['Kinect'].attrs['num_frame']
        self.num_kinects = self.smc['Kinect'].attrs['num_device']
        self.kinect_color_resolution = self.get_kinect_color_resolution(0)
        self.kinect_depth_resolution = self.get_kinect_depth_resolution(0)
        self.iphone_exists = 'iPhone' in self.smc.keys()
        self.num_iphones = 1
        if self.iphone_exists:
            self.iphone_num_frames = self.smc['iPhone'].attrs['num_frame']
            self.iphone_color_resolution = \
                self.smc['iPhone'].attrs['color_resolution']  # vertical
            self.iphone_depth_resolution = \
                self.smc['iPhone'].attrs['depth_resolution']  # vertical
        self.keypoint_exists = 'Keypoints3D' in self.smc.keys()
        if self.keypoint_exists:
            self.keypoints_num_frames = self.smc['Keypoints3D'].attrs[
                'num_frame']
            self.keypoints_convention = self.smc['Keypoints3D'].attrs[
                'convention']
            self.keypoints_created_time = self.smc['Keypoints3D'].attrs[
                'created_time']
        self.smpl_exists = 'SMPL' in self.smc.keys()
        if self.smpl_exists:
            self.smpl_num_frames = self.smc['SMPL'].attrs['num_frame']
            self.smpl_created_time = self.smc['SMPL'].attrs['created_time']

            # initialize body model
            if isinstance(body_model, torch.nn.Module):
                self.body_model = body_model
            elif isinstance(body_model, dict):
                self.body_model = build_body_model(body_model)
            else:
                # in most cases, SMCReader is instantiated for image reading
                # only. Hence, it is wasteful to initialize a body model until
                # really needed in get_smpl()
                self.body_model = None
                self.default_body_model_config = dict(
                    type='SMPL',
                    gender='neutral',
                    num_betas=10,
                    keypoint_src='smpl_45',
                    keypoint_dst='smpl_45',
                    model_path='data/body_models/smpl',
                    batch_size=1,
                )

    def get_kinect_color_extrinsics(self, kinect_id, homogeneous=True):
        """Get extrinsics(cam2world) of a kinect RGB camera by kinect id.

        Args:
            kinect_id (int):
                ID of a kinect, starts from 0.
            homogeneous (bool, optional):
                If true, returns rotation and translation in
                one 4x4 matrix. Defaults to True.

        Returns:
            homogeneous is True
                ndarray: A 4x4 matrix of rotation and translation(cam2world).
            homogeneous is False
                dict: A dict of rotation and translation,
                        keys are R and T,
                        each value is an ndarray.
        """
        R = np.asarray(self.calibration_dict[str(kinect_id * 2)]['R']).reshape(
            3, 3)
        T = np.asarray(self.calibration_dict[str(kinect_id *
                                                 2)]['T']).reshape(3)
        if homogeneous:
            extrinsics = np.identity(4, dtype=float)
            extrinsics[:3, :3] = R
            extrinsics[:3, 3] = T
            return extrinsics
        else:
            return {'R': R, 'T': T}

    @property
    def calibration_dict(self):
        """Get the dict of calibration.

        Returns:
            dict:
                A dict of calibrated extrinsics.
        """
        if self.__calibration_dict__ is not None:
            return self.__calibration_dict__
        else:
            return json.loads(self.smc['Extrinsics'][()])

    def get_kinect_depth_extrinsics(self, kinect_id, homogeneous=True):
        """Get extrinsics(cam2world) of a kinect depth camera by kinect id.

        Args:
            kinect_id (int):
                ID of a kinect, starts from 0.
            homogeneous (bool, optional):
                If true, returns rotation and translation in
                one 4x4 matrix. Defaults to True.

        Returns:
            homogeneous is True
                ndarray: A 4x4 matrix of rotation and translation(cam2world).
            homogeneous is False
                dict: A dict of rotation and translation,
                        keys are R and T,
                        each value is an ndarray.
        """
        R = np.asarray(self.calibration_dict[str(kinect_id * 2 +
                                                 1)]['R']).reshape(3, 3)
        T = np.asarray(self.calibration_dict[str(kinect_id * 2 +
                                                 1)]['T']).reshape(3)
        if homogeneous:
            extrinsics = np.identity(4, dtype=float)
            extrinsics[:3, :3] = R
            extrinsics[:3, 3] = T
            return extrinsics
        else:
            return {'R': R, 'T': T}

    def get_kinect_color_intrinsics(self, kinect_id):
        """Get intrinsics of a kinect RGB camera by kinect id.

        Args:
            kinect_id (int):
                ID of a kinect, starts from 0.

        Returns:
            ndarray: A 3x3 matrix.
        """
        kinect_dict = self.smc['Kinect'][str(kinect_id)]
        intrinsics = \
            kinect_dict['Calibration']['Color']['Intrinsics'][()]
        cx, cy, fx, fy = intrinsics[:4]
        intrinsics = \
            np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return intrinsics

    def get_kinect_color_resolution(self, kinect_id):
        """Get resolution of a kinect RGB camera by kinect id.

        Args:
            kinect_id (int):
                ID of a kinect, starts from 0.

        Returns:
            ndarray:
                An ndarray of (width, height), shape=[2, ].
        """
        kinect_dict = self.smc['Kinect'][str(kinect_id)]
        resolution = \
            kinect_dict['Calibration']['Color']['Resolution'][()]
        return resolution

    def get_kinect_depth_resolution(self, kinect_id):
        """Get resolution of a kinect depth camera by kinect id.

        Args:
            kinect_id (int):
                ID of a kinect, starts from 0.

        Returns:
            ndarray:
                An ndarray of (width, height), shape=[2, ].
        """
        kinect_dict = self.smc['Kinect'][str(kinect_id)]
        resolution = \
            kinect_dict['Calibration']['Depth']['Resolution'][()]
        return resolution

    def get_kinect_depth_intrinsics(self, kinect_id):
        """Get intrinsics of a kinect depth camera by kinect id.

        Args:
            kinect_id (int):
                ID of a kinect, starts from 0.

        Returns:
            ndarray: A 3x3 matrix.
        """
        kinect_dict = self.smc['Kinect'][str(kinect_id)]
        intrinsics = \
            kinect_dict['Calibration']['Depth']['Intrinsics'][()]
        cx, cy, fx, fy = intrinsics[:4]
        intrinsics = \
            np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return intrinsics

    def get_iphone_intrinsics(self, iphone_id=0, frame_id=0, vertical=True):
        """Get intrinsics of an iPhone RGB camera by iPhone id.

        Args:
            iphone_id (int, optional):
                ID of an iPhone, starts from 0.
                Defaults to 0.
            frame_id (int, optional):
                int: frame id of one selected frame
                Defaults to 0.
            vertical (bool, optional):
                iPhone assumes landscape orientation
                if True, convert data to vertical orientation
                Defaults to True.

        Returns:
            ndarray: A 3x3 matrix.
        """
        camera_info = self.smc['iPhone'][str(iphone_id)]['CameraInfo'][str(
            frame_id)]
        camera_info = json.loads(camera_info[()])
        intrinsics = np.asarray(camera_info['cameraIntrinsics']).transpose()

        # Intrinsics have to be adjusted to achieve rotation
        #   1. swapping fx, fy
        #   2. cx -> image height - cy; cy -> cx
        if vertical:
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
            W, H = self.get_iphone_color_resolution(vertical=False)
            intrinsics = np.eye(3)
            intrinsics[0, 0], intrinsics[1, 1] = fy, fx
            intrinsics[0, 2], intrinsics[1, 2] = H - cy, cx

        return intrinsics

    def get_iphone_extrinsics(self,
                              iphone_id=0,
                              homogeneous=True,
                              vertical=True):
        """Get extrinsics(cam2world) of an iPhone RGB camera by iPhone id.

        Args:
            iphone_id (int, optional):
                ID of an iPhone, starts from 0.
                Defaults to 0.
            homogeneous (bool, optional):
                If true, returns rotation and translation in
                one 4x4 matrix. Defaults to True.
            vertical (bool, optional):
                iPhone assumes landscape orientation
                if True, convert data to vertical orientation
                Defaults to True.

        Returns:
            homogeneous is True
                ndarray: A 4x4 transformation matrix(cam2world).
            homogeneous is False
                dict: A dict of rotation and translation,
                    keys are R and T,
                    each value is an ndarray.
        """
        if iphone_id != 0:
            raise KeyError('Currently only one iPhone.')
        R = np.asarray(self.calibration_dict['iPhone']['R']).reshape(3, 3)
        T = np.asarray(self.calibration_dict['iPhone']['T']).reshape(3)

        # cam2world
        extrinsics = np.identity(4, dtype=float)
        extrinsics[:3, :3] = R
        extrinsics[:3, 3] = T

        # Extrinsics have to be adjusted to achieve rotation
        # A rotation matrix is applied on the extrinsics
        if vertical:
            # 90-degree clockwise rotation around z-axis
            R = np.eye(4)
            R[:2, :2] = np.array([[0, -1], [1, 0]])
            # Note the extrinsics is cam2world
            # world2cam_adjusted = R @ world2cam
            # => cam2world_adjusted = cam2world @ inv(R)
            extrinsics = extrinsics @ np.linalg.inv(R)
            R = extrinsics[:3, :3]
            T = extrinsics[:3, 3]

        if homogeneous:
            return extrinsics
        else:
            return {'R': R, 'T': T}

    def get_iphone_color_resolution(self, iphone_id=0, vertical=True):
        """Get color image resolution of an iPhone RGB camera by iPhone id.

        Args:
            iphone_id (int, optional):
                ID of an iPhone, starts from 0.
                Defaults to 0.
            vertical (bool, optional):
                iPhone assumes landscape orientation
                if True, convert data to vertical orientation
                Defaults to True.

        Returns:
            ndarray:get_iphone_keypoints2d
                An ndarray of (width, height), shape=[2, ].
        """
        if iphone_id != 0:
            raise KeyError('Currently only one iPhone.')
        if vertical:
            W_horizontal, H_horizontal = self.iphone_color_resolution
            W_vertical, H_vertical = H_horizontal, W_horizontal
            return np.array([W_vertical, H_vertical])
        else:
            return self.iphone_color_resolution

    def get_kinect_color(self, kinect_id, frame_id=None, disable_tqdm=True):
        """Get several frames captured by a kinect RGB camera.

        Args:
            kinect_id (int):
                ID of a kinect, starts from 0.
            frame_id (int, list or None, optional):
                int: frame id of one selected frame
                list: a list of frame id
                None: all frames will be returned
                Defaults to None.
            disable_tqdm (bool, optional):
                Whether to disable the entire progressbar wrapper.
                Defaults to True.

        Returns:
            ndarray:
                An ndarray in shape [frame_number, height, width, channels].
        """
        frames = []
        if frame_id is None:
            frame_list = range(self.get_kinect_num_frames())
        elif isinstance(frame_id, list):
            frame_list = frame_id
        elif isinstance(frame_id, int):
            assert frame_id < self.get_kinect_num_frames(),\
                'Index out of range...'
            frame_list = [frame_id]
        else:
            raise TypeError('frame_id should be int, list or None.')
        for i in tqdm.tqdm(frame_list, disable=disable_tqdm):
            frames.append(
                self.__read_color_from_bytes__(
                    self.smc['Kinect'][str(kinect_id)]['Color'][str(i)][()]))
        return np.stack(frames, axis=0)

    def get_kinect_rgbd(self,
                        kinect_id,
                        frame_id,
                        mode='color2depth',
                        threshold=0):
        if mode == 'color2depth':
            mapped_color = \
                self.__map_color_to_depth__(
                    kinect_id, frame_id, threshold=threshold
                )
            depth = self.get_kinect_depth(kinect_id, frame_id)[0]
            return mapped_color, depth
        else:
            print('Model {} is not supported...'.format(mode))

    def get_kinect_depth(self, kinect_id, frame_id=None, disable_tqdm=True):
        """Get several frames captured by a kinect depth camera.

        Args:
            kinect_id (int):
                ID of a kinect, starts from 0.
            frame_id (int, list or None, optional):
                int: frame id of one selected frame
                list: a list of frame id
                None: all frames will be returned
                Defaults to None.
            disable_tqdm (bool, optional):
                Whether to disable the entire progressbar wrapper.
                Defaults to True.

        Returns:
            ndarray:
                An ndarray in shape [frame_number, height, width, channels].
        """
        frames = []
        frame_list = []
        if frame_id is None or type(frame_id) == list:
            frame_list = range(self.get_kinect_num_frames())
            if frame_id:
                frame_list = frame_id
        else:
            assert frame_id < self.get_kinect_num_frames(),\
                'Index out of range...'
            frame_list.append(frame_id)
        for i in tqdm.tqdm(frame_list, disable=disable_tqdm):
            frames.append(
                self.smc['Kinect'][str(kinect_id)]['Depth'][str(i)][()])
        return np.stack(frames, axis=0)

    def __read_color_from_bytes__(self, color_array):
        """Decode an RGB image from an encoded byte array."""
        return cv2.cvtColor(
            cv2.imdecode(color_array, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    def get_num_kinect(self):
        """Get the number of Kinect devices.

        Returns:
            int:
                Number of Kinect devices.
        """
        return self.num_kinects

    def get_kinect_num_frames(self):
        """Get the number of frames recorded by one Kinect RGB camera.

        Returns:
            int:
                Number of frames.
        """
        return self.kinect_num_frames

    def get_iphone_num_frames(self):
        """Get the number of frames recorded by one iPhone RGB camera.

        Returns:
            int:
                Number of frames.
        """
        return self.iphone_num_frames

    def get_depth_mask(self, device_id, frame_id):
        return self.smc['Kinect'][str(device_id)]['Mask'][str(frame_id)][()]

    def get_kinect_mask(self, device_id, frame_id):
        kinect_dict = self.smc['Kinect'][str(device_id)]
        return kinect_dict['Mask_k4abt'][str(frame_id)][()]

    def get_num_iphone(self):
        """Get the number of iPhone devices.

        Returns:
            int:
                Number of iPhone devices.
        """
        return self.num_iphones

    def get_iphone_color(self,
                         iphone_id=0,
                         frame_id=None,
                         disable_tqdm=True,
                         vertical=True):
        """Get several frames captured by an iPhone RGB camera.

        Args:
            iphone_id (int):
                ID of an iPhone, starts from 0.
            frame_id (int, list or None, optional):
                int: frame id of one selected frame
                list: a list of frame id
                None: all frames will be returned
                Defaults to None.
            disable_tqdm (bool, optional):
                Whether to disable the entire progressbar wrapper.
                Defaults to True.
            vertical (bool, optional):
                iPhone assumes horizontal orientation
                if True, convert data to vertical orientation
                Defaults to True.

        Returns:
            frames:
                An ndarray in shape [frame_number, height, width, channels].
        """
        frames = []
        if frame_id is None:
            frame_list = range(self.get_iphone_num_frames())
        elif isinstance(frame_id, list):
            frame_list = frame_id
        elif isinstance(frame_id, int):
            assert frame_id < self.get_iphone_num_frames(),\
                'Index out of range...'
            frame_list = [frame_id]
        else:
            raise TypeError('frame_id should be int, list or None.')
        for i in tqdm.tqdm(frame_list, disable=disable_tqdm):
            frame = self.__read_color_from_bytes__(
                self.smc['iPhone'][str(iphone_id)]['Color'][str(i)][()])
            if vertical:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frames.append(frame)
        return np.stack(frames, axis=0)

    def get_iphone_depth(self,
                         iphone_id=0,
                         frame_id=None,
                         disable_tqdm=True,
                         vertical=True):
        """Get several frames captured by an iPhone RGB camera.

        Args:
            iphone_id (int):
                ID of an iPhone, starts from 0.
            frame_id (int, list or None, optional):
                int: frame id of one selected frame
                list: a list of frame id
                None: all frames will be returned
                Defaults to None.
            disable_tqdm (bool, optional):
                Whether to disable the entire progressbar wrapper.
                Defaults to True.
            vertical (bool, optional):
                iPhone assumes horizontal orientation
                if True, convert data to vertical orientation
                Defaults to True.

        Returns:
            frames:
                An ndarray in shape [frame_number, height, width, channels].
        """
        frames = []
        if frame_id is None:
            frame_list = range(self.get_iphone_num_frames())
        elif isinstance(frame_id, list):
            frame_list = frame_id
        elif isinstance(frame_id, int):
            assert frame_id < self.get_iphone_num_frames(),\
                'Index out of range...'
            frame_list = [frame_id]
        else:
            raise TypeError('frame_id should be int, list or None.')
        for i in tqdm.tqdm(frame_list, disable=disable_tqdm):
            frame = self.smc['iPhone'][str(iphone_id)]['Depth'][str(i)][()]
            if vertical:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frames.append(frame)
        return np.stack(frames, axis=0)

    def get_kinect_transformation_depth_to_color(self, device_id):
        """Get transformation matrix from depth to color from a single kinect.

        Args:
            kinect_id (int, optional):
                ID of a Kinect, starts from 0.

        Returns:
            ndarray: A 4x4 transformation matrix.
        """
        return np.linalg.inv(self.get_kinect_color_extrinsics(
            device_id)) @ self.get_kinect_depth_extrinsics(device_id)

    def get_kinect_transformation_color_to_depth(self, device_id):
        """Get transformation matrix from color to depth from a single kinect.

        Args:
            kinect_id (int, optional):
                ID of a Kinect, starts from 0.

        Returns:
            ndarray: A 4x4 transformation matrix.
        """
        return np.linalg.inv(self.get_kinect_depth_extrinsics(
            device_id)) @ self.get_kinect_color_extrinsics(device_id)

    def __map_color_to_depth__(self, device_id, frame_id, threshold=100):
        color_image = self.get_kinect_color(device_id, frame_id)[0]
        depth_image = self.get_kinect_depth(device_id, frame_id)[0]
        color_intrinsic = self.get_kinect_color_intrinsics(device_id)
        depth_intrinsic = self.get_kinect_depth_intrinsics(device_id)

        mask = self.get_depth_mask(device_id, frame_id)

        Td2c = self.get_kinect_transformation_depth_to_color(device_id)

        colidx = np.arange(depth_image.shape[1])
        rowidx = np.arange(depth_image.shape[0])
        colidx_map, rowidx_map = np.meshgrid(colidx, rowidx)
        col_indices = colidx_map[mask >= threshold]
        row_indices = rowidx_map[mask >= threshold]

        homo_padding = \
            np.ones((col_indices.shape[0], 1), dtype=np.float32)
        homo_indices = \
            np.concatenate(
                (col_indices[..., None], row_indices[..., None], homo_padding),
                axis=1
            )

        depth_intrinsic_inv = np.linalg.inv(depth_intrinsic)
        normalized_points = \
            depth_intrinsic_inv[None, ...] @ homo_indices[..., None]

        z_values = (depth_image / 1000)[mask >= threshold]
        valid_points = \
            normalized_points.squeeze() * z_values[..., None]

        R = Td2c[:3, :3]
        T = Td2c[:3, 3]
        valid_points = \
            R[None, ...] @ valid_points[..., None] + T[None, ..., None]
        valid_uvs = \
            color_intrinsic[None, ...] @\
            valid_points / valid_points[:, 2][..., None]
        valid_uvs = np.int32(valid_uvs.squeeze()[..., :2] + 0.5)
        valid_uvs[:, 0] = np.clip(valid_uvs[:, 0], 0, color_image.shape[1] - 1)
        valid_uvs[:, 1] = np.clip(valid_uvs[:, 1], 0, color_image.shape[0] - 1)
        mapped_color = np.ones((depth_image.shape[0], depth_image.shape[1], 3),
                               dtype=np.uint8) * 255
        mapped_color[mask >= threshold] = \
            color_image[valid_uvs[:, 1], valid_uvs[:, 0]]

        if threshold == 1:
            return valid_uvs
        return mapped_color

    def get_kinect_skeleton_3d(self, device_id, frame_id):
        """Get the 3D skeleton key points from a certain kinect.

        Args:
            device_id (int):
                ID of a kinect, starts from 0.

        Returns:
            list:
                A list with 3D keypoints
        """
        kinect_dict = self.smc['Kinect'][str(device_id)]
        return json.loads(kinect_dict['Skeleton_k4abt'][str(frame_id)][()])

    def get_depth_floor(self, device_id: int) -> dict:
        """Get the floor plane defined by a normal vector and a center point
        from a certain kinect.

        Args:
            device_id (int):
                ID of a kinect, starts from 0.

        Raises:
            KeyError:
                Key 'floor' not in ID of a kinect.

        Returns:
            dict:
                A dict with 'center', 'normal' and 'pnum'.
        """
        device_dict = self.calibration_dict[str(device_id * 2 + 1)]
        if 'floor' in device_dict:
            return device_dict['floor']
        else:
            raise KeyError(f'Kinect {device_id} has no floor data.')

    def get_keypoints2d(self, device, device_id, frame_id=None, vertical=True):
        """Get keypoints2d projected from keypoints3d.

        Args:
            device (str):
                Device name, should be Kinect or iPhone.
            device_id (int):
                ID of a device, starts from 0.
            frame_id (int, list or None, optional):
                int: frame id of one selected frame
                list: a list of frame id
                None: all frames will be returned
                Defaults to None.
            vertical (bool, optional):
                Only applicable to iPhone as device
                iPhone assumes horizontal orientation
                if True, convert data to vertical orientation
                Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                keypoints2d (N, J, 3) and its mask (J, )
        """
        assert device in {
            'Kinect', 'iPhone'
        }, f'Undefined device: {device}, should be "Kinect" or "iPhone"'
        assert device_id >= 0

        kps2d_dict = self.smc['Keypoints2D'][device][str(device_id)]
        keypoints2d = kps2d_dict['keypoints2d'][...]
        keypoints2d_mask = kps2d_dict['keypoints2d_mask'][...]

        if frame_id is None:
            frame_list = range(self.get_keypoints_num_frames())
        elif isinstance(frame_id, list):
            frame_list = frame_id
        elif isinstance(frame_id, int):
            assert frame_id < self.get_keypoints_num_frames(),\
                'Index out of range...'
            frame_list = [frame_id]
        else:
            raise TypeError('frame_id should be int, list or None.')

        keypoints2d = keypoints2d[frame_list, ...]

        if device == 'iPhone' and vertical:
            # rotate keypoints 2D clockwise by 90 degrees
            W, H = self.get_iphone_color_resolution(vertical=False)
            xs, ys, conf = \
                keypoints2d[..., 0], keypoints2d[..., 1], keypoints2d[..., 2]
            xs, ys = H - ys, xs  # horizontal -> vertical
            keypoints2d[..., 0], keypoints2d[..., 1] = xs.copy(), ys.copy()
            keypoints2d[conf == 0.0] = 0.0

        return keypoints2d, keypoints2d_mask

    def get_kinect_keypoints2d(self, device_id, frame_id=None):
        """Get Kinect 2D keypoints.

        Args:
            device_id (int):
                ID of Kinect, starts from 0.
            frame_id (int, list or None, optional):
                int: frame id of one selected frame
                list: a list of frame id
                None: all frames will be returned
                Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                keypoints2d (N, J, 3) and its mask (J, )
        """
        assert self.num_kinects > device_id >= 0
        return self.get_keypoints2d('Kinect', device_id, frame_id)

    def get_iphone_keypoints2d(self,
                               device_id=0,
                               frame_id=None,
                               vertical=True):
        """Get iPhone 2D keypoints.

        Args:
            device_id (int):
                ID of iPhone, starts from 0.
            frame_id (int, list or None, optional):
                int: frame id of one selected frame
                list: a list of frame id
                None: all frames will be returned
                Defaults to None.
            vertical (bool, optional):
                iPhone assumes horizontal orientation
                if True, convert data to vertical orientation
                Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                keypoints2d (N, J, 3) and its mask (J, )
        """
        assert device_id >= 0
        return self.get_keypoints2d(
            'iPhone', device_id, frame_id, vertical=vertical)

    def get_color(self,
                  device,
                  device_id,
                  frame_id=None,
                  disable_tqdm=True,
                  vertical=True):
        """Get RGB image(s) from Kinect RGB or iPhone RGB camera.

        Args:
            device (str):
                Device name, should be Kinect or iPhone.
            device_id (int):
                Device ID, starts from 0.
            frame_id (int, list or None, optional):
                int: frame id of one selected frame
                list: a list of frame id
                None: all frames will be returned
                Defaults to None.
            disable_tqdm (bool, optional):
                Whether to disable the entire progressbar wrapper.
                Defaults to True.
            vertical (bool, optional):
                Only applicable to iPhone as device
                iPhone assumes horizontal orientation
                if True, convert data to vertical orientation
                Defaults to True.

        Returns:
            img (ndarray):
                An ndarray in shape [frame_number, height, width, channels].
        """

        assert device in {
            'Kinect', 'iPhone'
        }, f'Undefined device: {device}, should be "Kinect" or "iPhone"'

        if device == 'Kinect':
            img = self.get_kinect_color(device_id, frame_id, disable_tqdm)
        else:
            img = self.get_iphone_color(
                device_id, frame_id, disable_tqdm, vertical=vertical)

        return img

    def get_keypoints_num_frames(self):
        return self.keypoints_num_frames

    def get_keypoints_convention(self):
        return self.keypoints_convention

    def get_keypoints_created_time(self):
        return self.keypoints_created_time

    def get_keypoints3d(self,
                        device=None,
                        device_id=None,
                        frame_id=None,
                        vertical=True):
        """Get keypoints3d (world coordinate) computed by mocap processing
        pipeline.

        Args:
            device (str):
                Device name, should be Kinect or iPhone.
                None: world coordinate
                Defaults to None.
            device_id (int):
                ID of a device, starts from 0.
                None: world coordinate
                Defaults to None
            frame_id (int, list or None, optional):
                int: frame id of one selected frame
                list: a list of frame id
                None: all frames will be returned
                Defaults to None.
            vertical (bool, optional):
                Only applicable to iPhone as device
                iPhone assumes horizontal orientation
                if True, convert data to vertical orientation
                Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                keypoints3d (N, J, 4) and its mask (J, )
        """
        assert (device is None and device_id is None) or \
            (device is not None and device_id is not None), \
            'device and device_id should be both None or both not None.'
        if device is not None:
            assert device in {
                'Kinect', 'iPhone'
            }, f'Undefined device: {device}, should be "Kinect" or "iPhone"'
        if device_id is not None:
            assert device_id >= 0

        if frame_id is None:
            frame_list = range(self.get_keypoints_num_frames())
        elif isinstance(frame_id, list):
            frame_list = frame_id
        elif isinstance(frame_id, int):
            assert frame_id < self.get_keypoints_num_frames(),\
                'Index out of range...'
            frame_list = [frame_id]
        else:
            raise TypeError('frame_id should be int, list or None.')

        kps3d_dict = self.smc['Keypoints3D']

        # keypoints3d are in world coordinate system
        keypoints3d_world = kps3d_dict['keypoints3d'][...]
        keypoints3d_world = keypoints3d_world[frame_list, ...]
        keypoints3d_mask = kps3d_dict['keypoints3d_mask'][...]

        # return keypoints3d in world coordinate system
        if device is None:
            return keypoints3d_world, keypoints3d_mask

        # return keypoints3d in device coordinate system
        else:
            if device == 'Kinect':
                cam2world = self.get_kinect_color_extrinsics(
                    kinect_id=device_id, homogeneous=True)
            else:
                cam2world = self.get_iphone_extrinsics(
                    iphone_id=device_id, vertical=vertical)

            xyz, conf = keypoints3d_world[..., :3], keypoints3d_world[..., [3]]
            xyz_homogeneous = np.ones([*xyz.shape[:-1], 4])
            xyz_homogeneous[..., :3] = xyz
            world2cam = np.linalg.inv(cam2world)
            keypoints3d = np.einsum('ij,kmj->kmi', world2cam, xyz_homogeneous)
            keypoints3d = np.concatenate([keypoints3d[..., :3], conf], axis=-1)

            return keypoints3d, keypoints3d_mask

    def get_smpl_num_frames(self):
        return self.smpl_num_frames

    def get_smpl_created_time(self):
        return self.smpl_created_time

    def get_smpl(self,
                 device=None,
                 device_id=None,
                 frame_id=None,
                 vertical=True):
        """Get SMPL (world coordinate) computed by mocap processing pipeline.

        Args:
            device (str):
                Device name, should be Kinect or iPhone.
                None: world coordinate
                Defaults to None.
            device_id (int):
                ID of a device, starts from 0.
                None: world coordinate
                Defaults to None
            frame_id (int, list or None, optional):
                int: frame id of one selected frame
                list: a list of frame id
                None: all frames will be returned
                Defaults to None.
            vertical (bool, optional):
                Only applicable to iPhone as device
                iPhone assumes horizontal orientation
                if True, convert data to vertical orientation
                Defaults to True.

        Returns:
            dict:
                'global_orient': np.ndarray of shape (N, 3)
                'body_pose': np.ndarray of shape (N, 69)
                'transl': np.ndarray of shape (N, 3)
                'betas': np.ndarray of shape (N, 10)
        """
        smpl_dict = self.smc['SMPL']
        global_orient = smpl_dict['global_orient'][...]
        body_pose = smpl_dict['body_pose'][...]
        transl = smpl_dict['transl'][...]
        betas = smpl_dict['betas'][...]

        if frame_id is None:
            frame_list = range(self.get_smpl_num_frames())
        elif isinstance(frame_id, list):
            frame_list = frame_id
        elif isinstance(frame_id, int):
            assert frame_id < self.get_keypoints_num_frames(),\
                'Index out of range...'
            frame_list = [frame_id]
        else:
            raise TypeError('frame_id should be int, list or None.')

        body_pose = body_pose[frame_list, ...]
        global_orient = global_orient[frame_list, ...]
        transl = transl[frame_list, ...]

        # return SMPL parameters in world coordinate system
        if device is None:
            smpl_dict = dict(
                global_orient=global_orient,
                body_pose=body_pose,
                transl=transl,
                betas=betas)

            return smpl_dict

        # return SMPL parameters in device coordinate system
        else:

            if self.body_model is None:
                self.body_model = \
                    build_body_model(self.default_body_model_config)
            torch_device = self.body_model.global_orient.device

            assert device in {
                'Kinect', 'iPhone'
            }, f'Undefined device: {device}, should be "Kinect" or "iPhone"'
            assert device_id >= 0

            if device == 'Kinect':
                T_cam2world = self.get_kinect_color_extrinsics(
                    kinect_id=device_id, homogeneous=True)
            else:
                T_cam2world = self.get_iphone_extrinsics(
                    iphone_id=device_id, vertical=vertical)

            T_world2cam = np.linalg.inv(T_cam2world)

            output = self.body_model(
                global_orient=torch.tensor(global_orient, device=torch_device),
                body_pose=torch.tensor(body_pose, device=torch_device),
                transl=torch.tensor(transl, device=torch_device),
                betas=torch.tensor(betas, device=torch_device))
            joints = output['joints'].detach().cpu().numpy()
            pelvis = joints[:, 0, :]

            new_global_orient, new_transl = batch_transform_to_camera_frame(
                global_orient=global_orient,
                transl=transl,
                pelvis=pelvis,
                extrinsic=T_world2cam)

            smpl_dict = dict(
                global_orient=new_global_orient,
                body_pose=body_pose,
                transl=new_transl,
                betas=betas)

            return smpl_dict
