import json

import cv2
import h5py
import numpy as np
import tqdm


class SMCReader:

    def __init__(self, file_path):
        """Read SenseMocapFile endswith ".smc".

        Args:
            file_path (str):
                Path to an SMC file.
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
        if self.iphone_exists:
            self.iphone_num_frames = self.smc['iPhone'].attrs['num_frame']
            self.iphone_color_resolution = \
                self.smc['iPhone'].attrs['color_resolution']
            self.iphone_depth_resolution = \
                self.smc['iPhone'].attrs['depth_resolution']

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

    def get_iphone_intrinsics(self, iphone_id=0, frame_id=0):
        """Get intrinsics of an iPhone RGB camera by iPhone id.

        Args:
            iphone_id (int, optional):
                ID of an iPhone, starts from 0.
                Defaults to 0.

        Returns:
            ndarray: A 3x3 matrix.
        """
        camera_info = self.smc['iPhone'][str(iphone_id)]['CameraInfo'][str(
            frame_id)]
        camera_info = json.loads(camera_info[()])
        intrinsics = np.asarray(camera_info['cameraIntrinsics']).transpose()
        return intrinsics

    def get_iphone_extrinsics(self, iphone_id=0, frame_id=0):
        """Get extrinsics(cam2world) of an iPhone RGB camera by iPhone id.

        Args:
            iphone_id (int, optional):
                ID of an iPhone, starts from 0.
                Defaults to 0.

        Returns:
            ndarray: A 4x4 transformation matrix(cam2world).
        """
        R = np.asarray(self.calibration_dict['iPhone']['R']).reshape(3, 3)
        T = np.asarray(self.calibration_dict['iPhone']['T']).reshape(3)
        extrinsics = np.identity(4, dtype=float)
        extrinsics[:3, :3] = R
        extrinsics[:3, 3] = T
        return extrinsics

    def get_iphone_color_resolution(self, iphone_id=0, frame_id=0):
        return self.iphone_color_resolution

    def get_kinect_color(self, kinect_id, frame_id=None, disable_tqdm=False):
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
                Defaults to False.

        Returns:
            ndarray:
                An ndarray in shape [frame_number, height, width, channels].
        """
        frame_list = []
        frames = []
        if frame_id is None or type(frame_id) == list:
            frame_list = range(self.get_kinect_num_frames())
            if frame_id:
                frame_list = frame_id
        elif type(frame_id) == int:
            assert frame_id < self.get_kinect_num_frames(),\
                'Index out of range...'
            frame_list.append(frame_id)
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

    def get_kinect_depth(self, kinect_id, frame_id=None, disable_tqdm=False):
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
                Defaults to False.

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

    def get_iphone_color(self, iphone_id=0, frame_id=None, disable_tqdm=False):
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
                Defaults to False.

        Returns:
            ndarray:
        """
        if frame_id is None or type(frame_id) == list:
            frames = []
            frame_list = range(self.get_iphone_num_frames())
            if frame_id:
                frame_list = frame_id
            for i in tqdm.tqdm(frame_list, disable=disable_tqdm):
                frames.append(
                    self.__read_color_from_bytes__(
                        self.smc['iPhone'][str(iphone_id)]['Color'][str(i)][(
                        )]))
            return np.stack(frames, axis=0)
        else:
            assert frame_id < self.get_iphone_num_frames(),\
                'Index out of range...'
            return np.stack([
                self.__read_color_from_bytes__(self.smc['iPhone'][str(
                    iphone_id)]['Color'][str(frame_id)][()])
            ],
                            axis=0)

    def get_iphone_depth(self, iphone_id=0, frame_id=None, disable_tqdm=False):
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
                Defaults to False.

        Returns:
            ndarray:
        """
        if frame_id is None or type(frame_id) == list:
            frames = []
            frame_list = range(self.iphone_num_frames)
            if frame_id:
                frame_list = frame_id
            for i in tqdm.tqdm(frame_list, disable=disable_tqdm):
                frames.append(
                    self.smc['iPhone'][str(iphone_id)]['Depth'][str(i)][()])
            return np.stack(frames, axis=0)
        else:
            assert frame_id < self.iphone_num_frames,\
                'Index out of range...'
            return np.stack([
                self.smc['iPhone'][str(iphone_id)]['Depth'][str(frame_id)][()]
            ],
                            axis=0)

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
