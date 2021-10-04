from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class RenderDataset(Dataset):

    def __init__(self,
                 vertices: Union[np.ndarray, torch.Tensor],
                 K: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 R: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 T: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 img_format: str = '%06d.png',
                 images: Optional[Union[np.ndarray, torch.Tensor]] = None):
        """Prepare the render dataset as for function `render_smpl`.

        Args:
            vertices (Union[np.ndarray, torch.Tensor]):
                vertices to be renderer. Shape could be (num_frames, num_verts,
                3) or (num_frames, num_person, num_verts, 3).
                Required.
            K (Optional[Union[np.ndarray, torch.Tensor]], optional):
                Intrinsic matrix of cameras.
                Shape should be (num_frames, 4, 4).
                Could use (1, 4, 4) to make a fixed intrinsic matrix.
                If `K` is `None`, will use default FovPerspectiveCameras.
                Defaults to None.
            R (Optional[Union[np.ndarray, torch.Tensor]], optional):
                Extrinsic Rotation matrix of cameras.
                Shape should be (num_frames, 4, 4).
                Could use (1, 4, 4) to make a fixed extrinsic rotation.
                If `R` is `None`, will use torch.eyes by default.
                Defaults to None.
            T (Optional[Union[np.ndarray, torch.Tensor]], optional):
                Extrinsic Translation matrix of cameras.
                Shape should be (num_frames, 3).
                Could use (1, 3) to make a fixed extrinsic translation.
                If `T` is `None`, will use torch.zeroes by default.
                Defaults to None.
            img_format (str, optional):
                Output image format, set to serve the functions in
                `ffmpeg_utils`.
                Defaults to '%06d.png'.
            images (Optional[Union[np.ndarray, torch.Tensor]], optional):
                Background images.
                Defaults to None.
        """
        super(RenderDataset, self).__init__()
        self.num_frames = vertices.shape[0]
        self.len = self.num_frames
        self.img_format = img_format
        if images is not None:
            self.images = torch.from_numpy(images.astype(np.float32)) \
                if isinstance(images, np.ndarray) else images
            self.with_origin_image = True
        else:
            self.images = None
            self.with_origin_image = False
        self.vertices = torch.from_numpy(vertices.astype(
            np.float32)) if isinstance(vertices, np.ndarray) else vertices
        self.K = torch.from_numpy(K.astype(np.float32)) if isinstance(
            K, np.ndarray) else K
        self.R = torch.from_numpy(R.astype(np.float32)) if isinstance(
            R, np.ndarray) else R
        self.T = torch.from_numpy(T.astype(np.float32)) if isinstance(
            T, np.ndarray) else T

    def __getitem__(self, index):
        result_dict = {
            'vertices': self.vertices[index],
            'file_names': self.img_format % (index),
        }
        if self.with_origin_image:
            result_dict.update({'images': self.images[index]})
        if self.K is not None:
            if self.K.shape[0] == self.num_frames:
                result_dict.update({'K': self.K[index]})

            else:
                result_dict.update({'K': self.K[0]})
        if self.R is not None:
            if self.R.shape[0] == self.num_frames:
                result_dict.update({'R': self.R[index]})

            else:
                result_dict.update({'R': self.R[0]})
        if self.T is not None:
            if self.T.shape[0] == self.num_frames:
                result_dict.update({'T': self.T[index]})

            else:
                result_dict.update({'T': self.T[0]})
        return result_dict

    def __len__(self):
        return self.len
