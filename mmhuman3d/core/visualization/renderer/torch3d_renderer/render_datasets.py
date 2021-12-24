import numpy as np
import torch
from torch.utils.data import Dataset


class RenderDataset(Dataset):
    """Render dataset for smpl renderer."""

    def __init__(self, vertices, **kwargs):
        """Prepare the render dataset as for function `render_smpl`.

        Args:
            vertices (Union[np.ndarray, torch.Tensor]):
                vertices to be renderer. Shape could be (num_frames, num_verts,
                3) or (num_frames, num_person, num_verts, 3).
                Required.
            K (Optional[Union[np.ndarray, torch.Tensor]], optional):
                Intrinsic matrix of cameras.
                Shape should be (num_frames, 4, 4).
                Could use (1, 4, 4) to make a fixed intrinsic matrix.pose_dict.
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
            images (Optional[Union[np.ndarray, torch.Tensor]], optional):
                Background images.
                Defaults to None.

            joints (Optional[Union[np.ndarray, torch.Tensor]], optional):
                Shape should be (num_frames, num_kps, 3)
                Defaults to None.
            joints_gt (Optional[Union[np.ndarray, torch.Tensor]], optional):
                Shape should be (num_frames, num_kps, 3)
                Defaults to None.
        """
        super(RenderDataset, self).__init__()
        self.len = vertices.shape[0]
        kwargs['vertices'] = vertices
        required_keys = [
            'vertices',
            'K',
            'R',
            'T',
            'images',
            'joints',
            'joints_gt',
            'faces',
        ]
        self.vars = []
        for k in required_keys:
            v = kwargs.get(k, None)
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)

            if v is not None:
                self.vars.append(k)
                setattr(self, k, v)

    def __getitem__(self, index):
        """clip the index and get item."""
        result_dict = {
            'indexes': index,
        }

        for k in self.vars:
            v = getattr(self, k)
            idx = min(len(v) - 1, index)
            result_dict.update({k: v[idx].to(torch.float32)})

        return result_dict

    def __len__(self):
        """get length."""
        return self.len
