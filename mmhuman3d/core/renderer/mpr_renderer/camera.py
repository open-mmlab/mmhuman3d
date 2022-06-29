import numpy as np
import torch


class Pinhole2D:

    def __init__(self, K=None, fx=None, fy=None, cx=None, cy=None, h=0, w=0):
        if K is not None:
            assert fx is None and fy is None and cx is None and cy is None
            self.fx = K[0, 0]
            self.fy = K[1, 1]
            self.cx = K[0, 2]
            self.cy = K[1, 2]
        else:
            assert \
                fx is not None and fy is not None and \
                cx is not None and cy is not None
            self.fx = fx
            self.fy = fy
            self.cx = cx
            self.cy = cy
        self.h = h
        self.w = w

    def get_K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy],
                         [0, 0, 1]])

    def project_ndc(self, vertices, eps=1e-9):
        """
        vertices: torch.Tensor of shape (N, 3), 3 stands for xyz
        """
        assert isinstance(vertices, torch.Tensor)
        assert len(vertices.shape) == 2
        assert vertices.shape[1] == 3
        K = torch.tensor(
            self.get_K(), device=vertices.device, dtype=vertices.dtype)

        # apply intrinsics
        vertices_ndc = vertices @ K.transpose(1, 0)

        # divide xy by z, leave z unchanged
        vertices_ndc[:, [0, 1]] /= vertices_ndc[:, [2]] + eps

        # convert x from [0, w) to [-1, 1] range
        # convert y from [0, h) to [-1, 1] range
        wh = torch.tensor([self.w, self.h],
                          device=vertices.device,
                          dtype=vertices.dtype).unsqueeze(0)
        vertices_ndc[:, [0, 1]] = 2 * vertices_ndc[:, [0, 1]] / wh - 1

        return vertices_ndc
