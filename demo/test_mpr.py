import minimal_pytorch_rasterizer as mpr
import numpy as np
import torch
import mmcv
import cv2

from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.builder import build_body_model
from mmhuman3d.utils.demo_utils import conver_verts_to_cam_coord

def colormap_z(z, vmin=None, vmax=None, verbose_vlims=False, inverse_cmap=True, v_pad=.5, percentile=1):
    vmin, vmax = get_vlims(z, vmin=vmin, vmax=vmax, v_pad=v_pad, percentile=percentile, verbose_vlims=verbose_vlims)

    z_vis = deepcopy(z)
    if inverse_cmap:
        z_vis[z == 0] = vmax
        cm = matplotlib.cm.viridis_r
    else:
        z_vis[z == 0] = vmin
        cm = matplotlib.cm.viridis

    z_vis = (z_vis - vmin) / (vmax - vmin)

    colored = cm(z_vis)[..., :3]

    return colored

class VisualizerMeshSMPL:
    def __init__(
        self,
        device=None,
        body_models=None,
        # size=1024,
        # f_scale=2,
        z=5,
        normals=True,
        sides=False,
        # smplx_wrapper=None,
        kinect_bones=None,
        focal_length=5000.,
        camera_center=[112.,112.],
        resolution=None,
        # K=None,
        scale=None
    ):
        # self.size = size
        # self.f_scale = f_scale
        self.z = z
        self.normals = normals
        self.sides = sides
        self.no_dist = True
        # if K is not None:
        #     assert isinstance(size, dict)
        #     if scale is None:
        #         scale = 1
        #     self.pinhole2d = mpr.Pinhole2D(
        #         fx=scale * K[0, 0], fy=scale * K[1, 1],
        #         cx=scale * K[0, 2], cy=scale * K[1, 2],
        #         h=int(scale * size['h']), w=int(scale * size['w'])
        #     )
        #     self.no_dist = True
        # else:
        #     self.pinhole2d = mpr.Pinhole2D(
        #         fx=self.f_scale * self.size, fy=self.f_scale * self.size,
        #         cx=self.size // 2, cy=self.size // 2,
        #         h=self.size, w=self.size
        #     )
        self.body_models = body_models
        self.pinhole2d = mpr.Pinhole2D(
                fx=focal_length, 
                fy=focal_length,
                cx=camera_center[0],
                cy=camera_center[1],
                h=1920,
                w=1080)

        # if smplx_wrapper is not None:
        #     self.smplx_wrapper = smplx_wrapper
        # else:
        #     assert body_models_dp is not None
        #     assert device is not None
        #     self.smplx_wrapper = SMPLXWrapper(body_models_dp, device)

        self.device = torch.device(device)
        self.faces = self.body_models.faces_tensor.to(
            dtype=torch.int32,
            device=self.device
        )


    def vertices2vis(self, vertices):
        assert vertices.device == self.faces.device

        if self.sides:
            n, m = 3, 3
            coords = [
                ([0, 0, 0], -0.5, (1, 1)),
                ([0, np.pi / 3, 0], -0.5, (1, 2)),
                ([0, -np.pi / 3, 0], -0.5, (1, 0)),
                ([np.pi / 3, 0, 0], -0.3, (0, 1)),
                ([-np.pi / 3, 0, 0], -0.3, (2, 1)),
            ]
        else:
            n, m = 1, 1
            coords = [
                ([0, 0, 0], -0.5, (0, 0)),
            ]

        h, w = self.pinhole2d.h * n, self.pinhole2d.w * m
        result = np.zeros((h, w, 3), dtype=np.uint8)

        for rvec, t_up, (ax_i, ax_j) in coords:
            if self.no_dist:
                vertices_transformed = vertices
            else:
                R = torch.tensor(
                    cv2.Rodrigues(np.array(rvec, dtype=np.float32))[0],
                    dtype=torch.float32, device=self.device
                )
                t = torch.tensor([[0, -t_up, self.z]], dtype=torch.float32, device=self.device)

                vertices_transformed = vertices @ R.T + t

            if self.normals:
                coords, normals = mpr.estimate_normals(
                    vertices=vertices_transformed,
                    faces=self.faces,
                    pinhole=self.pinhole2d
                )
                vis = mpr.vis_normals(coords, normals)
                vis = cv2.merge((vis, vis, vis))  # convert gray to 3 channel img
            else:
                z_buffer = mpr.project_mesh(
                    vertices=vertices_transformed,
                    faces=self.faces,
                    vertice_values=vertices_transformed[:, [2]],
                    pinhole=self.pinhole2d
                )
                z_buffer = z_buffer[:, :, 0].cpu().numpy()
                vis = colormap_z(z_buffer, percentile=1)
                vis = (vis * 255).round().clip(0, 255).astype(np.uint8)[..., :3]

            result[
                ax_i * self.pinhole2d.h: (ax_i + 1) * self.pinhole2d.h,
                ax_j * self.pinhole2d.w: (ax_j + 1) * self.pinhole2d.w
            ] = vis

        return result

    def get_vis(
            self,
            body_pose, gender='male',
            betas=np.zeros(10),
            rvec=np.array((np.pi, 0, 0), dtype=np.float32),
            tvec=np.zeros(3, dtype=np.float32)):
        model_output = self.smplx_wrapper.get_output(
            gender=gender,
            betas=betas,
            body_pose=body_pose,
            rvec=rvec,
            tvec=tvec
        )
        vertices = model_output.vertices.detach()[0].contiguous()
        return self.vertices2vis(vertices)




if __name__ == '__main__':

    device = 'cuda'
    focal_length = 5000.
    result_path = 'data/demo_result/inference_result.npz'
    data = HumanData.fromfile(result_path)
    pred_cams = data['pred_cams']
    bboxes_xy = data['bboxes_xyxy']
    image = mmcv.imread('data/demo_result/' + data['image_path'][0])
    verts, K = conver_verts_to_cam_coord(
        data['verts'], pred_cams, bboxes_xy, focal_length=5000.)
    verts = torch.tensor(verts[0], dtype=torch.float32).to(device = device)

    body_model = build_body_model(
            dict(type='SMPL', model_path='data/body_models/smpl'))
    faces = torch.tensor(body_model.faces.astype(np.float32))[None].to(device)

    # mpr.Pinhole2D(fx=focal_length,fy=focal_length,cx=112.,cy=112.,w=1080,h=1920)
    vis_smpl = VisualizerMeshSMPL(device='cuda',body_models=body_model)
    import time
    T = time.time()
    for i in range(100):
        img = vis_smpl.vertices2vis(verts)
    print(time.time()-T)
    cv2.imwrite('./test.png', img)

