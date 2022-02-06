from mmhuman3d.core.cameras.builder import build_cameras
from mmhuman3d.core.visualization.renderer import build_renderer
import torch
import torch.nn as nn
from mmcv.runner import build_optimizer
from mmhuman3d.models.builder import REGISTRANTS, build_body_model, build_loss
from .smplify import OptimizableParameters, SMPLify
from pytorch3d.structures import Meshes
from functools import partial
from typing import List, Union, Tuple, Iterable
from mmhuman3d.core.cameras import NewAttributeCameras
import torch.nn.functional as F
from tqdm import trange
import cv2
import matplotlib.pyplot as plt
from mmhuman3d.core.conventions import convert_kps
from pytorch3d.renderer.lighting import PointLights, DirectionalLights
from mmhuman3d.utils.mesh_utils import (join_batch_meshes_as_scene,
                                        load_plys_as_meshes)
import numpy as np
from mmhuman3d.core.visualization.visualize_keypoints2d import (_CavasProducer,
                                                                visualize_kp2d)
from mmhuman3d.utils.path_utils import prepare_output_path
from pytorch3d.loss import (
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)


@REGISTRANTS.register_module(name=['Flow2Avatar', 'flow2avatar'])
class Flow2Avatar(SMPLify):

    def __init__(
            self,
            body_model: Union[dict, torch.nn.Module],
            img_res: Union[Tuple[int], int] = 1000,
            texture_res: Union[Tuple[int], int] = 1024,
            uv_res: Union[Tuple[int], int] = 512,
            select_frame: dict = None,
            optimizer: dict = None,
            num_epochs: int = 1,
            stages: List = None,
            experiment_dir: str = None,
            use_one_betas_per_video: bool = True,
            renderer_rgb: dict = None,
            renderer_silhouette: dict = None,
            renderer_flow: dict = None,
            renderer_uv: dict = None,
            verbose: bool = False,
            device=torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        self.device = device
        # initialize body model
        if isinstance(body_model, dict):
            self.body_model = build_body_model(body_model).to(self.device)
        elif isinstance(body_model, torch.nn.Module):
            self.body_model = body_model.to(self.device)
        else:
            raise TypeError(f'body_model should be either dict or '
                            f'torch.nn.Module, but got {type(body_model)}')
        self.img_res = img_res
        self.texture_res = texture_res
        self.uv_res = uv_res

        self.select_frame = select_frame
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.stages_config = stages
        self.experiment_dir = experiment_dir
        prepare_output_path(experiment_dir, path_type='dir')
        self.use_one_betas_per_video = use_one_betas_per_video

        if isinstance(renderer_rgb, dict):
            self.renderer_rgb = build_renderer(renderer_rgb).to(self.device)
        elif isinstance(renderer_rgb, nn.Modudle):
            self.renderer_rgb = renderer_rgb.to(self.device)

        if isinstance(renderer_silhouette, dict):
            self.renderer_silhouette = build_renderer(renderer_silhouette).to(
                self.device)
        elif isinstance(renderer_silhouette, nn.Module):
            self.renderer_silhouette = renderer_silhouette.to(self.device)

        if isinstance(renderer_flow, dict):
            self.renderer_flow = build_renderer(renderer_flow).to(self.device)
        elif isinstance(renderer_flow, nn.Module):
            self.renderer_flow = renderer_flow.to(self.device)

        if isinstance(renderer_uv, dict):
            self.renderer_uv = build_renderer(renderer_uv).to(self.device)
        else:
            self.renderer_uv = renderer_uv.to(self.device)

        self.verbose = verbose

    def __call__(
        self,
        image_paths: Iterable[str] = None,
        cameras: str = None,
        keypoints2d: torch.Tensor = None,
        keypoints2d_conf: torch.Tensor = None,
        init_global_orient: torch.Tensor = None,
        init_transl: torch.Tensor = None,
        init_body_pose: torch.Tensor = None,
        init_betas: torch.Tensor = None,
        init_displacement: torch.Tensor = None,
        init_background: torch.Tensor = None,
        init_texture_image: torch.Tensor = None,
        init_light: torch.Tensor = None,
        return_verts: bool = True,
        return_joints: bool = True,
        return_full_pose: bool = True,
        return_light: bool = True,
        return_mesh: bool = True,
        return_displacement: bool = True,
        return_texture: bool = True,
        return_background: bool = True,
    ) -> dict:
        image_reader = _CavasProducer(
            frame_list=image_paths,
            kp2d=keypoints2d,
            resolution=(self.img_res, self.img_res))
        num_frames = max(len(image_reader), 1)
        self.keypoints2d_conf = keypoints2d_conf

        global_orient = self._match_init_batch_size(
            init_global_orient, self.body_model.global_orient, num_frames)
        transl = self._match_init_batch_size(init_transl,
                                             self.body_model.transl,
                                             num_frames)
        body_pose = self._match_init_batch_size(init_body_pose,
                                                self.body_model.body_pose,
                                                num_frames)
        if init_betas is None and self.use_one_betas_per_video:
            betas = torch.zeros(1, self.body_model.betas.shape[-1]).to(
                self.device)
        else:
            betas = self._match_init_batch_size(init_betas,
                                                self.body_model.betas,
                                                num_frames)

        displacement = self.body_model.displacement if init_displacement is None else init_displacement

        background = torch.zeros(self.img_res, self.img_res, 3).to(
            self.device) if init_background is None else init_background

        texture_image = self.body_model.texture_image if init_texture_image is None else init_texture_image

        lights = torch.zeros(4, 3) if init_light is None else init_light

        for i in range(self.num_epochs):
            for stage_idx, stage_config in enumerate(self.stages_config):

                self._optimize_stage(
                    stage_idx=stage_idx,
                    global_orient=global_orient,
                    transl=transl,
                    body_pose=body_pose,
                    betas=betas,
                    displacement=displacement,
                    background=background,
                    texture_image=texture_image,
                    lights=lights,
                    image_reader=image_reader,
                    cameras=cameras,
                    **stage_config,
                )

        # collate results
        ret = {
            'global_orient': global_orient,
            'transl': transl,
            'body_pose': body_pose,
            'betas': betas,
        }

        if return_verts or return_joints or \
                return_full_pose:
            body_model_output = self.body_model(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                transl=transl,
                return_verts=return_verts,
                return_joints=return_joints,
                return_full_pose=return_full_pose,
                return_texture=return_texture,
                return_mesh=return_mesh)
            if return_verts:
                ret['vertices'] = body_model_output['vertices']
            if return_joints:
                ret['joints'] = body_model_output['joints']
            if return_full_pose:
                ret['full_pose'] = body_model_output['full_pose']
            if return_mesh:
                ret['meshes'] = body_model_output['meshes']
            if return_displacement:
                ret['displacement'] = displacement
            if return_texture:
                ret['texture_image'] = texture_image
            if return_background:
                ret['background'] = background
            if return_light:
                ret['lights'] = lights

        for k, v in ret.items():
            if isinstance(v, torch.Tensor):
                ret[k] = v.detach().clone()

        return ret

    def _optimize_stage(self,
                        stage_idx: int,
                        betas: torch.Tensor,
                        body_pose: torch.Tensor,
                        global_orient: torch.Tensor,
                        transl: torch.Tensor,
                        displacement: torch.Tensor,
                        texture_image: torch.Tensor,
                        background: torch.Tensor,
                        lights: DirectionalLights,
                        image_reader: object,
                        cameras: NewAttributeCameras,
                        fit_global_orient: bool = False,
                        fit_transl: bool = False,
                        fit_body_pose: bool = False,
                        fit_betas: bool = False,
                        fit_displacement: bool = False,
                        fit_background: bool = False,
                        fit_texture: bool = False,
                        fit_lights: bool = False,
                        batch_size: int = 1,
                        num_iter: int = 1,
                        plot_period: int = 100,
                        losses_config: dict = None):
        parameters = OptimizableParameters()
        parameters.set_param(fit_global_orient, global_orient)
        parameters.set_param(fit_transl, transl)
        parameters.set_param(fit_body_pose, body_pose)
        parameters.set_param(fit_betas, betas)
        parameters.set_param(fit_displacement, displacement)
        parameters.set_param(fit_texture, texture_image)
        parameters.set_param(fit_background, background)
        parameters.set_param(fit_lights, lights)
        optimizer = build_optimizer(parameters, self.optimizer)

        ############################################
        from pytorch3d.renderer import look_at_view_transform
        im = cv2.imread('/mnt/lustre/wangwenjia/programs/room.jpeg')
        im = cv2.resize(im, (self.img_res, self.img_res),
                        cv2.INTER_CUBIC).astype(np.float32) / 255.0
        B_syn = torch.Tensor(im).to(self.device)[None]
        lights = PointLights(
            device=self.device,
            location=[[0.0, 0.0, -3.0]],
            ambient_color=[[1.0, 1.0, 1.0]],
            diffuse_color=[[0, 0, 0]],
            specular_color=[[0, 0, 0]],
        )
        mesh_scan = load_plys_as_meshes(['/mnt/lustre/wangwenjia/mesh/ai.ply'],
                                        device=self.device)

        ############################################

        losses_plot = {}
        for k in losses_config:
            if self.use_loss(losses_config, k):
                losses_plot[k] = {"values": []}
        for iter_idx in trange(num_iter):
            # indexes_source, indexes_target = self._select_frames_index(
            #     batch_size=batch_size,
            #     num_frames=len(image_reader),
            #     **self.select_frame)
            if ((iter_idx % plot_period) == 0) or (iter_idx == num_iter - 1):
                plot_flag = f'stage_{stage_idx}_iter_{iter_idx}'
            else:
                plot_flag = None

            ############################################
            dist1 = 1.2 + np.random.uniform(
                low=-0.3, high=0.3, size=(batch_size, 1))
            elev1 = np.random.uniform(low=160, high=200, size=(batch_size, 1))
            azim1 = np.random.uniform(low=0, high=360, size=(batch_size, 1))
            fov1 = 90

            R1, T1 = look_at_view_transform(dist=dist1, elev=elev1, azim=azim1)

            dist2 = dist1 + np.random.uniform(
                low=-0.4, high=0.4, size=(batch_size, 1))
            elev2 = elev1 + np.random.uniform(
                low=-5, high=5, size=(batch_size, 1))
            azim2 = azim1 + np.random.uniform(
                low=-5, high=5, size=(batch_size, 1))

            R2, T2 = look_at_view_transform(dist=dist2, elev=elev2, azim=azim2)

            cameras_source = build_cameras(
                dict(
                    type='fovperspective',
                    device=self.device,
                    R=R1,
                    T=T1,
                    fov=fov1))
            cameras_target = build_cameras(
                dict(
                    type='fovperspective',
                    device=self.device,
                    R=R2,
                    T=T2,
                    fov=fov1))

            images_source = self.renderer_rgb(
                mesh_scan.extend(batch_size),
                cameras=cameras_source,
                lights=lights)
            masks = (images_source[..., 3:] > 0) * 1.0
            images_source = masks * images_source[..., :3] + (1 -
                                                              masks) * B_syn

            images_target = self.renderer_rgb(
                mesh_scan.extend(batch_size),
                cameras=cameras_target,
                lights=lights)
            masks = (images_target[..., 3:] > 0) * 1.0
            images_target = masks * images_target[..., :3] + (1 -
                                                              masks) * B_syn

            global_orient_source = global_orient.repeat(batch_size, 1)
            body_pose_source = body_pose.repeat(batch_size, 1)
            transl_source = transl.repeat(batch_size, 1)
            global_orient_target = global_orient.repeat(batch_size, 1)
            body_pose_target = body_pose.repeat(batch_size, 1)
            transl_target = transl.repeat(batch_size, 1)

            ############################################

            def closure():
                optimizer.zero_grad()
                betas_video = self._expand_betas(body_pose.shape[0], betas)

                # images_source, keypoints2d_source = image_reader[
                #     indexes_source]
                # images_target, keypoints2d_target = image_reader[
                #     indexes_target]
                # cameras_source = cameras[indexes_source]
                # cameras_target = cameras[indexes_target]

                # global_orient_source=global_orient[indexes_source]
                # body_pose_source=body_pose[indexes_source]
                # transl_source=transl[indexes_source]
                # global_orient_target=global_orient[indexes_target]
                # body_pose_target=body_pose[indexes_target]
                # transl_target=transl[indexes_target]

                loss_dict = self.evaluate(
                    betas=betas_video,
                    global_orient_source=global_orient_source,
                    body_pose_source=body_pose_source,
                    transl_source=transl_source,
                    global_orient_target=global_orient_target,
                    body_pose_target=body_pose_target,
                    transl_target=transl_target,
                    images_source=images_source,
                    images_target=images_target,
                    cameras_source=cameras_source,
                    cameras_target=cameras_target,
                    # keypoints2d_source=keypoints2d_source,
                    keypoints2d_source_conf=self.keypoints2d_conf,
                    # keypoints2d_target=keypoints2d_target,
                    keypoints2d_target_conf=self.keypoints2d_conf,
                    displacement=displacement,
                    texture_image=texture_image,
                    background=background,
                    lights=lights,
                    losses_config=losses_config,
                    plot_flag=plot_flag,
                )

                loss = loss_dict['total_loss']
                loss.backward()

                for k, l in loss_dict.items():
                    if k != 'total_loss':
                        losses_plot[k]["values"].append(
                            float(l.detach().cpu()))

            optimizer.step(closure)

        self.plot_loss(
            losses=losses_plot,
            path=f'{self.experiment_dir}/losses/stage_{stage_idx}.png')

    @staticmethod
    def _select_frames_index(batch_size: int,
                             num_frames: int,
                             temporal_successive: bool,
                             interval_range: int,
                             fix_interval: bool = True):
        if temporal_successive:
            index_0_source = np.random.randint(
                low=0, high=num_frames - batch_size + 1)

            if fix_interval:
                index_0_target = min(index_0_source + interval_range,
                                     num_frames - batch_size + 1)
            else:
                index_0_target = np.random.randint(
                    low=max(0, index_0_source - interval_range),
                    high=min(num_frames - batch_size + 1,
                             index_0_source + interval_range))
            indexes_source = np.arange(index_0_source,
                                       index_0_source + batch_size)
            indexes_target = np.arange(index_0_target,
                                       index_0_target + batch_size)
        else:
            indexes_source = np.random.randint(
                low=0, high=num_frames, size=batch_size)
            if fix_interval:
                indexes_target = indexes_source + interval_range
                indexes_target = np.clip(indexes_target, 0, num_frames)
            else:
                indexes_target = np.random.randint(
                    low=-interval_range, high=interval_range,
                    size=batch_size) + indexes_source
                indexes_target = np.clip(indexes_target, 0, num_frames)
        return indexes_source, indexes_target

    def evaluate(
        self,
        betas=None,
        global_orient_source=None,
        body_pose_source=None,
        transl_source=None,
        global_orient_target=None,
        body_pose_target=None,
        transl_target=None,
        images_source=None,
        images_target=None,
        cameras_source=None,
        cameras_target=None,
        displacement=None,
        texture_image=None,
        background=None,
        lights=None,
        keypoints2d_source=None,
        keypoints2d_source_conf=None,
        keypoints2d_target=None,
        keypoints2d_target_conf=None,
        silhouette_source=None,
        silhouette_target=None,
        losses_config=None,
        plot_flag: str = None,
    ):

        ret = {}

        body_model_output_source = self.body_model(
            global_orient=global_orient_source,
            body_pose=body_pose_source,
            transl=transl_source,
            betas=betas,
            displacement=displacement,
            texture_image=texture_image,
            return_texture=True,
            return_mesh=True)

        body_model_output_target = self.body_model(
            global_orient=global_orient_target,
            body_pose=body_pose_target,
            transl=transl_target,
            betas=betas,
            displacement=displacement,
            texture_image=texture_image,
            return_texture=True,
            return_mesh=True)

        meshes_source = body_model_output_source['meshes']
        meshes_target = body_model_output_target['meshes']

        loss_dict = self._compute_loss(
            losses_config=losses_config,
            meshes_source=meshes_source,
            meshes_target=meshes_target,
            cameras_source=cameras_source,
            cameras_target=cameras_target,
            images_source=images_source,
            images_target=images_target,
            texture_image=texture_image,
            displacement=displacement,
            background=background,
            lights=lights,
            keypoints2d_source=keypoints2d_source,
            keypoints2d_source_conf=keypoints2d_source_conf,
            keypoints2d_target=keypoints2d_target,
            keypoints2d_target_conf=keypoints2d_target_conf,
            silhouette_source=silhouette_source,
            silhouette_target=silhouette_target,
            plot_flag=plot_flag)
        ret.update(loss_dict)

        return ret

    @staticmethod
    def use_loss(loss_dict, key):
        if key in loss_dict:
            if loss_dict[key]['weight'] > 0:
                return True
            else:
                return False
        else:
            return False

    def _compute_loss(
        self,
        losses_config: dict,
        meshes_source: Meshes = None,
        meshes_target: Meshes = None,
        cameras_source: NewAttributeCameras = None,
        cameras_target: NewAttributeCameras = None,
        images_source: torch.Tensor = None,
        images_target: torch.Tensor = None,
        texture_image: torch.Tensor = None,
        displacement: torch.Tensor = None,
        background: torch.Tensor = None,
        lights: DirectionalLights = None,
        model_joints_source: torch.Tensor = None,
        model_joints_target: torch.Tensor = None,
        model_joints_mask_source: torch.Tensor = None,
        model_joints_mask_target: torch.Tensor = None,
        keypoints2d_source: torch.Tensor = None,
        keypoints2d_target: torch.Tensor = None,
        keypoints2d_source_conf: torch.Tensor = None,
        keypoints2d_target_conf: torch.Tensor = None,
        silhouette_source: torch.Tensor = None,
        silhouette_target: torch.Tensor = None,
        plot_flag: str = None,
    ):
        use_loss = partial(self.use_loss, loss_dict=losses_config)
        losses = {}
        batch_size = len(meshes_source)

        cached_data = {}

        if use_loss(key="edge"):
            loss_edge = mesh_edge_loss(meshes_source)
            loss_edge += mesh_edge_loss(meshes_target)
            losses["edge"] = loss_edge / batch_size
        if use_loss(key="normal"):
            # mesh normal consistency
            loss_normal = mesh_normal_consistency(meshes_source)
            loss_normal += mesh_normal_consistency(meshes_target)
            losses["normal"] = loss_normal / batch_size
        if use_loss(key="laplacian"):
            loss_laplacian = mesh_laplacian_smoothing(
                meshes_source, method="uniform")
            loss_laplacian += mesh_laplacian_smoothing(
                meshes_target, method="uniform")
            losses["laplacian"] = loss_laplacian / batch_size

        if use_loss(key="mse_flow_visible"):
            flow_source_to_target = self.renderer_flow(
                meshes_source=meshes_source,
                meshes_target=meshes_target,
                cameras_source=cameras_source,
                cameras_target=cameras_target)
            visible_mask_source_to_target = flow_source_to_target[..., 3:]
            optical_flow_source_to_target = flow_source_to_target[..., :2]

            wraped_source_to_target = F.grid_sample(
                images_source.permute(0, 3, 1, 2),
                optical_flow_source_to_target,
                align_corners=False).permute(0, 2, 3, 1)

            flow_target_to_source = self.renderer_flow(
                meshes_source=meshes_target,
                meshes_target=meshes_source,
                cameras_source=cameras_target,
                cameras_target=cameras_source)
            visible_mask_target_to_source = flow_target_to_source[..., 3:]
            optical_flow_target_to_source = flow_target_to_source[..., :2]

            wraped_target_to_source = F.grid_sample(
                images_target.permute(0, 3, 1, 2),
                optical_flow_target_to_source,
                align_corners=False).permute(0, 2, 3, 1)

            rendered_silhouette_target = self.renderer_silhouette(
                meshes_target, cameras=cameras_target)[..., 3:]

            rendered_silhouette_source = self.renderer_silhouette(
                meshes_source, cameras=cameras_source)[..., 3:]

            cached_data.update(
                visible_mask_target_to_source=visible_mask_target_to_source,
                visible_mask_source_to_target=visible_mask_source_to_target,
                rendered_silhouette_source=rendered_silhouette_source,
                rendered_silhouette_target=rendered_silhouette_target)

            if losses_config["mse_flow_visible"]["use_visible_mask"]:

                loss_mse_visible = (
                    ((wraped_source_to_target - images_target) *
                     visible_mask_source_to_target *
                     rendered_silhouette_target)**2).mean()
                loss_mse_visible += (
                    ((wraped_target_to_source - images_source) *
                     visible_mask_target_to_source *
                     rendered_silhouette_source)**2).mean()
            else:
                loss_mse_visible = (
                    ((wraped_source_to_target - images_target) *
                     rendered_silhouette_target)**2).mean()
                loss_mse_visible += (
                    ((wraped_target_to_source - images_source) *
                     rendered_silhouette_source)**2).mean()
            losses["mse_flow_visible"] = loss_mse_visible / batch_size

            if plot_flag is not None:
                images = [
                    images_source[0], images_target[0],
                    wraped_source_to_target[0], rendered_silhouette_target[0],
                    visible_mask_source_to_target[0],
                    wraped_source_to_target[0] * rendered_silhouette_target[0],
                    wraped_source_to_target[0] *
                    rendered_silhouette_target[0] *
                    visible_mask_source_to_target[0]
                ]

                self.plot(
                    images,
                    titles=[
                        'image_source', 'image_target', 'wraped_image_source',
                        'rendered_silhouette_target', 'visible_mask',
                        'wraped_image_source_cropped',
                        'wraped_image_source_visible'
                    ],
                    path=
                    f'{self.experiment_dir}/visible_optical_flow/{plot_flag}.png',
                    padding=[5, 5, 5, 50])

        if use_loss(key="silhouette"):
            if 'rendered_silhouette_target' not in cached_data:
                rendered_silhouette_target = self.renderer_silhouette(
                    meshes_target, cameras=cameras_target)[..., 3:]
            else:
                rendered_silhouette_target = cached_data[
                    'rendered_silhouette_target']
            if 'rendered_silhouette_source' not in cached_data:
                rendered_silhouette_source = self.renderer_silhouette(
                    meshes_source, cameras=cameras_source)[..., 3:]
            else:
                rendered_silhouette_source = cached_data[
                    'rendered_silhouette_source']

            loss_silhouette = ((silhouette_target -
                                rendered_silhouette_target)**2).mean()
            loss_silhouette += ((silhouette_source -
                                 rendered_silhouette_source)**2).mean()
            losses["silhouette"] = loss_silhouette / batch_size

            if plot_flag is not None:

                images = [
                    rendered_silhouette_source[0],
                    silhouette_source[0],
                    rendered_silhouette_target[0],
                    silhouette_target[0],
                ]
                self.plot(
                    images,
                    titles=[
                        'rendered_silhouette_source', 'silhouette_source',
                        'rendered_silhouette_target', 'silhouette_target'
                    ],
                    path=f'{self.experiment_dir}/silhouette/{plot_flag}.png',
                    padding=[5, 5, 5, 50])

        if use_loss(key="kp2d_mse_loss"):
            kp2d_target_pred = cameras_target.transform_points_screen(
                model_joints_target,
                image_size=(self.img_res, self.img_res))[..., :2]

            loss_kp2d = (((keypoints2d_target - kp2d_target_pred)**2) *
                         keypoints2d_target_conf *
                         model_joints_mask_target).mean() / self.img_res

            kp2d_source_pred = cameras_source.transform_points_screen(
                model_joints_source,
                image_size=(self.img_res, self.img_res))[..., :2]
            loss_kp2d += (((keypoints2d_source - kp2d_source_pred)**2) *
                          keypoints2d_source_conf *
                          model_joints_mask_source).mean() / self.img_res

            losses["kp2d_mse_loss"] = loss_kp2d / batch_size

            if plot_flag is not None:

                image_target = (images_target[0].detach().cpu().numpy() *
                                255).astype(np.uint8)
                image_kp2d_target = visualize_kp2d(
                    kp2d=keypoints2d_target[0],
                    image_array=image_target,
                    return_array=True,
                    data_source=self.keypoints_convention)
                image_kp2d_pred_target = visualize_kp2d(
                    kp2d=kp2d_target_pred[0],
                    image_array=image_target,
                    return_array=True,
                    data_source='')

                image_source = (images_source[0].detach().cpu().numpy() *
                                255).astype(np.uint8)
                image_kp2d_source = visualize_kp2d(
                    kp2d=keypoints2d_source[0],
                    image_array=image_source,
                    return_array=True,
                    data_source=self.keypoints_convention)
                image_kp2d_pred_source = visualize_kp2d(
                    kp2d=kp2d_source_pred[0],
                    image_array=image_source,
                    return_array=True,
                    data_source='')
                images = [
                    image_kp2d_target[0], image_kp2d_pred_target[0],
                    image_kp2d_source[0], image_kp2d_pred_source[0]
                ]

                self.plot(
                    images,
                    titles=[
                        'image_kp2d_target', 'image_kp2d_pred_target',
                        'rendered_silhouette_target', 'silhouette_target'
                    ],
                    path=f'{self.experiment_dir}/kp2d/{plot_flag}.png',
                    padding=[5, 5, 5, 50])

        if use_loss(key="texture_mse"):
            rendered_images_target = self.renderer_rgb(
                meshes_target, cameras=cameras_target, lights=lights)[..., :3]
            rendered_images_source = self.renderer_rgb(
                meshes_source, cameras=cameras_source, lights=lights)[..., :3]

            if 'rendered_silhouette_target' not in cached_data:
                rendered_silhouette_target = self.renderer_silhouette(
                    meshes_target, cameras=cameras_target)[..., 3:]
            else:
                rendered_silhouette_target = cached_data[
                    'rendered_silhouette_target']

            if 'rendered_silhouette_source' not in cached_data:
                rendered_silhouette_source = self.renderer_silhouette(
                    meshes_source, cameras=cameras_source)[..., 3:]
            else:
                rendered_silhouette_source = cached_data[
                    'rendered_silhouette_source']

            loss_texture_mse = ((rendered_images_target -
                                 images_target)**2).mean()

            loss_texture_mse += ((rendered_images_source -
                                  images_source)**2).mean()

            losses["texture_mse"] = loss_texture_mse

            if use_loss(key="texture_smooth"):
                loss_texture_smooth = 0
                for stride in losses_config["texture_smooth"]["strides"]:
                    loss_texture_smooth += (
                        (texture_image[:-stride] - texture_image[stride:])**
                        2).mean() + ((texture_image[:, :-stride] -
                                      texture_image[:, stride:])**2).mean()
                losses["texture_smooth"] = loss_texture_smooth

            if use_loss(key="texture_max"):
                max_bound = losses_config["texture_max"]["max_bound"]
                loss_texture_max = ((texture_image > max_bound) * 1.0 *
                                    (texture_image - max_bound)**2).mean()
                losses["texture_max"] = loss_texture_max

            if use_loss(key="texture_min"):
                min_bound = losses_config["texture_min"]["min_bound"]
                loss_texture_min = ((texture_image < min_bound) * 1.0 *
                                    (min_bound - texture_image)**2).mean()
                losses["texture_min"] = loss_texture_min
            if plot_flag is not None:
                self.plot(
                    [texture_image],
                    titles=['texture_image'],
                    path=f'{self.experiment_dir}/texture/{plot_flag}.png')

        if use_loss(key="mse_silhouette_background"):
            transform_source = cameras_source.get_world_to_view_transform()
            transform_target = cameras_target.get_world_to_view_transform()
            new_src_mesh_transformed = meshes_source.clone()
            new_src_mesh_transformed = new_src_mesh_transformed.update_padded(
                transform_source.compose(
                    transform_target.inverse()).transform_points(
                        new_src_mesh_transformed.verts_padded()))
            meshes_join = join_batch_meshes_as_scene(
                [meshes_source, new_src_mesh_transformed])
            silhouette_union = self.renderer_silhouette(
                meshes=meshes_join, cameras=cameras_target)[..., 3:]
            cached_data.update(silhouette_union=silhouette_union)

            loss_mse_silhouette_background = (
                ((1 - silhouette_union) *
                 (images_target - images_source))**2).mean()
            losses[
                "mse_silhouette_background"] = loss_mse_silhouette_background / batch_size

            if plot_flag is not None:
                self.plot(
                    [(1 - silhouette_union[0]) * images_target[0],
                     (1 - silhouette_union[0]) * images_source[0]],
                    titles=['images_target_cropped', 'images_source_cropped'],
                    path=
                    f'{self.experiment_dir}/silhouette_background/{plot_flag}.png'
                )
        if use_loss(key="mse_background_image"):
            if 'silhouette_union' in cached_data:
                silhouette_union = cached_data['silhouette_union']
            else:
                transform_source = cameras_source.get_world_to_view_transform()
                transform_target = cameras_target.get_world_to_view_transform()
                new_src_mesh_transformed = meshes_source.clone()
                new_src_mesh_transformed = new_src_mesh_transformed.update_padded(
                    transform_source.compose(
                        transform_target.inverse()).transform_points(
                            new_src_mesh_transformed.verts_padded()))
                meshes_join = join_batch_meshes_as_scene(
                    [meshes_source, new_src_mesh_transformed])
                silhouette_union = self.renderer_silhouette(
                    meshes=meshes_join, cameras=cameras_target)[..., 3:]

            loss_mse_bg_image = (((1 - silhouette_union) *
                                  (images_target - background))**2).mean()
            loss_mse_bg_image += (((1 - silhouette_union) *
                                   (images_source - background))**2).mean()

            losses["mse_background_image"] = loss_mse_bg_image / batch_size

            if plot_flag is not None:
                self.plot(
                    [
                        background,
                        (1 - silhouette_union[0]) * images_target[0],
                        (1 - silhouette_union[0]) * images_source[0]
                    ],
                    titles=[
                        'background', 'images_target_cropped',
                        'images_source_cropped'
                    ],
                    path=f'{self.experiment_dir}/background/{plot_flag}.png')

        if use_loss(key="displacement_smooth"):
            uv_size = losses_config['displacement_smooth'].get(
                'resolution', None)
            strides = losses_config["displacement_smooth"].get("strides", [1])
            displacement_map = self.renderer_uv(
                displacement, resolution=uv_size)
            loss_d_smooth = 0.

            for stride in strides:
                loss_d_smooth += (
                    (displacement_map[:, :-stride] -
                     displacement_map[:, stride:])**2).mean() + (
                         (displacement_map[:, :, :-stride] -
                          displacement_map[:, :, stride:])**2).mean()
            losses["displacement_smooth"] = loss_d_smooth / batch_size

            if plot_flag is not None:
                self.plot(
                    [
                        displacement_map,
                    ],
                    titles=['displacement_map'],
                    path=
                    f'{self.experiment_dir}/displacement_map/{plot_flag}.png')

        if use_loss(key="normal_smooth"):
            uv_size = losses_config['normal_smooth'].get('resolution', None)
            strides = losses_config["normal_smooth"].get("strides", [1])
            normal_map = self.renderer_uv(
                meshes_source.verts_normals_padded(), resolution=uv_size)
            loss_normal_smooth = 0.
            for stride in strides:
                loss_normal_smooth += (
                    (normal_map[:, :-stride] - normal_map[:, stride:])**
                    2).mean() + ((normal_map[:, :, :-stride] -
                                  normal_map[:, :, stride:])**2).mean()
            losses["normal_smooth"] = loss_normal_smooth / batch_size

            if plot_flag is not None:
                self.plot(
                    [
                        normal_map,
                    ],
                    titles=['normal_map'],
                    path=f'{self.experiment_dir}/normal_map/{plot_flag}.png')

        if use_loss(key="displacement_max"):
            if isinstance(losses_config["displacement_max"]["max_bound"],
                          dict):
                max_bound = torch.zeros_like(displacement).to(self.device)
                max_dict = losses_config["displacement_max"]["max_bound"]
                for k in max_dict:
                    max_bound[self.body_model.
                              body_part_segmentation[k]] = max_dict[k]
            else:
                max_bound = losses_config["displacement_max"]["max_bound"]

            loss_max = ((displacement > max_bound) * 1.0 *
                        (displacement - max_bound)**2).mean()
            losses["displacement_max"] = loss_max / batch_size

        if use_loss(key="displacement_min"):
            min_bound = losses_config["displacement_min"]["min_bound"]
            loss_min = ((displacement < min_bound) * 1.0 *
                        (min_bound - displacement)**2).mean()
            losses["displacement_min"] = loss_min / batch_size

        if self.verbose:
            msg = ''
            for loss_name, loss in losses.items():
                msg += f'{loss_name}={loss.mean().item():.6f}'
            print(msg)

        sum_loss = torch.tensor(0.0, device=self.device)
        for k, l in losses.items():
            sum_loss += l * losses_config[k]["weight"]
        losses['total_loss'] = sum_loss

        return losses

    @staticmethod
    def plot(
            images: List[Union[torch.Tensor, np.ndarray]],
            titles: List[str],
            path: str,
            bg_color: Iterable[int] = (255, 255, 255),
            resolution=(512, 512),
            padding: Iterable[int] = (0, 0, 0, 50),
    ):
        if not isinstance(images, list):
            images = [images]
        prepare_output_path(
            path, allowed_suffix=['.png', '.jpg'], path_type='file')
        H, W = resolution
        image_num = len(images)
        final_images = np.ones(
            (H + padding[-2] + padding[-1],
             W * image_num + padding[0] * 2 + padding[1] * (image_num - 1),
             3)).astype(np.uint8) * np.array(bg_color).reshape(1, 1, 3).astype(
                 np.uint8)

        for index, image in enumerate(images):
            if image.ndim == 4:
                image = image[0]
            if isinstance(image, torch.Tensor):
                image = (image - image.min()) / (image.max() - image.min())
                if image.shape[-1] == 1:
                    image = image.repeat(1, 1, 3)
                image = (image.detach().cpu().numpy() * 255).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = cv2.resize(image, (H, W))
            elif isinstance(image, np.ndarray):
                if image.shape[-1] == 1:
                    image = image.repeat(3, -1)
            final_images[padding[2]:padding[2] + H,
                         padding[0] + (index * (padding[1] + W)):padding[0] +
                         (index * (padding[1] + W)) + W] = image
            cv2.putText(final_images, titles[index],
                        (int(padding[0] + (index * (padding[1] + W)) + W / 2),
                         int(padding[2] + H + padding[3] / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        np.array([255, 0, 0]).astype(np.int32).tolist(), 1)
        cv2.imwrite(path, final_images)

    def plot_loss(self, losses, path):
        prepare_output_path(
            path, allowed_suffix=['.png', '.jpg'], path_type='file')
        fig = plt.figure(figsize=(13, 5))
        ax = fig.gca()
        for k, l in losses.items():
            ax.plot(l['values'], label=k + " loss")
        ax.legend(fontsize="16")
        ax.set_xlabel("Iteration", fontsize="16")
        ax.set_ylabel("Loss", fontsize="16")
        ax.set_title("Loss vs iterations", fontsize="16")
        fig.savefig(path)
