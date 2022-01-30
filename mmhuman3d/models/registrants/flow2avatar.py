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
from mmhuman3d.utils.mesh_utils import join_batch_meshes_as_scene
import numpy as np
from mmhuman3d.core.visualization.visualize_keypoints2d import (_CavasProducer,
                                                                visualize_kp2d)


@REGISTRANTS.register_module(name=['Flow2Avatar', 'flow2avatar'])
class Flow2Avatar(SMPLify):

    def __init__(
            self,
            body_model: Union[dict, torch.nn.Module],
            img_res: Union[Tuple[int], int] = 1000,
            texture_res: Union[Tuple[int], int] = 1024,
            optimizer: dict = None,
            num_epochs: int = 1,
            stages: List = None,
            experiment_dir: str = None,
            renderer_rgb: dict = None,
            renderer_silhouette: dict = None,
            renderer_flow: dict = None,
            renderer_uv: dict = None,
            use_one_betas_per_video: bool = True,
            cameras_config: NewAttributeCameras = None,
            image_paths: Iterable[str] = None,
            verbose: bool = False,
            device=torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'),
    ):

        self.stages_config = stages
        self.num_epochs = num_epochs
        self.cameras = build_cameras(cameras_config)
        self.image_paths = image_paths
        self.img_res = img_res
        self.device = device
        self.texture_res = texture_res
        self.use_one_betas_per_video = use_one_betas_per_video
        self.optimizer = optimizer
        if isinstance(renderer_rgb, dict):
            self.renderer_rgb = build_renderer(renderer_rgb).to(self.device)
        elif isinstance(renderer_rgb, nn.Modudle):
            self.renderer_rgb = renderer_rgb

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
        self.experiment_dir = experiment_dir
        # initialize body model
        if isinstance(body_model, dict):
            self.body_model = build_body_model(body_model).to(self.device)
        elif isinstance(body_model, torch.nn.Module):
            self.body_model = body_model.to(self.device)
        else:
            raise TypeError(f'body_model should be either dict or '
                            f'torch.nn.Module, but got {type(body_model)}')

        self.verbose = verbose

    def __call__(
        self,
        batch_size: int,
        keypoints2d: torch.Tensor = None,
        keypoints2d_conf: torch.Tensor = None,
        # keypoints3d: torch.Tensor = None,
        # keypoints3d_conf: torch.Tensor = None,
        init_global_orient: torch.Tensor = None,
        init_transl: torch.Tensor = None,
        init_body_pose: torch.Tensor = None,
        init_betas: torch.Tensor = None,
        init_displacement: torch.Tensor = None,
        init_background: torch.Tensor = None,
        init_texture_image: torch.Tensor = None,
        init_light: torch.Tensor = None,
        return_verts: bool = False,
        return_joints: bool = False,
        return_full_pose: bool = False,
        return_light: bool = False,
        return_avatar: bool = False,
        return_smpl_mesh: bool = False,
        return_displacement: bool = False,
        return_texture: bool = False,
        return_background: bool = False,
    ) -> dict:
        self.image_reader = _CavasProducer(
            frame_list=self.image_paths,
            kp2d=keypoints2d,
            resolution=(self.img_res, self.img_res))
        self.keypoints2d_conf = keypoints2d_conf

        global_orient = self._match_init_batch_size(
            init_global_orient, self.body_model.global_orient, batch_size)
        transl = self._match_init_batch_size(init_transl,
                                             self.body_model.transl,
                                             batch_size)
        body_pose = self._match_init_batch_size(init_body_pose,
                                                self.body_model.body_pose,
                                                batch_size)
        if init_betas is None and self.use_one_betas_per_video:
            betas = torch.zeros(1, self.body_model.betas.shape[-1]).to(
                self.device)
        else:
            betas = self._match_init_batch_size(init_betas,
                                                self.body_model.betas,
                                                batch_size)

        displacement = torch.zeros(
            self.body_model.NUM_VERTS,
            1) if init_displacement is None else init_displacement

        background = torch.zeros(
            self.img_res, self.img_res,
            3) if init_background is None else init_background

        texture_image = torch.zeros(
            self.texture_res, self.texture_res,
            3) if init_texture_image is None else init_texture_image
        light = torch.zeros(4, 3) if init_light is None else init_light
        for i in range(self.num_epochs):
            for stage_idx, stage_config in enumerate(self.stage_config):

                self._optimize_stage(
                    global_orient=global_orient,
                    transl=transl,
                    body_pose=body_pose,
                    betas=betas,
                    displacement=displacement,
                    background=background,
                    texture_image=texture_image,
                    light=light,
                    stage_idx=stage_idx,
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
                return_smpl_mesh=return_smpl_mesh,
                return_avatar=return_avatar)
            if return_verts:
                ret['vertices'] = body_model_output['vertices']
            if return_joints:
                ret['joints'] = body_model_output['joints']
            if return_full_pose:
                ret['full_pose'] = body_model_output['full_pose']
            if return_avatar:
                ret['avatar'] = body_model_output['meshes']
            if return_displacement:
                ret['displacement'] = displacement
            if return_texture:
                ret['texture'] = texture_image
            if return_background:
                ret['background'] = background
            if return_light:
                ret['light'] = light

        for k, v in ret.items():
            if isinstance(v, torch.Tensor):
                ret[k] = v.detach().clone()

        return ret

    def _optimize_stage(self,
                        betas: torch.Tensor,
                        body_pose: torch.Tensor,
                        global_orient: torch.Tensor,
                        transl: torch.Tensor,
                        displacement: torch.Tensor,
                        texture_image: torch.Tensor,
                        background: torch.Tensor,
                        light: DirectionalLights,
                        stage_idx: int,
                        fit_global_orient: bool = False,
                        fit_transl: bool = False,
                        fit_body_pose: bool = False,
                        fit_betas: bool = False,
                        fit_displacement: bool = False,
                        fit_background: bool = False,
                        fit_texture: bool = False,
                        fit_light: bool = False,
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
        parameters.set_param(fit_light, light)
        optimizer = build_optimizer(parameters, self.optimizer)

        losses_plot = {}
        for k in losses_config:
            losses_plot[k] = {"values": []}
        for iter_idx in trange(num_iter):
            indexes_source, indexes_target = self.select_frames_index(
                batch_size)
            if (iter_idx % plot_period) == 0:
                plot_flag = f'stage_{stage_idx}_iter_{iter_idx}'
            else:
                plot_flag = None

            def closure():
                optimizer.zero_grad()
                betas_video = self._expand_betas(body_pose.shape[0], betas)
                images_source, keypoints2d_source = self.image_reader[
                    indexes_source]
                images_target, keypoints2d_target = self.image_reader[
                    indexes_target]

                loss_dict = self.evaluate(
                    betas=betas_video,
                    global_orient_source=global_orient[indexes_source],
                    body_pose_source=body_pose[indexes_source],
                    transl_source=transl[indexes_source],
                    global_orient_target=global_orient[indexes_target],
                    body_pose_target=body_pose[indexes_target],
                    transl_target=transl[indexes_target],
                    images_source=images_source,
                    images_target=images_target,
                    cameras_source=self.cameras[indexes_source],
                    cameras_target=self.cameras[indexes_target],
                    keypoints2d_source=keypoints2d_source,
                    keypoints2d_source_conf=self.keypoints2d_conf,
                    keypoints2d_target=keypoints2d_target,
                    keypoints2d_target_conf=self.keypoints2d_conf,
                    displacement=displacement,
                    texture_image=texture_image,
                    background=background,
                    light=light,
                    losses_config=losses_config,
                    plot_flag=plot_flag,
                )

                loss = loss_dict['total_loss']
                loss.backward()

                for k, l in loss_dict.items():
                    losses_plot[k]["values"].append(float(l.detach().cpu()))

            optimizer.step(closure)
        self.plot_loss(
            losses_plot,
            path=f'{self.experiment_dir}/losses/stage_{stage_idx}.png')

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
        light=None,
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
            return_avatar=True)

        body_model_output_target = self.body_model(
            global_orient=global_orient_target,
            body_pose=body_pose_target,
            transl=transl_target,
            betas=betas,
            displacement=displacement,
            texture_image=texture_image,
            return_avatar=True)

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
            light=light,
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
            if loss_dict['weight'] > 0:
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
        keypoints2d_conf: torch.Tensor = None,
        silhouette_source: torch.Tensor = None,
        silhouette_target: torch.Tensor = None,
        plot_flag: str = None,
    ):
        use_loss = partial(func=self.use_loss, loss_dict=losses_config)
        losses = {}
        batch_size = len(meshes_source)

        cached_data = {}

        if use_loss("mse_visible"):
            flow_source_to_target = self.renderer_flow(
                meshes_source=meshes_source,
                meshes_target=meshes_target,
                cameras_source=cameras_source,
                cameras_target=cameras_target)
            visible_mask_source_to_target = flow_source_to_target[..., 3:]
            optical_flow_source_to_target = flow_source_to_target[..., :2]

            wraped_source_to_target = F.grid_sample(
                images_source.permute(0, 3, 1, 2),
                optical_flow_source_to_target).permute(0, 2, 3, 1)

            flow_target_to_source = self.renderer_flow(
                meshes_source=meshes_target,
                meshes_target=meshes_source,
                cameras_source=cameras_target,
                cameras_target=cameras_source)
            visible_mask_target_to_source = flow_target_to_source[..., 3:]
            optical_flow_target_to_source = flow_target_to_source[..., :2]

            wraped_target_to_source = F.grid_sample(
                images_target.permute(0, 3, 1, 2),
                optical_flow_target_to_source).permute(0, 2, 3, 1)

            rendered_silhouette_target = self.renderer_silhouette(
                meshes_target, cameras=cameras_target)[..., 3:]

            rendered_silhouette_source = self.renderer_silhouette(
                meshes_source, cameras=cameras_source)[..., 3:]

            cached_data.update(
                visible_mask_target_to_source=visible_mask_target_to_source,
                visible_mask_source_to_target=visible_mask_source_to_target,
                rendered_silhouette_source=rendered_silhouette_source,
                rendered_silhouette_target=rendered_silhouette_target)

            if losses_config["mse_visible"]["use_visible_mask"]:

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
            losses["mse_visible"] += loss_mse_visible / batch_size

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
                    size=(512, 512),
                    padding=[5, 5, 5, 50])

        if use_loss("silhouette"):
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
            losses["silhouette"] += loss_silhouette / batch_size

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
                    size=(512, 512),
                    padding=[5, 5, 5, 50])

        if use_loss("kp2d_mse_loss"):
            kp2d_target_pred = cameras_target.transform_points_screen(
                model_joints_target,
                image_size=(self.img_res, self.img_res))[..., :2]

            loss_kp2d = (((keypoints2d_target - kp2d_target_pred)**2) *
                         keypoints2d_conf *
                         model_joints_mask_target).mean() / self.img_res

            kp2d_source_pred = cameras_source.transform_points_screen(
                model_joints_source,
                image_size=(self.img_res, self.img_res))[..., :2]
            loss_kp2d += (((keypoints2d_source - kp2d_source_pred)**2) *
                          keypoints2d_conf *
                          model_joints_mask_source).mean() / self.img_res

            losses["kp2d_mse_loss"] += loss_kp2d / batch_size

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
                    size=(512, 512),
                    padding=[5, 5, 5, 50])

        if use_loss("texture_mse"):
            rendered_images_target = self.renderer_rgb(
                meshes_target, cameras=cameras_target, lights=lights)
            rendered_images_source = self.renderer_rgb(
                meshes_source, cameras=cameras_source, lights=lights)

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

            loss_texture_mse = (((rendered_images_target - images_target))**2 *
                                rendered_silhouette_target).mean()

            loss_texture_mse += ((
                (rendered_images_source - images_source))**2 *
                                 rendered_silhouette_source).mean()

            losses["texture_mse"] = loss_texture_mse

            if use_loss("texture_smooth"):
                loss_texture_smooth = 0
                for stride in losses_config["texture_smooth"]["strides"]:
                    loss_texture_smooth += (
                        (texture_image[:-stride] - texture_image[stride:])**
                        2).mean() + ((texture_image[:, :-stride] -
                                      texture_image[:, stride:])**2).mean()
                losses["texture_smooth"] = loss_texture_smooth

            if use_loss("texture_max"):
                max_bound = losses_config["texture_max"]["max_bound"]
                loss_texture_max = ((texture_image > max_bound) * 1.0 *
                                    (texture_image - max_bound)).mean()
                losses["texture_max"] = loss_texture_max

            if use_loss("texture_min"):
                min_bound = losses_config["texture_min"]["min_bound"]
                loss_texture_min = ((texture_image < min_bound) * 1.0 *
                                    (min_bound - texture_image)).mean()
                losses["texture_min"] = loss_texture_min
            if plot_flag is not None:
                self.plot(
                    texture_image,
                    titles=['texture_image'],
                    path=f'{self.experiment_dir}/texture/{plot_flag}.png')

        if use_loss("mse_silhouette_background"):
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
                meshes_world=meshes_join, cameras=cameras_target)[..., 3:]
            cached_data.upddate(silhouette_union=silhouette_union)

            loss_mse_silhouette_background = (
                ((1 - silhouette_union) *
                 (images_target - images_source))**2).mean()
            losses[
                "mse_silhouette_background"] += loss_mse_silhouette_background / batch_size

            if plot_flag is not None:
                self.plot(
                    [(1 - silhouette_union[0]) * images_target[0],
                     (1 - silhouette_union[0]) * images_source[0]],
                    titles=['images_target_cropped', 'images_source_cropped'],
                    path=
                    f'{self.experiment_dir}/silhouette_background/{plot_flag}.png'
                )
        if use_loss("mse_background_image"):
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
                    meshes_world=meshes_join, cameras=cameras_target)[..., 3:]

            loss_mse_bg_image = (((1 - silhouette_union) *
                                  (images_target - background))**2).mean()
            loss_mse_bg_image += (((1 - silhouette_union) *
                                   (images_source - background))**2).mean()

            losses["mse_background_image"] += loss_mse_bg_image / batch_size

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

        if use_loss("displacement_smooth"):
            uv_size = losses_config['displacement_smooth']['resolution']
            displacement_map = self.renderer_uv(
                displacement, resolution=uv_size)
            loss_d_smooth = 0.
            for stride in losses_config["displacement_smooth"]["strides"]:
                loss_d_smooth += (
                    (displacement_map[:, :-stride] -
                     displacement_map[:, stride:])**2).mean() + (
                         (displacement_map[:, :, :-stride] -
                          displacement_map[:, :, stride:])**2).mean()
            losses["displacement_smooth"] += loss_d_smooth / batch_size

            if plot_flag is not None:
                self.plot(
                    [
                        displacement_map,
                    ],
                    titles=['displacement_map'],
                    path=
                    f'{self.experiment_dir}/displacement_map/{plot_flag}.png')

        if use_loss("normal_smooth"):
            uv_size = losses_config['normal_smooth']['resolution']
            normal_map = self.renderer_uv(
                meshes_source.verts_normals_padded(), resolution=uv_size)
            loss_normal_smooth = 0.
            for stride in losses_config["normal_smooth"]["strides"]:
                loss_normal_smooth += (
                    (normal_map[:, :-stride] - normal_map[:, stride:])**
                    2).mean() + ((normal_map[:, :, :-stride] -
                                  normal_map[:, :, stride:])**2).mean()
            losses["normal_smooth"] += loss_normal_smooth / batch_size

            if plot_flag is not None:
                self.plot(
                    [
                        normal_map,
                    ],
                    titles=['normal_map'],
                    path=f'{self.experiment_dir}/normal_map/{plot_flag}.png')

        if use_loss("displacement_max"):
            max_bound = losses_config["displacement_max"]["max_bound"]
            loss_max = ((displacement > max_bound) * 1.0 *
                        (displacement - max_bound)**2).mean()
            losses["displacement_max"] += loss_max / batch_size

        if use_loss("displacement_min"):
            min_bound = losses_config["displacement_max"]["min_bound"]
            loss_min = ((displacement < min_bound) * 1.0 *
                        (min_bound - displacement)**2).mean()
            losses["displacement_min"] += loss_min / batch_size

        if self.verbose:
            msg = ''
            for loss_name, loss in losses.items():
                msg += f'{loss_name}={loss.mean().item():.6f}'
            print(msg)

        total_loss = 0
        for k, loss in losses.items():
            if loss.ndim == 3:
                total_loss += loss.sum(dim=(2, 1)) * losses_config[k]['weight']
            elif loss.ndim == 2:
                total_loss += loss.sum(dim=-1) * losses_config[k]['weight']
            else:
                total_loss += loss * losses_config[k]['weight']
        losses['total_loss'] = total_loss

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
        H, W = resolution
        image_num = len(images)
        final_images = np.ones(
            H + padding[-2] + padding[-1],
            W * image_num + padding[0] * 2 + padding[1] * (image_num - 1),
            3).astype(np.uint8) * np.array(bg_color).reshape(1, 1, 3)

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
                         padding[0] + (index * padding[1] + W):padding[0] +
                         (index * padding[1] + W) + W] = image
            cv2.putText(final_images, titles[index],
                        (int(padding[0] + (index * padding[1] + W) + W / 2),
                         int(padding[2] + H + padding[3] / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        np.array([255, 255, 255]).astype(np.int32).tolist(), 2)

        cv2.imwrite(path, final_images)

    def plot_loss(losses, path):
        fig = plt.figure(figsize=(13, 5))
        ax = fig.gca()
        for k, l in losses.items():
            ax.plot(l['values'], label=k + " loss")
        ax.legend(fontsize="16")
        ax.set_xlabel("Iteration", fontsize="16")
        ax.set_ylabel("Loss", fontsize="16")
        ax.set_title("Loss vs iterations", fontsize="16")
        fig.savefig(path)
