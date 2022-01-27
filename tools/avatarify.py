import os
import sys
import torch

import os
import torch
import matplotlib.pyplot as plt

import numpy as np
from tqdm.notebook import tqdm
from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.utils.mesh_utils import load_plys_as_meshes
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, save_obj

from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.renderer import TexturesUV

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform, OpenGLPerspectiveCameras, PointLights,
    DirectionalLights, AmbientLights, RasterizationSettings, MeshRenderer,
    MeshRasterizer, SoftPhongShader, SoftSilhouetteShader, SoftPhongShader,
    TexturesVertex)
from mmhuman3d.core.visualization.renderer import OpticalFlowRenderer,\
     OpticalFlowShader
import sys
import os

import cv2
import matplotlib.pyplot as plt

from mmhuman3d.models.builder import build_body_model
import torch

import cv2
import numpy as np
import random
import torch.nn.functional as F
from mmhuman3d.utils.mesh_utils import join_batch_meshes_as_scene
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer.cameras import FoVOrthographicCameras

body_model = build_body_model(
    dict(type='smpl', model_path='/mnt/lustre/share/sugar/SMPLmodels/smpl'))
pose_tensor = torch.zeros(1, 72)
pose_dict = body_model.tensor2dict(pose_tensor)
output = body_model(**pose_dict)
smpl_mesh = Meshes(
    verts=output['vertices'], faces=body_model.faces_tensor[None])

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def norm_mesh(mesh):
    verts = mesh.verts_packed()
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center)
    mesh.scale_verts_((1.0 / float(scale)))


camera = OpenGLPerspectiveCameras(
    device=device, R=torch.eye(3, 3)[None], T=torch.zeros(1, 3))

image_size = 1000

im = cv2.imread('room.jpeg')
im = cv2.resize(im, (image_size, image_size), cv2.INTER_CUBIC)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
B_target = torch.FloatTensor(im)[None].to(device) / 255.0

lights = PointLights(
    ambient_color=((0.5, 0.5, 0.5), ),
    diffuse_color=((0.3, 0.3, 0.3), ),
    specular_color=((0.2, 0.2, 0.2), ),
    location=((0, 2, 3), ),
)

renderer_rgb = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=camera,
        raster_settings=RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=False,
        )),
    shader=SoftPhongShader(device=device, cameras=camera, lights=lights))

# Silhouette renderer

sigma = 2e-6
raster_settings_silhouette = RasterizationSettings(
    image_size=image_size,
    blur_radius=np.log(1. / 2e-6 - 1.) * sigma,
    faces_per_pixel=50,
    perspective_correct=False,
)

renderer_silhouette = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=camera, raster_settings=raster_settings_silhouette),
    shader=SoftSilhouetteShader())

raster_settings_flow = RasterizationSettings(
    image_size=image_size,
    blur_radius=0.0,
    faces_per_pixel=1,
    perspective_correct=False,
)

renderer_flow = OpticalFlowRenderer(
    rasterizer=MeshRasterizer(raster_settings=raster_settings_flow),
    shader=OpticalFlowShader())


def visualize_prediction(predicted_mesh,
                         target_mesh,
                         renderer,
                         title='',
                         image=None,
                         silhouette=False):
    inds = 3 if silhouette else range(3)
    with torch.no_grad():
        predicted_images = renderer(predicted_mesh)
        target_images = renderer(target_mesh)
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(predicted_images[0, ..., inds].cpu().detach().numpy())

    plt.subplot(1, 3, 2)
    plt.imshow(target_images.cpu().detach().numpy()[0, ..., :3])

    plt.subplot(1, 3, 3)
    image = (image - image.min()) / (image.max() - image.min())
    plt.imshow(image.cpu().detach().numpy())
    plt.title(title)
    plt.axis("off")


# Plot losses as a function of optimization iteration
def plot_losses(losses):
    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()
    for k, l in losses.items():
        ax.plot(l['values'], label=k + " loss")
    ax.legend(fontsize="16")
    ax.set_xlabel("Iteration", fontsize="16")
    ax.set_ylabel("Loss", fontsize="16")
    ax.set_title("Loss vs iterations", fontsize="16")


losses = {
    "silhouette": {
        "weight": 0,
        "values": []
    },
    "edge": {
        "weight": .1,
        "values": []
    },
    "normal": {
        "weight": .01,
        "values": []
    },
    "laplacian": {
        "weight": .01,
        "values": []
    },
    "min": {
        "weight": 10,
        "values": []
    },
    "max": {
        "weight": 10,
        "values": []
    },
    "mse_background": {
        "weight": 20.0,
        "values": []
    },
    "mse_visible": {
        "weight": 20.0,
        "values": []
    },
    "kp2d_mse_loss": {
        "weight": 1e-6,
        "values": []
    },
    "d_smooth": {
        "weight": 0.,
        "values": []
    },
    "normal_smooth": {
        "weight": 0.,
        "values": []
    },
    "background": {
        "weight": .0,
        "values": []
    },
    "texture_mse": {
        "weight": 0.0,
        "values": []
    },
    "texture_smooth": {
        "weight": 0.0,
        "values": []
    },
    "texture_min": {
        "weight": 0.0,
        "values": []
    },
    "texture_max": {
        "weight": 0.0,
        "values": []
    },
}

max_bound = 0.03
min_bound = 0

Niter = 1
num_views_per_iteration = 1
plot_period = 50


def update_mesh_shape_prior_losses(mesh, loss):
    # and (b) the edge length of the predicted mesh

    loss["edge"] = mesh_edge_loss(mesh)

    # mesh normal consistency
    loss["normal"] = mesh_normal_consistency(mesh)

    # mesh laplacian smoothing
    loss["laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")


mesh_uv = load_objs_as_meshes(
    ['/mnt/lustre/wangwenjia/mesh/SMPL/SMPL_female_default_resolution.obj'],
    device=device)
origin_mesh = load_plys_as_meshes(['/mnt/lustre/wangwenjia/mesh/ai.ply'],
                                  device=device)

mesh = origin_mesh.clone()

src_mesh = load_objs_as_meshes(['/mnt/lustre/wangwenjia/mesh/ai.obj'],
                               device=device)
src_mesh = src_mesh.update_padded(src_mesh.verts_padded() -
                                  src_mesh.verts_normals_padded() * 0.006)

verts_shape = src_mesh.verts_packed().shape
src_mesh.textures = TexturesVertex(torch.zeros_like(src_mesh.verts_padded()))

deform_verts = torch.full((verts_shape[0], 1),
                          0.0,
                          device=device,
                          requires_grad=True)
texture_image = torch.full((1, 1024, 1024, 3),
                           0.0,
                           device=device,
                           requires_grad=True)
background = torch.full((image_size, image_size, 3),
                        0.0,
                        device=device,
                        requires_grad=True)

optimizer = torch.optim.SGD([deform_verts, background], lr=1.0, momentum=0.9)

faces = src_mesh.faces_padded()[0]

import cv2
import numpy as np
import random
import torch.nn.functional as F
from mmhuman3d.utils.mesh_utils import join_batch_meshes_as_scene
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer.cameras import FoVOrthographicCameras

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps

faces = src_mesh.faces_padded()[0]

loop = tqdm(range(Niter))
#     losses = losses_default.copy()
#     losses.update(stage['losses'])
#     num_views_per_iteration = stage['num_views_per_iteration']
#     plot_period = stage['plot_period']
for i in loop:
    # Initialize optimizer
    optimizer.zero_grad()

    # Deform the mesh
    new_src_mesh = src_mesh.offset_verts(deform_verts *
                                         src_mesh.verts_normals_padded()[0])
    #     new_src_mesh = src_mesh.offset_verts(deform_verts)
    new_src_mesh.textures._maps_padded = texture_image

    loss = {k: torch.tensor(0.0, device=device) for k in losses}
    update_mesh_shape_prior_losses(new_src_mesh, loss)

    # Randomly select two views to optimize over in this iteration.  Compared
    # to using just one view, this helps resolve ambiguities between updating
    # mesh shape vs. updating mesh texture
    for j in np.random.permutation(
            num_views).tolist()[:num_views_per_iteration]:

        dist1 = 1.5 + random.uniform(-0.5, 0.5)
        elev1 = random.uniform(0, 360)
        azim1 = random.uniform(150, 210)
        fov1 = 90

        R1, T1 = look_at_view_transform(dist=dist1, elev=elev1, azim=azim1)

        dist2 = dist1 + random.uniform(-0.4, 0.4)
        elev2 = elev1 + random.uniform(-5, 5)
        azim2 = azim1 + random.uniform(-5, 5)

        R2, T2 = look_at_view_transform(dist=dist2, elev=elev2, azim=azim2)

        random_camera1 = OpenGLPerspectiveCameras(
            device=device, R=R1, T=T1, fov=fov1)
        random_camera2 = OpenGLPerspectiveCameras(
            device=device, R=R2, T=T2, fov=fov1)

        rendered_images_source = renderer_rgb(
            mesh, cameras=random_camera1, lights=lights)
        masks = (rendered_images_source[..., 3:] > 0) * 1.0
        rendered_images_source = masks * rendered_images_source[..., :3] + (
            1 - masks) * B_target

        rendered_images_target = renderer_rgb(
            mesh, cameras=random_camera2, lights=lights)
        masks = (rendered_images_target[..., 3:] > 0) * 1.0
        rendered_images_target = masks * rendered_images_target[..., :3] + (
            1 - masks) * B_target

        if losses["mse_visible"]["weight"] > 0:
            rendered_flow = renderer_flow(
                meshes_source=new_src_mesh,
                meshes_target=new_src_mesh,
                cameras_source=random_camera1,
                cameras_target=random_camera2)
            optical_flow = rendered_flow[..., :2]
            visible_mask = rendered_flow[..., 3:]
            warped_images_source = F.grid_sample(
                rendered_images_source.permute(0, 3, 1, 2),
                optical_flow).permute(0, 2, 3, 1)

            silhouette_target = renderer_silhouette(
                new_src_mesh, cameras=random_camera2)[..., 3:]

            # loss_mse_visible = (((warped_images_source - rendered_images_target)*silhouette_target) **2).mean()
            loss_mse_visible = (
                ((warped_images_source - rendered_images_target) *
                 visible_mask * silhouette_target)**2).mean()
            loss["mse_visible"] += loss_mse_visible / num_views_per_iteration

            if losses["silhouette"]["weight"] > 0:
                silhouette_gt = renderer_silhouette(
                    mesh, cameras=random_camera2)[..., 3:]
                loss_silhouette = ((silhouette_target -
                                    silhouette_gt)**2).mean()
                loss["silhouette"] += loss_silhouette / num_views_per_iteration

        if losses["kp2d_mse_loss"]["weight"] > 0:
            kp2d_target = random_camera2.transform_points_screen(
                kp3d, image_size=(image_size, image_size))[..., :2]
            kp3d_pred = torch.einsum(
                'bik,ji->bjk',
                [new_src_mesh.verts_padded(), body_model.J_regressor])
            kp2d_pred = random_camera2.transform_points_screen(
                kp3d_pred, image_size=(image_size, image_size))[..., :2]
            kp2d_target_, mask = convert_kps(
                kp2d_target, src='coco_wholebody', dst='smpl')
            # mask = torch.Tensor(mask).to(device)
            mask = mask.view(1, -1, 1)
            loss_kp2d = ((
                (kp2d_pred - kp2d_target_)**2) * mask).mean() / image_size
            loss["kp2d_mse_loss"] += loss_kp2d / num_views_per_iteration

        if losses["texture_mse"]["weight"] > 0:
            images_predicted = renderer_rgb(
                new_src_mesh, cameras=random_camera2, lights=lights)
            target_rgb = renderer_rgb(
                src_mesh, cameras=random_camera2, lights=lights)[..., :3]
            #         target_rgb = target_rgbs_all[j]

            predicted_rgb = images_predicted[..., :3]

            loss_texture_mse = (((predicted_rgb - target_rgb))**2).mean()

            loss["texture_mse"] = loss_texture_mse

            if losses["texture_smooth"]["weight"]:
                loss_texture_smooth = ((predicted_rgb[:, :-1]-predicted_rgb[:, 1:])**2).mean() +\
                    ((predicted_rgb[:, :, :-1]-predicted_rgb[:,:, 1:])**2).mean()

                loss_texture_smooth += ((predicted_rgb[:, :-3]-predicted_rgb[:, 3:])**2).mean() +\
                    ((predicted_rgb[:, :, :-3]-predicted_rgb[:,:, 3:])**2).mean()
                loss["texture_smooth"] = loss_texture_smooth

            if losses["texture_max"]["weight"]:
                loss_texture_max = ((texture_image > 1) * 1.0 *
                                    (texture_image - 1)).mean()
                loss["texture_max"] = loss_texture_max

            if losses["texture_min"]["weight"]:
                loss_texture_min = ((texture_image < 0) * 1.0 *
                                    (-texture_image)).mean()
                loss["texture_min"] = loss_texture_min

        if losses["background"]["weight"] > 0 or losses["mse_background"][
                "weight"] > 0:
            transform_source = random_camera1.get_world_to_view_transform()
            transform_target = random_camera2.get_world_to_view_transform()
            new_src_mesh_transformed = new_src_mesh.clone()
            new_src_mesh_transformed = new_src_mesh_transformed.\
                            update_padded(transform_source.compose(transform_target.inverse()).\
                            transform_points(new_src_mesh_transformed.verts_padded()))
            meshes_join = join_batch_meshes_as_scene(
                [new_src_mesh, new_src_mesh_transformed])
            silhouette_union = renderer_silhouette(
                meshes_world=meshes_join, cameras=random_camera2)[..., 3:]
            loss_mse_background = (
                ((1 - silhouette_union) *
                 (rendered_images_target - rendered_images_source))**2).mean()
            loss_bg = (((1 - silhouette_union) *
                        (rendered_images_target - background))**2).mean()
            loss_bg += (((1 - silhouette_union) *
                         (rendered_images_source - background))**2).mean()
            if losses["background"]["weight"] > 0:
                loss["background"] += loss_bg / num_views_per_iteration
            if losses["mse_background"]["weight"] > 0:
                loss[
                    "mse_background"] += loss_mse_background / num_views_per_iteration

        if losses["d_smooth"]["weight"] > 0:
            fragments = renderer_flow.rasterizer(
                new_src_mesh, cameras=random_camera2)
            face_attr = deform_verts[faces]
            D = deform_verts.shape[-1]
            face_attr = face_attr.view(faces.shape[0], 3, D)
            verts_uv = torch.cat([
                mesh_uv.textures.verts_uvs_padded(),
                torch.ones(1,
                           mesh_uv.textures.verts_uvs_padded().shape[1],
                           1).to(device)
            ], -1)
            faces_uv = mesh_uv.textures.faces_uvs_padded()
            fragments = renderer_rgb.rasterizer(
                Meshes(verts=verts_uv, faces=faces_uv),
                cameras=FoVOrthographicCameras(
                    min_x=0, min_y=0, device=device))
            pix_to_face = fragments.pix_to_face
            bary_coords = fragments.bary_coords
            displacement_map = interpolate_face_attributes(
                pix_to_face=pix_to_face.to(device),
                barycentric_coords=bary_coords.to(device),
                face_attributes=face_attr.to(device),
            ).squeeze(-2).squeeze(0)
            # loss_d_smooth = ((displacement_map[:-3]-displacement_map[3:])**2).mean() +\
            #     ((displacement_map[:, :-3]-displacement_map[:, 3:])**2).mean()
            loss_d_smooth = ((displacement_map[:-1]-displacement_map[1:])**2).mean() +\
                ((displacement_map[:, :-1]-displacement_map[:, 1:])**2).mean()
            loss["d_smooth"] += loss_d_smooth / num_views_per_iteration

        if losses["normal_smooth"]["weight"] > 0:
            fragments = renderer_flow.rasterizer(
                new_src_mesh, cameras=random_camera2)
            face_attr = new_src_mesh.verts_normals_padded()[0][faces]
            face_attr = face_attr.view(faces.shape[0], 3, 3)
            verts_uv = torch.cat([
                mesh_uv.textures.verts_uvs_padded(),
                torch.ones(1,
                           mesh_uv.textures.verts_uvs_padded().shape[1],
                           1).to(device)
            ], -1)
            faces_uv = mesh_uv.textures.faces_uvs_padded()
            fragments = renderer_rgb.rasterizer(
                Meshes(verts=verts_uv, faces=faces_uv),
                cameras=FoVOrthographicCameras(
                    min_x=0, min_y=0, device=device))
            pix_to_face = fragments.pix_to_face
            bary_coords = fragments.bary_coords
            normal_map = interpolate_face_attributes(
                pix_to_face=pix_to_face.to(device),
                barycentric_coords=bary_coords.to(device),
                face_attributes=face_attr.to(device),
            ).squeeze(-2).squeeze(0)
            loss_normal_smooth = ((normal_map[:-3]-normal_map[3:])**2).mean() +\
                ((normal_map[:, :-3]-normal_map[:, 3:])**2).mean()
            loss_normal_smooth = ((normal_map[:-1]-normal_map[1:])**2).mean() +\
                ((normal_map[:, :-1]-normal_map[:, 1:])**2).mean()
            loss[
                "normal_smooth"] += loss_normal_smooth / num_views_per_iteration

        if losses["max"]["weight"] > 0:
            #             norm_square = deform_verts[:, 0:1]**2 + deform_verts[:, 1:2]**2 + deform_verts[:, 2:3]**2
            loss_max = ((deform_verts > max_bound) * 1.0 *
                        (deform_verts - max_bound)**2).mean()
            #             loss_max = ((norm_square>0.2**2)*1.0 * (norm_square-0.2**2)**2).mean()
            loss["max"] += loss_max / num_views_per_iteration

        if losses["min"]["weight"] > 0:

            loss_min = ((deform_verts < min_bound) * 1.0 *
                        (min_bound - deform_verts)**2).mean()
            loss["min"] += loss_min / num_views_per_iteration

# Weighted sum of the losses
    sum_loss = torch.tensor(0.0, device=device)
    for k, l in loss.items():
        sum_loss += l * losses[k]["weight"]
        losses[k]["values"].append(float(l.detach().cpu()))

    # Print the losses
    loop.set_description("total_loss = %.6f" % sum_loss)

    # Plot mesh
    if i % plot_period == 0:
        visualize_prediction(
            new_src_mesh,
            mesh,
            renderer=renderer_rgb,
            image=displacement_map,
            title="iter: %d" % i,
            silhouette=False)

    # Optimization step
    import time

    sum_loss.backward(retain_graph=True)
    optimizer.step()
print(time.asctime())
