import numpy as np
import pyrender
import torch
import trimesh


class PyRenderer(object):

    def __init__(self,
                 resolution,
                 focal_length=5000.,
                 camera_center=[112., 112.],
                 **kwargs):
        # self.renderer = pyrender.OffscreenRenderer(height, width)
        self.renderer = pyrender.OffscreenRenderer(resolution[0],
                                                   resolution[1])
        self.camera_center = np.array(camera_center)
        self.focal_length = focal_length
        self.colors = [
            (.7, .7, .6, 1.),
            (.7, .5, .5, 1.),  # Pink
            (.5, .5, .7, 1.),  # Blue
            (.5, .55, .3, 1.),  # capsule
            (.3, .5, .55, 1.),  # Yellow
        ]

    def __call__(self, verts, faces, colors=None, camera_pose=None, **kwargs):

        if isinstance(faces, torch.Tensor):
            verts = verts.detach().cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.detach().cpu().numpy()

        # Need to flip x-axis
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])

        num_people = verts.shape[0]

        # Create a scene for each image and render all meshes
        scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        # Create camera. Camera will always be at [0,0,0]
        if camera_pose is None:
            camera_pose = np.eye(4)

        camera = pyrender.camera.IntrinsicsCamera(
            fx=self.focal_length,
            fy=self.focal_length,
            cx=self.camera_center[0],
            cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)
        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.5)
        # for every person in the scene
        for n in range(num_people):
            mesh = trimesh.Trimesh(verts[n], faces[n])
            mesh.apply_transform(rot)
            trans = np.array([0, 0, 0])
            if colors is None:
                mesh_color = self.colors[
                    0]  # self.colors[n % len(self.colors)]
            else:
                mesh_color = colors[n % len(colors)]
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                alphaMode='OPAQUE',
                baseColorFactor=mesh_color)
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            scene.add(mesh, 'mesh')

            # Use 3 directional lights
            light_pose = np.eye(4)
            light_pose[:3, 3] = np.array([0, -1, 1]) + trans
            scene.add(light, pose=light_pose)
            light_pose[:3, 3] = np.array([0, 1, 1]) + trans
            scene.add(light, pose=light_pose)
            light_pose[:3, 3] = np.array([1, 1, 2]) + trans
            scene.add(light, pose=light_pose)
        # Alpha channel was not working previously need to check again
        # Until this is fixed use hack with depth image to get the opacity
        color, rend_depth = self.renderer.render(
            scene, flags=pyrender.RenderFlags.RGBA)
        return color

    def delete(self):
        self.renderer.delete()
