import pyrender
import trimesh
import numpy as np
import torch
import mmcv
import cv2
import scipy.sparse

from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.builder import build_body_model
from mmhuman3d.utils.demo_utils import conver_verts_to_cam_coord
from mmhuman3d.utils.demo_utils import conver_verts_to_cam_coord

class Renderer(object):

    def __init__(self, focal_length=5000., height=224., width=224.,**kwargs):
        # self.renderer = pyrender.OffscreenRenderer(height, width)
        self.renderer = pyrender.OffscreenRenderer(1080, 1920)
        self.camera_center = np.array([width / 2., height / 2.])
        self.focal_length = focal_length
        self.colors = [
                        (.7, .7, .6, 1.),
                        (.7, .5, .5, 1.),  # Pink
                        (.5, .5, .7, 1.),  # Blue
                        (.5, .55, .3, 1.),  # capsule
                        (.3, .5, .55, 1.),  # Yellow
                    ]

    def __call__(self, verts, faces, colors=None,focal_length=None,camera_pose=None,**kwargs):
        # Need to flip x-axis
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])

        #self.renderer.viewport_height = img.shape[0]
        #self.renderer.viewport_width = img.shape[1]
        num_people = verts.shape[0]
        verts = verts.detach().cpu().numpy()
        if isinstance(faces, torch.Tensor):
        	faces = faces.detach().cpu().numpy()

        # Create a scene for each image and render all meshes
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))

        
        # Create camera. Camera will always be at [0,0,0]
        # CHECK If I need to swap x and y
        if camera_pose is None:
            camera_pose = np.eye(4)

        if focal_length is None:
            fx,fy = self.focal_length, self.focal_length
        else:
            fx,fy = focal_length, focal_length
        camera = pyrender.camera.IntrinsicsCamera(fx=fx, fy=fy,
                                                  cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)
        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.5)
        # for every person in the scene
        for n in range(num_people):
            mesh = trimesh.Trimesh(verts[n], faces[n])
            mesh.apply_transform(rot)
            trans = np.array([0,0,0])
            if colors is None:
                mesh_color = self.colors[0] #self.colors[n % len(self.colors)]
            else:
                mesh_color = colors[n % len(colors)]
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                alphaMode='OPAQUE',
                baseColorFactor=mesh_color)
            mesh = pyrender.Mesh.from_trimesh(
                mesh,
                material=material)
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
        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        return color

    def delete(self):
        self.renderer.delete()

def get_renderer(test=False,**kwargs):
    renderer = Renderer(**kwargs)
    if test:
        import cv2, pickle, os
        import torch
        from config import args
        model = pickle.load(open(os.path.join(args().smpl_model_path,'SMPL_NEUTRAL.pkl'),'rb'), encoding='latin1')
        np_v_template = torch.from_numpy(np.array(model['v_template'])).cuda().float()[None]
        face = model['f'].astype(np.int32)[None]
        np_v_template = np_v_template.repeat(2,1,1)
        np_v_template[1] += 0.3
        np_v_template[:,:,2] += 6
        result = renderer(np_v_template, face)
        cv2.imwrite('test_pyrenderer.png',(result[:,:,:3]*255).astype(np.uint8))
    return renderer

# if __name__ == '__main__':
#     # get_renderer(test=True, perps=True)
#     device = 'cuda'
#     focal_length = 5000.
#     result_path = 'data/demo_result/inference_result.npz'
#     data = HumanData.fromfile(result_path)
#     pred_cams = data['pred_cams']
#     bboxes_xy = data['bboxes_xyxy']
#     image = mmcv.imread('data/demo_result/' + data['image_path'][0])
#     # verts, K = conver_verts_to_cam_coord(
#     #     data['verts'], pred_cam, bboxes_xy, focal_length=5000.)
#     verts =data['verts']
#     cam_trans = np.concatenate([
#         pred_cams[..., [1]], pred_cams[..., [2]], 2 * focal_length /
#         (224 * pred_cams[..., [0]] + 1e-9)
#     ], -1)
#     verts = verts + cam_trans[..., None, :]
#     verts = torch.tensor(verts[0])[None]

#     w,h = image.shape[0], image.shape[1]

#     body_model = build_body_model(
#             dict(type='SMPL', model_path='data/body_models/smpl'))
#     faces = torch.tensor(body_model.faces.astype(np.float32)).to(device)[None]

#     renderer = Renderer()

#     img = renderer(verts=verts,faces=faces)
#     cv2.imwrite('test.png', img[:,:,:3])

if __name__ == '__main__':
    # get_renderer(test=True, perps=True)
    device = 'cuda'
    focal_length = 5000.
    result_path = 'data/demo_result/inference_result.npz'
    data = HumanData.fromfile(result_path)
    pred_cams = data['pred_cams']
    bboxes_xy = data['bboxes_xyxy']
    image = mmcv.imread('data/demo_result/' + data['image_path'][0])
    verts, K = conver_verts_to_cam_coord(
        data['verts'], pred_cams, bboxes_xy, focal_length=5000.)
    verts = torch.tensor(verts[0], dtype=torch.float32)[None].to(device)

    body_model = build_body_model(
            dict(type='SMPL', model_path='data/body_models/smpl'))
    faces = torch.tensor(body_model.faces.astype(np.float32))[None].to(device)
    downsampling = 2
    if downsampling is not None:
        assert downsampling == 1 or downsampling ==2, \
            f"Only support 1 or 2, but got {downsampling}."

        mesh_downsampling = np.load(
            'data/mesh_downsampling.npz', 
            allow_pickle=True, 
            encoding='latin1')

        U_ = mesh_downsampling['U'] # upsampling mat
        D_ = mesh_downsampling['D'] # downsampling mat
        F_ = mesh_downsampling['F'] # faces

        device = 'cuda'
        ptD = []
        for i in range(downsampling):
            d = scipy.sparse.coo_matrix(D_[i])
            i = torch.LongTensor(np.array([d.row, d.col]))
            v = torch.FloatTensor(d.data)
            D_mat = torch.sparse.FloatTensor(i, v, d.shape).to(device)
            verts = torch.spmm(D_mat, verts.squeeze())[None]
        faces = torch.IntTensor(F_[downsampling].astype(np.int16))[None].to(device)
    renderer = Renderer()   
    import time
    T = time.time()
    for i in range(100):
        img = renderer(verts=verts,faces=faces)
    
    print(time.time() - T)

    
    cv2.imwrite('test.png', img[:,:,:3])

    # 431 0.093
    # 1723 0.099
    # 6890 0.098