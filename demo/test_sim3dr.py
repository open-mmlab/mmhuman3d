import numpy as np
import torch
import mmcv
import cv2
import scipy.sparse

from mmhuman3d.core.visualization.sim3drender import Sim3DR
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.builder import build_body_model
from mmhuman3d.utils.demo_utils import conver_verts_to_cam_coord

device = 'cuda'

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
downsampling = None
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

renderer = Sim3DR()

import time
T = time.time()
for i in range(100):
    image=np.zeros_like(image)
    img = renderer(verts.cpu().numpy(),faces.cpu().numpy().astype(np.int32),image)

print(time.time() - T)