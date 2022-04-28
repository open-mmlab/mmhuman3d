# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import torch

from mmhuman3d.models.builder import build_body_model

body_model_load_dir = 'data/body_models/star'


def test_star_init():
    _ = build_body_model(dict(type='STAR', model_path=body_model_load_dir))


def test_star_forward():
    star = build_body_model(dict(type='STAR', model_path=body_model_load_dir))

    betas = np.array([[
        2.25176191, -3.7883464, 0.46747496, 3.89178988, 2.20098416, 0.26102114,
        -3.07428093, 0.55708514, -3.94442258, -2.88552087
    ]])
    batch_size = 1

    # Zero pose
    poses = torch.cuda.FloatTensor(np.zeros((batch_size, 72)))
    betas = torch.cuda.FloatTensor(betas)

    trans = torch.cuda.FloatTensor(np.zeros((batch_size, 3)))
    _ = star.forward(poses, betas, trans)


if __name__ == '__main__':
    test_star_init()
    test_star_forward()
