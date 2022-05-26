# Copyright (c) OpenMMLab. All rights reserved.

# Adapted from:
#
# https://github.com/ahmedosman/STAR/blob/master/star/pytorch/star.py
#
#
# -*- coding: utf-8 -*-
#
# LICENSE:
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os

import torch.nn as nn


class STAR(nn.Module):

    def __init__(self,
                 model_path: str,
                 gender: str = 'neutral',
                 num_betas: int = 10) -> None:
        """STAR model constructor.
        Parameters
        ----------
        model_path: str
            The path to the folder or to the file where the model
            parameters are stored
        num_betas: int, optional
            Number of shape components to use
            (default = 10).
        gender: str, optional
            Which gender to load
        """
        if gender not in ['male', 'female', 'neutral']:
            raise RuntimeError('Invalid gender! Should be one of '
                               '[\'male\', \'female\', or \'neutral\']!')
        self.gender = gender

        if os.path.isdir(model_path):
            star_path = os.path.join(model_path,
                                     'STAR_{}.npz'.format(gender.upper()))
        else:
            star_path = model_path
        if not os.path.exists(star_path):
            raise RuntimeError('Path {} does not exist!'.format(star_path))

        super(STAR, self).__init__()
