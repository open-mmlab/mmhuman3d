# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from .mesh_eval import compute_similarity_transform


def keypoint_mpjpe(pred, gt, mask, alignment='none'):
    """Calculate the mean per-joint position error (MPJPE) and the error after
    rigid alignment with the ground truth (PA-MPJPE).
    batch_size: N
    num_keypoints: K
    keypoint_dims: C
    Args:
        pred (np.ndarray[N, K, C]): Predicted keypoint location.
        gt (np.ndarray[N, K, C]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        alignment (str, optional): method to align the prediction with the
            groundtruth. Supported options are:
            - ``'none'``: no alignment will be applied
            - ``'scale'``: align in the least-square sense in scale
            - ``'procrustes'``: align in the least-square sense in scale,
                rotation and translation.
    Returns:
        tuple: A tuple containing joint position errors
        - mpjpe (float|np.ndarray[N]): mean per-joint position error.
        - pa-mpjpe (float|np.ndarray[N]): mpjpe after rigid alignment with the
            ground truth
    """
    assert mask.any()

    if alignment == 'none':
        pass
    elif alignment == 'procrustes':
        pred = np.stack([
            compute_similarity_transform(pred_i, gt_i)
            for pred_i, gt_i in zip(pred, gt)
        ])
    elif alignment == 'scale':
        pred_dot_pred = np.einsum('nkc,nkc->n', pred, pred)
        pred_dot_gt = np.einsum('nkc,nkc->n', pred, gt)
        scale_factor = pred_dot_gt / pred_dot_pred
        pred = pred * scale_factor[:, None, None]
    else:
        raise ValueError(f'Invalid value for alignment: {alignment}')

    error = np.linalg.norm(pred - gt, ord=2, axis=-1)[mask].mean()

    return error


def keypoint_accel_error(gt, pred, mask=None):
    """Computes acceleration error:

    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        gt (Nx14x3).
        pred (Nx14x3).
        mask (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)x14x3
    accel_gt = gt[:-2] - 2 * gt[1:-1] + gt[2:]
    accel_pred = pred[:-2] - 2 * pred[1:-1] + pred[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    if mask is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(mask)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return np.mean(normed[new_vis], axis=1)


def vertice_pve(pred_verts, target_verts):
    """Computes per vertex error (PVE).

    Args:
        verts_gt (N x verts_num x 3).
        verts_pred (N x verts_num x 3).
    Returns:
        error_verts.
    """
    assert len(pred_verts) == len(target_verts)
    error = np.linalg.norm(pred_verts - target_verts, ord=2, axis=-1).mean()
    return error


def keypoint_3d_pck(pred, gt, mask, alignment='none', threshold=150.):
    """Calculate the Percentage of Correct Keypoints (3DPCK) w. or w/o rigid
    alignment.
    Paper ref: `Monocular 3D Human Pose Estimation In The Wild Using Improved
    CNN Supervision' 3DV'2017. <https://arxiv.org/pdf/1611.09813>`__ .
    Note:
        - batch_size: N
        - num_keypoints: K
        - keypoint_dims: C
    Args:
        pred (np.ndarray[N, K, C]): Predicted keypoint location.
        gt (np.ndarray[N, K, C]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        alignment (str, optional): method to align the prediction with the
            groundtruth. Supported options are:
            - ``'none'``: no alignment will be applied
            - ``'scale'``: align in the least-square sense in scale
            - ``'procrustes'``: align in the least-square sense in scale,
                rotation and translation.
        threshold:  If L2 distance between the prediction and the groundtruth
            is less then threshold, the predicted result is considered as
            correct. Default: 150 (mm).
    Returns:
        pck: percentage of correct keypoints.
    """
    assert mask.any()

    if alignment == 'none':
        pass
    elif alignment == 'procrustes':
        pred = np.stack([
            compute_similarity_transform(pred_i, gt_i)
            for pred_i, gt_i in zip(pred, gt)
        ])
    elif alignment == 'scale':
        pred_dot_pred = np.einsum('nkc,nkc->n', pred, pred)
        pred_dot_gt = np.einsum('nkc,nkc->n', pred, gt)
        scale_factor = pred_dot_gt / pred_dot_pred
        pred = pred * scale_factor[:, None, None]
    else:
        raise ValueError(f'Invalid value for alignment: {alignment}')

    error = np.linalg.norm(pred - gt, ord=2, axis=-1)
    pck = (error < threshold).astype(np.float32)[mask].mean() * 100

    return pck


def keypoint_3d_auc(pred, gt, mask, alignment='none'):
    """Calculate the Area Under the Curve (3DAUC) computed for a range of 3DPCK
    thresholds.
    Paper ref: `Monocular 3D Human Pose Estimation In The Wild Using Improved
    CNN Supervision' 3DV'2017. <https://arxiv.org/pdf/1611.09813>`__ .
    This implementation is derived from mpii_compute_3d_pck.m, which is
    provided as part of the MPI-INF-3DHP test data release.
    Note:
        batch_size: N
        num_keypoints: K
        keypoint_dims: C
    Args:
        pred (np.ndarray[N, K, C]): Predicted keypoint location.
        gt (np.ndarray[N, K, C]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        alignment (str, optional): method to align the prediction with the
            groundtruth. Supported options are:
            - ``'none'``: no alignment will be applied
            - ``'scale'``: align in the least-square sense in scale
            - ``'procrustes'``: align in the least-square sense in scale,
                rotation and translation.
    Returns:
        auc: AUC computed for a range of 3DPCK thresholds.
    """
    assert mask.any()

    if alignment == 'none':
        pass
    elif alignment == 'procrustes':
        pred = np.stack([
            compute_similarity_transform(pred_i, gt_i)
            for pred_i, gt_i in zip(pred, gt)
        ])
    elif alignment == 'scale':
        pred_dot_pred = np.einsum('nkc,nkc->n', pred, pred)
        pred_dot_gt = np.einsum('nkc,nkc->n', pred, gt)
        scale_factor = pred_dot_gt / pred_dot_pred
        pred = pred * scale_factor[:, None, None]
    else:
        raise ValueError(f'Invalid value for alignment: {alignment}')

    error = np.linalg.norm(pred - gt, ord=2, axis=-1)

    thresholds = np.linspace(0., 150, 31)
    pck_values = np.zeros(len(thresholds))
    for i in range(len(thresholds)):
        pck_values[i] = (error < thresholds[i]).astype(np.float32)[mask].mean()

    auc = pck_values.mean() * 100

    return auc
