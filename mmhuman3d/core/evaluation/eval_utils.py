# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from .mesh_eval import compute_similarity_transform
import pymesh
from math import sqrt

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


def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Code from: https://stackoverflow.com/a/18927641.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform

def fg_vertices_to_mesh_distance(groundtruth_vertices, grundtruth_landmark_points, predicted_mesh_vertices,
                                       predicted_mesh_faces,
                                       predicted_mesh_landmark_points):
    """
    This script computes the reconstruction error between an input mesh and a ground truth mesh.
    :param groundtruth_vertices: An n x 3 numpy array of vertices from a ground truth scan.
    :param grundtruth_landmark_points: A 7 x 3 list with annotations of the ground truth scan.
    :param predicted_mesh_vertices: An m x 3 numpy array of vertices from a predicted mesh.
    :param predicted_mesh_faces: A k x 3 numpy array of vertex indices composing the predicted mesh.
    :param predicted_mesh_landmark_points: A 7 x 3 list containing the annotated 3D point locations in the predicted mesh.
    :return: A list of distances (errors), one for each vertex in the groundtruth mesh, and the associated vertex index in the ground truth scan.

    The grundtruth_landmark_points and predicted_mesh_landmark_points have to contain points in the following order:
    (1) right eye outer corner, (2) right eye inner corner, (3) left eye inner corner, (4) left eye outer corner,
    (5) nose bottom, (6) right mouth corner, (7) left mouth corner.
    """
    
    # Do procrustes based on the 7 points:
    # The ground truth scan is in mm, so by aligning the prediction to the ground truth, we get meaningful units.
    d, Z, tform = procrustes(np.array(grundtruth_landmark_points), np.array(predicted_mesh_landmark_points),
                             scaling=True, reflection='best')
    # Use tform to transform all vertices in predicted_mesh_vertices to the ground truth reference space:
    predicted_mesh_vertices_aligned = []
    for v in predicted_mesh_vertices:
        s = tform['scale']
        R = tform['rotation']
        t = tform['translation']
        transformed_vertex = s * np.dot(v, R) + t
        predicted_mesh_vertices_aligned.append(transformed_vertex)
    
    # Compute the mask: A circular area around the center of the face. Take the nose-bottom and go upwards a bit:
    nose_bottom = np.array(grundtruth_landmark_points[4])
    nose_bridge = (np.array(grundtruth_landmark_points[1]) + np.array(
        grundtruth_landmark_points[2])) / 2  # between the inner eye corners
    face_centre = nose_bottom + 0.3 * (nose_bridge - nose_bottom)
    # Compute the radius for the face mask:
    outer_eye_dist = np.linalg.norm(np.array(grundtruth_landmark_points[0]) - np.array(grundtruth_landmark_points[3]))
    nose_dist = np.linalg.norm(nose_bridge - nose_bottom)
    mask_radius = 1.2 * (outer_eye_dist + nose_dist) / 2

    # Find all the vertex indices in the ground truth scan that lie within the mask area:
    vertex_indices_mask = []  # vertex indices in the source mesh (the ground truth scan)
    points_on_groundtruth_scan_to_measure_from = []
    for vertex_idx, vertex in enumerate(groundtruth_vertices):
        dist = np.linalg.norm(vertex - face_centre) # We use Euclidean distance for the mask area for now.
        if dist <= mask_radius:
            vertex_indices_mask.append(vertex_idx)
            points_on_groundtruth_scan_to_measure_from.append(vertex)
    assert len(vertex_indices_mask) == len(points_on_groundtruth_scan_to_measure_from)
    # For each vertex on the ground truth mesh, find the closest point on the surface of the predicted mesh:
    predicted_mesh_pymesh = pymesh.meshio.form_mesh(np.array(predicted_mesh_vertices_aligned), predicted_mesh_faces)
    squared_distances, face_indices, closest_points = pymesh.distance_to_mesh(predicted_mesh_pymesh,
                                                                              points_on_groundtruth_scan_to_measure_from)
    distances = [sqrt(d2) for d2 in squared_distances]
    return np.mean(np.array(distances))