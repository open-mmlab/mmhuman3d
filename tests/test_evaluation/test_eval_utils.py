import numpy as np
import pytest

from mmhuman3d.core.evaluation import (
    fg_vertices_to_mesh_distance,
    keypoint_3d_auc,
    keypoint_3d_pck,
    keypoint_accel_error,
    keypoint_mpjpe,
    vertice_pve,
)


def test_accel_error():
    target = np.random.rand(10, 5, 3)
    output = np.copy(target)
    mask = np.ones((output.shape[0]), dtype=bool)

    error = keypoint_accel_error(output, target, mask)
    np.testing.assert_almost_equal(error, 0)

    error = keypoint_accel_error(output, target)
    np.testing.assert_almost_equal(error, 0)


def test_keypoinyt_mpjpe():
    target = np.random.rand(2, 5, 3)
    output = np.copy(target)
    mask = np.ones((output.shape[0], output.shape[1]), dtype=bool)
    with pytest.raises(ValueError):
        _ = keypoint_mpjpe(output, target, mask, alignment='norm')

    error = keypoint_mpjpe(output, target, mask, alignment='none')
    np.testing.assert_almost_equal(error, 0)

    error = keypoint_mpjpe(output, target, mask, alignment='scale')
    np.testing.assert_almost_equal(error, 0)

    error = keypoint_mpjpe(output, target, mask, alignment='procrustes')
    np.testing.assert_almost_equal(error, 0)

    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    output = np.dot(target, R)
    error = keypoint_mpjpe(output, target, mask, alignment='none')
    assert error > 1e-10

    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    output = np.dot(target, R)
    error = keypoint_mpjpe(output, target, mask, alignment='procrustes')
    np.testing.assert_almost_equal(error, 0)


def test_keypoinyt_pve():
    target = np.random.rand(2, 6890, 3)
    output = np.copy(target)

    error = vertice_pve(output, target)
    np.testing.assert_almost_equal(error, 0)

    error = vertice_pve(output, target, alignment='scale')
    np.testing.assert_almost_equal(error, 0)

    error = vertice_pve(output, target, alignment='procrustes')
    np.testing.assert_almost_equal(error, 0)

    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    output = np.dot(target, R)

    error = vertice_pve(output, target, alignment='none')
    assert error > 1e-10

    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    output = np.dot(target, R)
    error = vertice_pve(output, target, alignment='procrustes')
    np.testing.assert_almost_equal(error, 0)


def test_keypoint_3d_pck():
    target = np.random.rand(2, 5, 3) * 1000
    output = np.copy(target)
    mask = np.ones((output.shape[0], output.shape[1]), dtype=bool)

    with pytest.raises(ValueError):
        _ = keypoint_3d_pck(output, target, mask, alignment='norm')

    pck = keypoint_3d_pck(output, target, mask, alignment='none')
    np.testing.assert_almost_equal(pck, 100)

    output[0, 0, :] = target[0, 0, :] + 1000
    pck = keypoint_3d_pck(output, target, mask, alignment='none')
    np.testing.assert_almost_equal(pck, 90, 5)

    output = target * 2
    pck = keypoint_3d_pck(output, target, mask, alignment='scale')
    np.testing.assert_almost_equal(pck, 100)

    output = target + 2
    pck = keypoint_3d_pck(output, target, mask, alignment='procrustes')
    np.testing.assert_almost_equal(pck, 100)


def test_keypoint_3d_auc():
    target = np.random.rand(2, 5, 3) * 1000
    output = np.copy(target)
    mask = np.ones((output.shape[0], output.shape[1]), dtype=bool)

    with pytest.raises(ValueError):
        _ = keypoint_3d_auc(output, target, mask, alignment='norm')

    auc = keypoint_3d_auc(output, target, mask, alignment='none')
    np.testing.assert_almost_equal(auc, 30 / 31 * 100)

    output = target * 2
    auc = keypoint_3d_auc(output, target, mask, alignment='scale')
    np.testing.assert_almost_equal(auc, 30 / 31 * 100)

    output = target + 2000
    auc = keypoint_3d_auc(output, target, mask, alignment='procrustes')
    np.testing.assert_almost_equal(auc, 30 / 31 * 100)


def test_fg_vertices_to_mesh_distance():
    target = np.random.rand(10, 3) * 1000
    output = np.copy(target)
    target_points = target[:7]
    output_points = np.copy(target_points)
    faces = np.array([[i, i + 1, (i + 2) % 10] for i in range(0, 9, 2)])
    error = fg_vertices_to_mesh_distance(target, target_points, output, faces,
                                         output_points)
    np.testing.assert_almost_equal(error, 0)
