import numpy as np
import pytest
import torch

from mmhuman3d.utils.transforms import (
    aa_to_ee,
    aa_to_quat,
    aa_to_rot6d,
    aa_to_rotmat,
    aa_to_sja,
    ee_to_aa,
    ee_to_quat,
    ee_to_rot6d,
    ee_to_rotmat,
    quat_to_aa,
    quat_to_ee,
    quat_to_rot6d,
    quat_to_rotmat,
    rot6d_to_aa,
    rot6d_to_ee,
    rot6d_to_quat,
    rot6d_to_rotmat,
    rotmat_to_aa,
    rotmat_to_ee,
    rotmat_to_quat,
    rotmat_to_rot6d,
    sja_to_aa,
)


def check_func(var1, var2, correct_var2_shape=None):
    bool_type = type(var1) == type(var2)
    bool_dtype = var1.dtype == var2.dtype
    if type(var1) is torch.Tensor:
        bool_grad = var1.requires_grad == var2.requires_grad
        bool_device = var1.device == var2.device
    else:
        bool_grad = True
        bool_device = True
    if correct_var2_shape:
        bool_shape = var2.shape == correct_var2_shape
    else:
        bool_shape = True
    return bool_type and bool_dtype and bool_grad and bool_device and \
        bool_shape


def test_transforms_value():
    with pytest.raises(ValueError):
        aa_to_ee(torch.zeros(3), convention='xyzz')
    with pytest.raises(ValueError):
        aa_to_ee(torch.zeros(3), convention='abc')
    with pytest.raises(ValueError):
        aa_to_ee(torch.zeros(4), convention='xyz')
    with pytest.raises(ValueError):
        ee_to_aa(torch.zeros(4), convention='xyz')
    with pytest.raises(ValueError):
        quat_to_aa(torch.zeros(5))
    with pytest.raises(ValueError):
        rot6d_to_aa(torch.zeros(4))
    with pytest.raises(ValueError):
        rotmat_to_aa(torch.zeros(4))
    with pytest.raises(ValueError):
        rotmat_to_aa(torch.zeros(3, 4))
    with pytest.raises(ValueError):
        aa_to_sja(
            torch.zeros(1, 3),
            R_t=torch.eye(3, 3).unsqueeze(0).expand(1, 3, 3),
            R_t_inv=torch.eye(3, 3).unsqueeze(0).expand(1, 3, 3))
    with pytest.raises(ValueError):
        sja_to_aa(
            torch.zeros(1, 3),
            R_t=torch.eye(3, 3).unsqueeze(0).expand(1, 3, 3),
            R_t_inv=torch.eye(3, 3).unsqueeze(0).expand(1, 3, 3))


def test_transforms_tensor():
    aa = torch.zeros(3)
    ee = torch.zeros(3)
    quat = torch.FloatTensor([1, 0, 0, 0])
    rotmat = torch.eye(3, 3)
    rot6d = rotmat.view(-1)[:6]
    aa_21 = torch.zeros(21, 3)
    sja = torch.zeros(21, 3)
    R_t = torch.eye(3, 3).unsqueeze(0).expand(21, 3, 3)
    R_t_inv = torch.eye(3, 3).unsqueeze(0).expand(21, 3, 3)
    # test ndims
    for _ in range(10):
        # test convention
        assert check_func(aa, aa_to_ee(aa, convention='xyz'),
                          aa.shape[:-1] + (3, ))
        assert check_func(aa, aa_to_ee(aa, convention='xzy'),
                          aa.shape[:-1] + (3, ))
        assert check_func(aa, aa_to_ee(aa, convention='yxz'),
                          aa.shape[:-1] + (3, ))
        assert check_func(aa, aa_to_ee(aa, convention='yzx'),
                          aa.shape[:-1] + (3, ))
        assert check_func(aa, aa_to_ee(aa, convention='zyx'),
                          aa.shape[:-1] + (3, ))
        assert check_func(aa, aa_to_ee(aa, convention='zxy'),
                          aa.shape[:-1] + (3, ))

        # test aa convention
        assert check_func(aa, aa_to_rotmat(aa), aa.shape[:-1] + (3, 3))
        assert check_func(aa, aa_to_quat(aa), aa.shape[:-1] + (4, ))
        assert check_func(aa, aa_to_rot6d(aa), aa.shape[:-1] + (6, ))
        # test ee convention
        assert check_func(ee, ee_to_aa(ee, convention='xyz'),
                          ee.shape[:-1] + (3, ))
        assert check_func(ee, ee_to_rotmat(ee, convention='xyz'),
                          ee.shape[:-1] + (3, 3))
        assert check_func(ee, ee_to_quat(ee, convention='xyz'),
                          ee.shape[:-1] + (4, ))
        assert check_func(ee, ee_to_rot6d(ee, convention='xyz'),
                          ee.shape[:-1] + (6, ))
        # test quat convention
        assert check_func(quat, quat_to_aa(quat), quat.shape[:-1] + (3, ))
        assert check_func(quat, quat_to_rotmat(quat), quat.shape[:-1] + (3, 3))
        assert check_func(quat, quat_to_ee(quat, convention='xyz'),
                          quat.shape[:-1] + (3, ))
        assert check_func(quat, quat_to_rot6d(quat), quat.shape[:-1] + (6, ))
        # test rotmat convention
        assert check_func(rotmat, rotmat_to_aa(rotmat),
                          rotmat.shape[:-2] + (3, ))
        assert check_func(rotmat, rotmat_to_ee(rotmat, convention='xyz'),
                          rotmat.shape[:-2] + (3, ))
        assert check_func(rotmat, rotmat_to_quat(rotmat),
                          rotmat.shape[:-2] + (4, ))
        assert check_func(rotmat, rotmat_to_rot6d(rotmat),
                          rotmat.shape[:-2] + (6, ))
        # test rot6d convention
        assert check_func(rot6d, rot6d_to_aa(rot6d), rot6d.shape[:-1] + (3, ))
        assert check_func(rot6d, rot6d_to_rotmat(rot6d),
                          rot6d.shape[:-1] + (3, 3))
        assert check_func(rot6d, rot6d_to_quat(rot6d),
                          rot6d.shape[:-1] + (4, ))
        assert check_func(rot6d, rot6d_to_ee(rot6d, convention='xyz'),
                          rot6d.shape[:-1] + (3, ))
        # test standard joint angle convention
        assert check_func(aa_21, aa_to_sja(aa_21, R_t=R_t, R_t_inv=R_t_inv),
                          aa_21.shape[:-2] + (21, 3))
        assert check_func(sja, sja_to_aa(sja, R_t=R_t, R_t_inv=R_t_inv),
                          sja.shape[:-2] + (21, 3))

        # test inverse transform
        assert (aa == ee_to_aa(
            aa_to_ee(aa, convention='xyz'), convention='xyz')).all()
        assert (aa == rotmat_to_aa(aa_to_rotmat(aa))).all()
        assert (aa == quat_to_aa(aa_to_quat(aa))).all()
        assert (aa == rot6d_to_aa(aa_to_rot6d(aa))).all()

        assert (ee == aa_to_ee(
            ee_to_aa(ee, convention='xyz'), convention='xyz')).all()
        assert (ee == rotmat_to_ee(
            ee_to_rotmat(ee, convention='xyz'), convention='xyz')).all()
        assert (ee == quat_to_ee(
            ee_to_quat(ee, convention='xyz'), convention='xyz')).all()
        assert (ee == rot6d_to_ee(
            ee_to_rot6d(ee, convention='xyz'), convention='xyz')).all()

        assert (quat == ee_to_quat(
            quat_to_ee(quat, convention='xyz'), convention='xyz')).all()
        assert (quat == rotmat_to_quat(quat_to_rotmat(quat))).all()
        assert (quat == aa_to_quat(quat_to_aa(quat))).all()
        assert (quat == rot6d_to_quat(quat_to_rot6d(quat))).all()

        assert (rotmat == ee_to_rotmat(
            rotmat_to_ee(rotmat, convention='xyz'), convention='xyz')).all()
        assert (rotmat == aa_to_rotmat(rotmat_to_aa(rotmat))).all()
        assert (rotmat == quat_to_rotmat(rotmat_to_quat(rotmat))).all()
        assert (rotmat == rot6d_to_rotmat(rotmat_to_rot6d(rotmat))).all()

        assert (rot6d == ee_to_rot6d(
            rot6d_to_ee(rot6d, convention='xyz'), convention='xyz')).all()
        assert (rot6d == rotmat_to_rot6d(rot6d_to_rotmat(rot6d))).all()
        assert (rot6d == quat_to_rot6d(rot6d_to_quat(rot6d))).all()
        assert (rot6d == aa_to_rot6d(rot6d_to_aa(rot6d))).all()

        assert (aa_21 == sja_to_aa(
            aa_to_sja(aa_21, R_t=R_t, R_t_inv=R_t_inv),
            R_t=R_t,
            R_t_inv=R_t_inv)).all()
        assert (sja == aa_to_sja(
            sja_to_aa(sja, R_t=R_t, R_t_inv=R_t_inv), R_t=R_t,
            R_t_inv=R_t_inv)).all()

        aa = aa[None]
        ee = ee[None]
        quat = quat[None]
        rotmat = rotmat[None]
        rot6d = rot6d[None]
        aa_21 = aa_21[None]
        sja = sja[None]
        R_t = R_t[None]
        R_t_inv = R_t_inv[None]


def test_transforms_numpy():
    aa = np.zeros((3), dtype=np.float32)
    ee = np.zeros((3), dtype=np.float32)
    quat = np.array([1, 0, 0, 0], dtype=np.float32)
    rotmat = np.eye(3, 3, dtype=np.float32)
    rot6d = rotmat.reshape(-1)[:6]
    aa_21 = np.zeros((21, 3))
    sja = np.zeros((21, 3))
    R_t = np.eye(3, 3).reshape(1, 3, 3).repeat(21, axis=0)
    R_t_inv = np.eye(3, 3).reshape(1, 3, 3).repeat(21, axis=0)
    # test ndims
    for _ in range(10):
        # test convention
        assert check_func(aa, aa_to_ee(aa, convention='xyz'),
                          aa.shape[:-1] + (3, ))
        assert check_func(aa, aa_to_ee(aa, convention='xzy'),
                          aa.shape[:-1] + (3, ))
        assert check_func(aa, aa_to_ee(aa, convention='yxz'),
                          aa.shape[:-1] + (3, ))
        assert check_func(aa, aa_to_ee(aa, convention='yzx'),
                          aa.shape[:-1] + (3, ))
        assert check_func(aa, aa_to_ee(aa, convention='zyx'),
                          aa.shape[:-1] + (3, ))
        assert check_func(aa, aa_to_ee(aa, convention='zxy'),
                          aa.shape[:-1] + (3, ))

        # test aa convention
        assert check_func(aa, aa_to_rotmat(aa), aa.shape[:-1] + (3, 3))
        assert check_func(aa, aa_to_quat(aa), aa.shape[:-1] + (4, ))
        assert check_func(aa, aa_to_rot6d(aa), aa.shape[:-1] + (6, ))
        # test ee convention
        assert check_func(ee, ee_to_aa(ee, convention='xyz'),
                          ee.shape[:-1] + (3, ))
        assert check_func(ee, ee_to_rotmat(ee, convention='xyz'),
                          ee.shape[:-1] + (3, 3))
        assert check_func(ee, ee_to_quat(ee, convention='xyz'),
                          ee.shape[:-1] + (4, ))
        assert check_func(ee, ee_to_rot6d(ee, convention='xyz'),
                          ee.shape[:-1] + (6, ))
        # test quat convention
        assert check_func(quat, quat_to_aa(quat), quat.shape[:-1] + (3, ))
        assert check_func(quat, quat_to_rotmat(quat), quat.shape[:-1] + (3, 3))
        assert check_func(quat, quat_to_ee(quat, convention='xyz'),
                          quat.shape[:-1] + (3, ))
        assert check_func(quat, quat_to_rot6d(quat), quat.shape[:-1] + (6, ))
        # test rotmat convention
        assert check_func(rotmat, rotmat_to_aa(rotmat),
                          rotmat.shape[:-2] + (3, ))
        assert check_func(rotmat, rotmat_to_ee(rotmat, convention='xyz'),
                          rotmat.shape[:-2] + (3, ))
        assert check_func(rotmat, rotmat_to_quat(rotmat),
                          rotmat.shape[:-2] + (4, ))
        assert check_func(rotmat, rotmat_to_rot6d(rotmat),
                          rotmat.shape[:-2] + (6, ))
        # test rot6d convention
        assert check_func(rot6d, rot6d_to_aa(rot6d), rot6d.shape[:-1] + (3, ))
        assert check_func(rot6d, rot6d_to_rotmat(rot6d),
                          rot6d.shape[:-1] + (3, 3))
        assert check_func(rot6d, rot6d_to_quat(rot6d),
                          rot6d.shape[:-1] + (4, ))
        assert check_func(rot6d, rot6d_to_ee(rot6d, convention='xyz'),
                          rot6d.shape[:-1] + (3, ))

        # test inverse transform
        assert (aa == ee_to_aa(
            aa_to_ee(aa, convention='xyz'), convention='xyz')).all()
        assert (aa == rotmat_to_aa(aa_to_rotmat(aa))).all()
        assert (aa == quat_to_aa(aa_to_quat(aa))).all()
        assert (aa == rot6d_to_aa(aa_to_rot6d(aa))).all()

        assert (ee == aa_to_ee(
            ee_to_aa(ee, convention='xyz'), convention='xyz')).all()
        assert (ee == rotmat_to_ee(
            ee_to_rotmat(ee, convention='xyz'), convention='xyz')).all()
        assert (ee == quat_to_ee(
            ee_to_quat(ee, convention='xyz'), convention='xyz')).all()
        assert (ee == rot6d_to_ee(
            ee_to_rot6d(ee, convention='xyz'), convention='xyz')).all()

        assert (quat == ee_to_quat(
            quat_to_ee(quat, convention='xyz'), convention='xyz')).all()
        assert (quat == rotmat_to_quat(quat_to_rotmat(quat))).all()
        assert (quat == aa_to_quat(quat_to_aa(quat))).all()
        assert (quat == rot6d_to_quat(quat_to_rot6d(quat))).all()

        assert (rotmat == ee_to_rotmat(
            rotmat_to_ee(rotmat, convention='xyz'), convention='xyz')).all()
        assert (rotmat == aa_to_rotmat(rotmat_to_aa(rotmat))).all()
        assert (rotmat == quat_to_rotmat(rotmat_to_quat(rotmat))).all()
        assert (rotmat == rot6d_to_rotmat(rotmat_to_rot6d(rotmat))).all()

        assert (rot6d == ee_to_rot6d(
            rot6d_to_ee(rot6d, convention='xyz'), convention='xyz')).all()
        assert (rot6d == rotmat_to_rot6d(rot6d_to_rotmat(rot6d))).all()
        assert (rot6d == quat_to_rot6d(rot6d_to_quat(rot6d))).all()
        assert (rot6d == aa_to_rot6d(rot6d_to_aa(rot6d))).all()

        aa = aa[None]
        ee = ee[None]
        quat = quat[None]
        rotmat = rotmat[None]
        rot6d = rot6d[None]
        aa_21 = aa_21[None]
        sja = sja[None]
        R_t = R_t[None]
        R_t_inv = R_t_inv[None]
