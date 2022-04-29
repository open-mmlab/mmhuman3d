import numpy as np
import torch

from mmhuman3d.core.post_processing.builder import build_post_processing


#  test different data type
def test_data_type_torch():
    noisy_input = torch.randn((100, 24, 3))
    cfg = dict(type='OneEuroFilter', min_cutoff=0.004, beta=0.7)
    oneeuro = build_post_processing(cfg)
    out_g = oneeuro(noisy_input)
    cfg = dict(type='Gaus1dFilter', window_size=11, sigma=4)
    gaus1d = build_post_processing(cfg)
    out_s = gaus1d(noisy_input)
    cfg = dict(type='SGFilter', window_size=11, polyorder=2)
    savgol = build_post_processing(cfg)
    out_o = savgol(noisy_input)
    cfg = dict(
        type='deciwatch',
        interval=5,
        slide_window_q=1,
        checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
        'mmhuman3d/models/deciwatch/deciwatch_interval5_q1.pth.tar?versionId='
        'CAEQOhiBgIDfocS9gxgiIDkxN2Y3OWQzZmJiMTQyMTM5NWZhZTYxYmI0MDlmMDBh',
        device='cpu')
    deciwatch_5_1 = build_post_processing(cfg)
    out_d_5_1 = deciwatch_5_1(noisy_input)
    cfg = dict(
        type='deciwatch',
        interval=5,
        slide_window_q=2,
        checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
        'mmhuman3d/models/deciwatch/deciwatch_interval5_q1.pth.tar?versionId='
        'CAEQOhiBgIDfocS9gxgiIDkxN2Y3OWQzZmJiMTQyMTM5NWZhZTYxYmI0MDlmMDBh',
        device='cpu')
    deciwatch_5_2 = build_post_processing(cfg)
    out_d_5_2 = deciwatch_5_2(noisy_input)
    cfg = dict(
        type='deciwatch',
        interval=5,
        slide_window_q=3,
        checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
        'mmhuman3d/models/deciwatch/deciwatch_interval5_q3.pth.tar?versionId='
        'CAEQOhiBgIDJs8O9gxgiIDk1MDExMjI5Y2U1MDRmZjViMDBjOGU5YzY3OTRlNmE5',
        device='cpu')
    deciwatch_5_3 = build_post_processing(cfg)
    out_d_5_3 = deciwatch_5_3(noisy_input)
    cfg = dict(
        type='deciwatch',
        interval=5,
        slide_window_q=4,
        checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
        'mmhuman3d/models/deciwatch/deciwatch_interval5_q4.pth.tar?versionId='
        'CAEQOhiBgMC.t8O9gxgiIGZjZWY3OTdhNGRjZjQyNjY5MGU5YzkxZTZjMWU1MTA2',
        device='cpu')
    deciwatch_5_4 = build_post_processing(cfg)
    out_d_5_4 = deciwatch_5_4(noisy_input)
    cfg = dict(
        type='deciwatch',
        interval=5,
        slide_window_q=5,
        checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
        'mmhuman3d/models/deciwatch/deciwatch_interval5_q5.pth.tar?versionId='
        'CAEQOhiBgMCyq8O9gxgiIDRjMzViMjllNWRiNjRlMzA5ZjczYWIxOGU2OGFkYjdl',
        device='cpu')
    deciwatch_5_5 = build_post_processing(cfg)
    out_d_5_5 = deciwatch_5_5(noisy_input)
    cfg = dict(
        type='deciwatch',
        interval=10,
        slide_window_q=1,
        checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
        'mmhuman3d/models/deciwatch/deciwatch_interval10_q1.pth.tar?versionId='
        'CAEQOhiBgMChhsS9gxgiIDM5OGUwZGY0MTc4NTQ2M2NhZDEwMzU5MWUzMWNmZjY1',
        device='cpu')
    deciwatch_10_1 = build_post_processing(cfg)
    out_d_10_1 = deciwatch_10_1(noisy_input)
    cfg = dict(
        type='deciwatch',
        interval=10,
        slide_window_q=2,
        checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
        'mmhuman3d/models/deciwatch/deciwatch_interval10_q2.pth.tar?versionId='
        'CAEQOhiBgICau8O9gxgiIDk1Y2Y0MzUxMmY0MDQzZThiYzhkMGJlMjc3ZDQ2NTQ2',
        device='cpu')
    deciwatch_10_2 = build_post_processing(cfg)
    out_d_10_2 = deciwatch_10_2(noisy_input)
    cfg = dict(
        type='deciwatch',
        interval=10,
        slide_window_q=3,
        checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
        'mmhuman3d/models/deciwatch/deciwatch_interval10_q3.pth.tar?versionId='
        'CAEQOhiBgICIq8O9gxgiIDZiMjEzMjY3ODA4MTQwNGY5NTU3OWNkZjRjZjI2ZDFi',
        device='cpu')
    deciwatch_10_3 = build_post_processing(cfg)
    out_d_10_3 = deciwatch_10_3(noisy_input)
    cfg = dict(
        type='deciwatch',
        interval=10,
        slide_window_q=4,
        checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
        'mmhuman3d/models/deciwatch/deciwatch_interval10_q4.pth.tar?versionId='
        'CAEQOhiBgICUq8O9gxgiIDJkZjUwYWJmNTRkNjQxMDE4YmUyNWMwNTcwNGQ4M2Ix',
        device='cpu')
    deciwatch_10_4 = build_post_processing(cfg)
    out_d_10_4 = deciwatch_10_4(noisy_input)
    cfg = dict(
        type='deciwatch',
        interval=10,
        slide_window_q=5,
        checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
        'mmhuman3d/models/deciwatch/deciwatch_interval10_q5.pth.tar?versionId='
        'CAEQOhiBgMCN7MS9gxgiIDUwNGFhM2Y0MGI3MjRiYWQ5NzZjODMwMDk3ZjU1OTk3',
        device='cpu')
    deciwatch_10_5 = build_post_processing(cfg)
    out_d_10_5 = deciwatch_10_5(noisy_input)
    # verify the correctness
    accel_input = noisy_input[:-2] - 2 * noisy_input[1:-1] + noisy_input[2:]
    accel_out_g = out_g[:-2] - 2 * out_g[1:-1] + out_g[2:]
    accel_input_abs = torch.mean(torch.abs(accel_input))
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_g))
    accel_out_s = out_s[:-2] - 2 * out_s[1:-1] + out_s[2:]
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_s))
    accel_out_o = out_o[:-2] - 2 * out_o[1:-1] + out_o[2:]
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_o))
    accel_out_d_5_1 = out_d_5_1[:-2] - 2 * out_d_5_1[1:-1] + out_d_5_1[2:]
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_d_5_1))
    accel_out_d_5_2 = out_d_5_2[:-2] - 2 * out_d_5_2[1:-1] + out_d_5_2[2:]
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_d_5_2))
    accel_out_d_5_3 = out_d_5_3[:-2] - 2 * out_d_5_3[1:-1] + out_d_5_3[2:]
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_d_5_3))
    accel_out_d_5_4 = out_d_5_4[:-2] - 2 * out_d_5_4[1:-1] + out_d_5_4[2:]
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_d_5_4))
    accel_out_d_5_5 = out_d_5_5[:-2] - 2 * out_d_5_5[1:-1] + out_d_5_5[2:]
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_d_5_5))
    accel_out_d_10_1 = out_d_10_1[:-2] - 2 * out_d_10_1[1:-1] + out_d_10_1[2:]
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_d_10_1))
    accel_out_d_10_2 = out_d_10_2[:-2] - 2 * out_d_10_2[1:-1] + out_d_10_2[2:]
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_d_10_2))
    accel_out_d_10_3 = out_d_10_3[:-2] - 2 * out_d_10_3[1:-1] + out_d_10_3[2:]
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_d_10_3))
    accel_out_d_10_4 = out_d_10_4[:-2] - 2 * out_d_10_4[1:-1] + out_d_10_4[2:]
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_d_10_4))
    accel_out_d_10_5 = out_d_10_5[:-2] - 2 * out_d_10_5[1:-1] + out_d_10_5[2:]
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_d_10_5))
    assert out_g.shape == noisy_input.shape == \
        out_s.shape == out_o.shape == out_d_5_1.shape == out_d_5_2.shape \
        == out_d_5_3.shape == out_d_5_4.shape == out_d_5_5.shape \
        == out_d_10_1.shape == out_d_10_2.shape == out_d_10_3.shape \
        == out_d_10_4.shape == out_d_10_5.shape


def test_data_type_torch_zero():
    noisy_input = torch.zeros((50, 20, 3))
    cfg = dict(type='OneEuroFilter', min_cutoff=0.004, beta=0.7)
    oneeuro = build_post_processing(cfg)
    out_g = oneeuro(noisy_input)
    cfg = dict(type='Gaus1dFilter', window_size=11, sigma=4)
    gaus1d = build_post_processing(cfg)
    out_s = gaus1d(noisy_input)
    cfg = dict(type='SGFilter', window_size=11, polyorder=2)
    savgol = build_post_processing(cfg)
    out_o = savgol(noisy_input)
    # verify the correctness
    accel_input = noisy_input[:-2] - 2 * noisy_input[1:-1] + noisy_input[2:]
    accel_out_g = out_g[:-2] - 2 * out_g[1:-1] + out_g[2:]
    assert torch.mean(accel_input) >= torch.mean(accel_out_g)
    accel_out_s = out_s[:-2] - 2 * out_s[1:-1] + out_s[2:]
    assert torch.mean(accel_input) >= torch.mean(accel_out_s)
    accel_out_o = out_o[:-2] - 2 * out_o[1:-1] + out_o[2:]
    assert torch.mean(accel_input) >= torch.mean(accel_out_o)
    assert out_g.shape == noisy_input.shape == out_s.shape == out_o.shape


def test_data_type_torch_cuda():
    if not torch.cuda.is_available():
        return
    noisy_input = torch.randn((100, 24, 3)).cuda()
    cfg = dict(type='OneEuroFilter', min_cutoff=0.0004, beta=0.7)
    oneeuro = build_post_processing(cfg)
    out_g = oneeuro(noisy_input)
    cfg = dict(type='Gaus1dFilter', window_size=6, sigma=1)
    gaus1d = build_post_processing(cfg)
    out_s = gaus1d(noisy_input)
    cfg = dict(type='SGFilter', window_size=7, polyorder=2)
    savgol = build_post_processing(cfg)
    out_o = savgol(noisy_input)
    cfg = dict(
        type='deciwatch',
        interval=5,
        slide_window_q=1,
        checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
        'mmhuman3d/models/deciwatch/deciwatch_interval5_q1.pth.tar?versionId='
        'CAEQOhiBgIDfocS9gxgiIDkxN2Y3OWQzZmJiMTQyMTM5NWZhZTYxYmI0MDlmMDBh',
        device='cuda:0')
    deciwatch_5_1 = build_post_processing(cfg)
    out_d_5_1 = deciwatch_5_1(noisy_input)
    cfg = dict(
        type='deciwatch',
        interval=5,
        slide_window_q=2,
        checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
        'mmhuman3d/models/deciwatch/deciwatch_interval5_q1.pth.tar?versionId='
        'CAEQOhiBgIDfocS9gxgiIDkxN2Y3OWQzZmJiMTQyMTM5NWZhZTYxYmI0MDlmMDBh',
        device='cuda:0')
    deciwatch_5_2 = build_post_processing(cfg)
    out_d_5_2 = deciwatch_5_2(noisy_input)
    cfg = dict(
        type='deciwatch',
        interval=5,
        slide_window_q=3,
        checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
        'mmhuman3d/models/deciwatch/deciwatch_interval5_q3.pth.tar?versionId='
        'CAEQOhiBgIDJs8O9gxgiIDk1MDExMjI5Y2U1MDRmZjViMDBjOGU5YzY3OTRlNmE5',
        device='cuda:0')
    deciwatch_5_3 = build_post_processing(cfg)
    out_d_5_3 = deciwatch_5_3(noisy_input)
    cfg = dict(
        type='deciwatch',
        interval=5,
        slide_window_q=4,
        checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
        'mmhuman3d/models/deciwatch/deciwatch_interval5_q4.pth.tar?versionId='
        'CAEQOhiBgMC.t8O9gxgiIGZjZWY3OTdhNGRjZjQyNjY5MGU5YzkxZTZjMWU1MTA2',
        device='cuda:0')
    deciwatch_5_4 = build_post_processing(cfg)
    out_d_5_4 = deciwatch_5_4(noisy_input)
    cfg = dict(
        type='deciwatch',
        interval=5,
        slide_window_q=5,
        checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
        'mmhuman3d/models/deciwatch/deciwatch_interval5_q5.pth.tar?versionId='
        'CAEQOhiBgMCyq8O9gxgiIDRjMzViMjllNWRiNjRlMzA5ZjczYWIxOGU2OGFkYjdl',
        device='cuda:0')
    deciwatch_5_5 = build_post_processing(cfg)
    out_d_5_5 = deciwatch_5_5(noisy_input)
    cfg = dict(
        type='deciwatch',
        interval=10,
        slide_window_q=1,
        checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
        'mmhuman3d/models/deciwatch/deciwatch_interval10_q1.pth.tar?versionId='
        'CAEQOhiBgMChhsS9gxgiIDM5OGUwZGY0MTc4NTQ2M2NhZDEwMzU5MWUzMWNmZjY1',
        device='cuda:0')
    deciwatch_10_1 = build_post_processing(cfg)
    out_d_10_1 = deciwatch_10_1(noisy_input)
    cfg = dict(
        type='deciwatch',
        interval=10,
        slide_window_q=2,
        checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
        'mmhuman3d/models/deciwatch/deciwatch_interval10_q2.pth.tar?versionId='
        'CAEQOhiBgICau8O9gxgiIDk1Y2Y0MzUxMmY0MDQzZThiYzhkMGJlMjc3ZDQ2NTQ2',
        device='cuda:0')
    deciwatch_10_2 = build_post_processing(cfg)
    out_d_10_2 = deciwatch_10_2(noisy_input)
    cfg = dict(
        type='deciwatch',
        interval=10,
        slide_window_q=3,
        checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
        'mmhuman3d/models/deciwatch/deciwatch_interval10_q3.pth.tar?versionId='
        'CAEQOhiBgICIq8O9gxgiIDZiMjEzMjY3ODA4MTQwNGY5NTU3OWNkZjRjZjI2ZDFi',
        device='cuda:0')
    deciwatch_10_3 = build_post_processing(cfg)
    out_d_10_3 = deciwatch_10_3(noisy_input)
    cfg = dict(
        type='deciwatch',
        interval=10,
        slide_window_q=4,
        checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
        'mmhuman3d/models/deciwatch/deciwatch_interval10_q4.pth.tar?versionId='
        'CAEQOhiBgICUq8O9gxgiIDJkZjUwYWJmNTRkNjQxMDE4YmUyNWMwNTcwNGQ4M2Ix',
        device='cuda:0')
    deciwatch_10_4 = build_post_processing(cfg)
    out_d_10_4 = deciwatch_10_4(noisy_input)
    cfg = dict(
        type='deciwatch',
        interval=10,
        slide_window_q=5,
        checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
        'mmhuman3d/models/deciwatch/deciwatch_interval10_q5.pth.tar?versionId='
        'CAEQOhiBgMCN7MS9gxgiIDUwNGFhM2Y0MGI3MjRiYWQ5NzZjODMwMDk3ZjU1OTk3',
        device='cuda:0')
    deciwatch_10_5 = build_post_processing(cfg)
    out_d_10_5 = deciwatch_10_5(noisy_input)
    # verify the correctness
    accel_input = noisy_input[:-2] - 2 * noisy_input[1:-1] + noisy_input[2:]
    accel_out_g = out_g[:-2] - 2 * out_g[1:-1] + out_g[2:]
    accel_input_abs = torch.mean(torch.abs(accel_input))
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_g))
    accel_out_s = out_s[:-2] - 2 * out_s[1:-1] + out_s[2:]
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_s))
    accel_out_o = out_o[:-2] - 2 * out_o[1:-1] + out_o[2:]
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_o))
    accel_out_d_5_1 = out_d_5_1[:-2] - 2 * out_d_5_1[1:-1] + out_d_5_1[2:]
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_d_5_1))
    accel_out_d_5_2 = out_d_5_2[:-2] - 2 * out_d_5_2[1:-1] + out_d_5_2[2:]
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_d_5_2))
    accel_out_d_5_3 = out_d_5_3[:-2] - 2 * out_d_5_3[1:-1] + out_d_5_3[2:]
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_d_5_3))
    accel_out_d_5_4 = out_d_5_4[:-2] - 2 * out_d_5_4[1:-1] + out_d_5_4[2:]
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_d_5_4))
    accel_out_d_5_5 = out_d_5_5[:-2] - 2 * out_d_5_5[1:-1] + out_d_5_5[2:]
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_d_5_5))
    accel_out_d_10_1 = out_d_10_1[:-2] - 2 * out_d_10_1[1:-1] + out_d_10_1[2:]
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_d_10_1))
    accel_out_d_10_2 = out_d_10_2[:-2] - 2 * out_d_10_2[1:-1] + out_d_10_2[2:]
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_d_10_2))
    accel_out_d_10_3 = out_d_10_3[:-2] - 2 * out_d_10_3[1:-1] + out_d_10_3[2:]
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_d_10_3))
    accel_out_d_10_4 = out_d_10_4[:-2] - 2 * out_d_10_4[1:-1] + out_d_10_4[2:]
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_d_10_4))
    accel_out_d_10_5 = out_d_10_5[:-2] - 2 * out_d_10_5[1:-1] + out_d_10_5[2:]
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_d_10_5))
    assert out_g.shape == noisy_input.shape == \
        out_s.shape == out_o.shape == out_d_5_1.shape == out_d_5_2.shape \
        == out_d_5_3.shape == out_d_5_4.shape == out_d_5_5.shape \
        == out_d_10_1.shape == out_d_10_2.shape == out_d_10_3.shape \
        == out_d_10_4.shape == out_d_10_5.shape


def test_data_type_np():
    noisy_input = np.random.rand(100, 24, 6)
    cfg = dict(type='OneEuroFilter', min_cutoff=0.004, beta=0.1)
    oneeuro = build_post_processing(cfg)
    out_g = oneeuro(noisy_input)
    cfg = dict(type='Gaus1dFilter', window_size=5, sigma=2)
    gaus1d = build_post_processing(cfg)
    out_s = gaus1d(noisy_input)
    cfg = dict(type='SGFilter', window_size=5, polyorder=2)
    savgol = build_post_processing(cfg)
    out_o = savgol(noisy_input)
    assert out_g.shape == noisy_input.shape == out_s.shape == out_o.shape


if __name__ == '__main__':
    test_data_type_torch()
