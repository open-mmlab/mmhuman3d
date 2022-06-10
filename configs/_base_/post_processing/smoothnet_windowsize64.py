# Config for SmoothNet filter trained on SPIN data with a window size of 64.
smooth_cfg = dict(
    type='smoothnet',
    window_size=64,
    output_size=64,
    checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
    'mmhuman3d/models/smoothnet/smoothnet_windowsize64.pth.tar?versionId'
    '=CAEQPhiBgMCyw87shhgiIGEwODI4ZjdiYmFkYTQ0NzZiNDVkODk3MDBlYzE1Y2Rh')
