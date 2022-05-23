# Config for SmoothNet filter trained on SPIN data with a window size of 16.
smooth_cfg = dict(
    type='smoothnet',
    window_size=16,
    output_size=16,
    checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
    'mmhuman3d/models/smoothnet/smoothnet_windowsize16.pth.tar?versionId'
    '=CAEQPhiBgMC.s87shhgiIGM3ZTI1ZGY1Y2NhNDQ2YzRiNmEyOGZhY2VjYWFiN2Zi')
