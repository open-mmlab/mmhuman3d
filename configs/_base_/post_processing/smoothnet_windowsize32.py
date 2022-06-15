# Config for SmoothNet filter trained on SPIN data with a window size of 32.
smooth_cfg = dict(
    type='smoothnet',
    window_size=32,
    output_size=32,
    checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
    'mmhuman3d/models/smoothnet/smoothnet_windowsize32.pth.tar?versionId'
    '=CAEQPhiBgIDf0s7shhgiIDhmYmM3YWQ0ZGI3NjRmZTc4NTk2NDE1MTA2MTUyMGRm')
