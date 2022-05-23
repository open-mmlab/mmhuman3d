# Config for SmoothNet filter trained on SPIN data with a window size of 8.
smooth_cfg = dict(
    type='smoothnet',
    window_size=8,
    output_size=8,
    checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
    'mmhuman3d/models/smoothnet/smoothnet_windowsize8.pth.tar?versionId'
    '=CAEQPhiBgMDo0s7shhgiIDgzNTRmNWM2ZWEzYTQyYzRhNzUwYTkzZWZkMmU5MWEw')
