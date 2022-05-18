# Config for DeciWatch trained on PW3D dataset with an interval of 10,
# window size of 1 + 10*4(where q=4).
# The model is trained only on SMPL pose parameters.
speed_up_cfg = dict(
    type='deciwatch',
    interval=10,
    slide_window_q=4,
    checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
    'mmhuman3d/models/deciwatch/deciwatch_interval10_q4.pth.tar?versionId='
    'CAEQOhiBgICUq8O9gxgiIDJkZjUwYWJmNTRkNjQxMDE4YmUyNWMwNTcwNGQ4M2Ix')
