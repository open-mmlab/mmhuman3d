# Config for DeciWatch trained on PW3D dataset with an interval of 10,
# window size of 1 + 10*5(where q=5).
# The model is trained only on SMPL pose parameters.
speed_up_cfg = dict(
    type='deciwatch',
    interval=10,
    slide_window_q=5,
    checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
    'mmhuman3d/models/deciwatch/deciwatch_interval10_q5.pth.tar?versionId='
    'CAEQOhiBgMCN7MS9gxgiIDUwNGFhM2Y0MGI3MjRiYWQ5NzZjODMwMDk3ZjU1OTk3')
