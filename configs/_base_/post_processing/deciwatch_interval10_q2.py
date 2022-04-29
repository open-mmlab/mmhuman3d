# Config for DeciWatch trained on PW3D dataset with an interval of 10,
# window size of 1 + 10*2(where q=2).
# The model is trained only on SMPL pose parameters.
speed_up_cfg = dict(
    type='deciwatch',
    interval=10,
    slide_window_q=2,
    checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
    'mmhuman3d/models/deciwatch/deciwatch_interval10_q2.pth.tar?versionId='
    'CAEQOhiBgICau8O9gxgiIDk1Y2Y0MzUxMmY0MDQzZThiYzhkMGJlMjc3ZDQ2NTQ2')
