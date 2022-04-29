# Config for DeciWatch trained on PW3D dataset with an interval of 5,
# window size of 1 + 5*2(where q=2).
# The model is trained only on SMPL pose parameters.
speed_up_cfg = dict(
    type='deciwatch',
    interval=5,
    slide_window_q=2,
    checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
    'mmhuman3d/models/deciwatch/deciwatch_interval5_q2.pth.tar?versionId='
    'CAEQOhiBgIDgu8O9gxgiIDNjMDEyOWQ3NjRkODQ2YTI5MjUxYWU4NzhjOTc1YTRj')
