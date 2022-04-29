# Config for DeciWatch trained on PW3D dataset with an interval of 5,
# window size of 1 + 5*5(where q=5).
# The model is trained only on SMPL pose parameters.
speed_up_cfg = dict(
    type='deciwatch',
    interval=5,
    slide_window_q=5,
    checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
    'mmhuman3d/models/deciwatch/deciwatch_interval5_q5.pth.tar?versionId='
    'CAEQOhiBgMCyq8O9gxgiIDRjMzViMjllNWRiNjRlMzA5ZjczYWIxOGU2OGFkYjdl')
