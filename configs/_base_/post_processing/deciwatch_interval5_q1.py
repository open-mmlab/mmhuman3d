# Config for DeciWatch trained on PW3D dataset with an interval of 5,
# window size of 1 + 5*1(where q=1).
# The model is trained only on SMPL pose parameters.
speed_up_cfg = dict(
    type='deciwatch',
    interval=5,
    slide_window_q=1,
    checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
    'mmhuman3d/models/deciwatch/deciwatch_interval5_q1.pth.tar?versionId='
    'CAEQOhiBgIDfocS9gxgiIDkxN2Y3OWQzZmJiMTQyMTM5NWZhZTYxYmI0MDlmMDBh')
