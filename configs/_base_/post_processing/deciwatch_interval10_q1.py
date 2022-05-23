# Config for DeciWatch trained on PW3D dataset with an interval of 10,
# window size of 1 + 10*1(where q=1).
# The model is trained only on SMPL pose parameters.
speed_up_cfg = dict(
    type='deciwatch',
    interval=10,
    slide_window_q=1,
    checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
    'mmhuman3d/models/deciwatch/deciwatch_interval10_q1.pth.tar?versionId='
    'CAEQOhiBgMChhsS9gxgiIDM5OGUwZGY0MTc4NTQ2M2NhZDEwMzU5MWUzMWNmZjY1')
