# Config for DeciWatch trained on PW3D dataset with an interval of 5,
# window size of 1 + 5*3(where q=3).
# The model is trained only on SMPL pose parameters.
speed_up_cfg = dict(
    type='deciwatch',
    interval=5,
    slide_window_q=3,
    checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
    'mmhuman3d/models/deciwatch/deciwatch_interval5_q3.pth.tar?versionId='
    'CAEQOhiBgIDJs8O9gxgiIDk1MDExMjI5Y2U1MDRmZjViMDBjOGU5YzY3OTRlNmE5')
