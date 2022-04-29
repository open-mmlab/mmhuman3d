# Config for DeciWatch trained on PW3D dataset with an interval of 5,
# window size of 1 + 5*4(where q=4).
# The model is trained only on SMPL pose parameters.
speed_up_cfg = dict(
    type='deciwatch',
    interval=5,
    slide_window_q=4,
    checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
    'mmhuman3d/models/deciwatch/deciwatch_interval5_q4.pth.tar?versionId='
    'CAEQOhiBgMC.t8O9gxgiIGZjZWY3OTdhNGRjZjQyNjY5MGU5YzkxZTZjMWU1MTA2')
