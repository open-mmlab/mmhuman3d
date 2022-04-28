# Config for DeciWatch trained on PW3D dataset with an interval of 5, window size of 1 + 5*1(where q=1). The model is trained only on SMPL pose parameters.
speed_up_cfg = dict(
    type='deciwatch',
    interval=10,
    slide_window_q=3,
    checkpoint=
    'https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/deciwatch/deciwatch_interval10_q3.pth.tar?versionId=CAEQOhiBgICIq8O9gxgiIDZiMjEzMjY3ODA4MTQwNGY5NTU3OWNkZjRjZjI2ZDFi'
)
