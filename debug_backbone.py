from email.policy import strict
from mmhuman3d.models.builder import build_backbone
import torch
def get_cfg_defaults(width=32, downsample=False, use_conv=False):
    # pose_multi_resoluton_net related params
    hrnet_extra = dict(
        stage1=dict(
            num_modules=1,
            num_branches=1,
            block='BOTTLENECK',
            num_blocks=(4, ),
            num_channels=(64, )),
        stage2=dict(
            num_modules=1,
            num_branches=2,
            block='BASIC',
            num_blocks=(4, 4),
            num_channels=(width, width*2)),
        stage3=dict(
            num_modules=4,
            num_branches=3,
            block='BASIC',
            num_blocks=(4, 4, 4),
            num_channels=(width, width*2, width*4)),
        stage4=dict(
            num_modules=3,
            num_branches=4,
            block='BASIC',
            num_blocks=(4, 4, 4, 4),
            num_channels=(width, width*2, width*4, width*8)),
        downsample = downsample,
        use_conv = use_conv,
        pretrained_layers = [
            'conv1', 'bn1', 'conv2', 'bn2', 'layer1', 'transition1',
            'stage2', 'transition2', 'stage3', 'transition3', 'stage4',
        ],
        final_conv_kernel = 1,

        )
    cfg = dict(
        model = dict(
            init_weights = True,
            extra = hrnet_extra,
            num_joints = 17
        ),
    )

    return cfg


cfg = get_cfg_defaults(width=32, downsample=False, use_conv=True)
# backbone=dict(
#         type='PoseHighResolutionNet',
#         model_cfg=cfg,
#         pretrained = '/mnt/lustre/wangyanjun/data/pretrained_models/pose_coco/pose_hrnet_w32_256x192.pth'
#         )
backbone=dict(
        type='PoseHighResolutionNet',
        model_cfg=cfg,
        # pretrained = '/mnt/lustre/wangyanjun/data/pretrained_models/pose_coco/pose_hrnet_w32_256x192.pth',
        init_cfg=dict(type='Pretrained', checkpoint='/mnt/lustre/wangyanjun/data/pretrained_models/pose_coco/pose_hrnet_w32_256x192.pth')
        )
model = build_backbone(backbone)
input = torch.randn(1,3,224,224)
print(model(input).shape)


# t = torch.load('/mnt/lustre/wangyanjun/data/pretrained_models/pose_coco/pose_hrnet_w32_256x192.pth',map_location=torch.device('cpu'))
# # model.init_weights(cfg['model']['pretrained'])
# model.load_state_dict(t,strict=False)