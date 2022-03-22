from email.policy import strict
from mmhuman3d.models.builder import build_backbone, build_head
import torch
from loguru import logger
from collections import OrderedDict

# from .configs.pare.hrnet_w32_conv_pare import head
def load_pretrained_model(model, state_dict, strict=False, overwrite_shape_mismatch=True, remove_lightning=False):
    if remove_lightning:
        logger.warning(f'Removing "model." keyword from state_dict keys..')
        pretrained_keys = state_dict.keys()
        new_state_dict = OrderedDict()
        for pk in pretrained_keys:
            if pk.startswith('model.head'):
                new_state_dict[pk.replace('model.head.', '')] = state_dict[pk]
    print(new_state_dict.keys())

    model.load_state_dict(new_state_dict, strict=strict)
    
    return model

head=dict(
    type='PareHead',
    num_joints = 24,
    num_input_features = 480, # will change futer get_backbone_info
    smpl_mean_params='/mnt/lustre/wangyanjun/data/smpl_mean_params.npz',
    num_deconv_layers = 2,
    num_deconv_filters = [128]*2, # num_deconv_filters = [num_deconv_filters] * num_deconv_layers
    num_deconv_kernels = [4]*2, # num_deconv_kernels = [num_deconv_kernels] * num_deconv_layers
    use_coattention = False,
    shape_input_type = 'feats.shape.cam',
    pose_input_type = 'feats.self_pose.shape.cam',
    use_heatmaps = 'part_segm',
    use_keypoint_attention = True,
    backbone = 'hrnet_w32-conv',
    )
 
model = build_head(head)
model.init_weights()
# state_dict = torch.load('/mnt/lustre/wangyanjun/data/pare/checkpoints/pare_checkpoint.ckpt',map_location=torch.device('cpu'))['state_dict']
state_dict = torch.load('/mnt/lustre/wangyanjun/pare_log/pare/pare/22-02-2022_10-10-03_pare_train/tb_logs/0/checkpoints/epoch=5-step=29267.ckpt',map_location=torch.device('cpu'))['state_dict']


load_pretrained_model(model,state_dict,strict = True, remove_lightning=True)


pretrained_keys = state_dict.keys()
new_state_dict = OrderedDict()
for pk in pretrained_keys:
    if pk.startswith('model.'):
        new_state_dict[pk.replace('model.', '')] = state_dict[pk]
print(new_state_dict.keys())
torch.save(new_state_dict,'/mnt/lustre/wangyanjun/pare_log/mmhuman3d/pare_checkpoint.ckpt')
# t = torch.load('/mnt/lustre/wangyanjun/data/pretrained_models/pose_coco/pose_hrnet_w32_256x192.pth',map_location=torch.device('cpu'))
# # model.init_weights(cfg['model']['pretrained'])
# model.load_state_dict(t,strict=False)