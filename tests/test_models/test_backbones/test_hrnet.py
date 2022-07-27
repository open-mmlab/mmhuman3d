import pytest
import torch

from mmhuman3d.models.backbones.hrnet import (
    HRModule,
    PoseHighResolutionNet,
    PoseHighResolutionNetExpose,
)
from mmhuman3d.models.backbones.resnet import BasicBlock, Bottleneck


def all_zeros(modules):
    """Check if the weight(and bias) is all zero."""
    weight_zero = torch.equal(modules.weight.data,
                              torch.zeros_like(modules.weight.data))
    if hasattr(modules, 'bias'):
        bias_zero = torch.equal(modules.bias.data,
                                torch.zeros_like(modules.bias.data))
    else:
        bias_zero = True

    return weight_zero and bias_zero


@pytest.mark.parametrize('block', [BasicBlock, Bottleneck])
def test_hrmodule(block):
    # Test multiscale forward
    num_channles = (32, 64)
    in_channels = [c * block.expansion for c in num_channles]
    hrmodule = HRModule(
        num_branches=2,
        blocks=block,
        in_channels=in_channels,
        num_blocks=(4, 4),
        num_channels=num_channles,
    )

    feats = [
        torch.randn(1, in_channels[0], 64, 64),
        torch.randn(1, in_channels[1], 32, 32)
    ]
    feats = hrmodule(feats)

    assert len(feats) == 2
    assert feats[0].shape == torch.Size([1, in_channels[0], 64, 64])
    assert feats[1].shape == torch.Size([1, in_channels[1], 32, 32])

    # Test single scale forward
    num_channles = (32, 64)
    in_channels = [c * block.expansion for c in num_channles]
    hrmodule = HRModule(
        num_branches=2,
        blocks=block,
        in_channels=in_channels,
        num_blocks=(4, 4),
        num_channels=num_channles,
        multiscale_output=False,
    )

    feats = [
        torch.randn(1, in_channels[0], 64, 64),
        torch.randn(1, in_channels[1], 32, 32)
    ]
    feats = hrmodule(feats)

    assert len(feats) == 1
    assert feats[0].shape == torch.Size([1, in_channels[0], 64, 64])


def test_hrnet_backbone():
    # only have 3 stages
    extra = dict(
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
            num_channels=(32, 64)),
        stage3=dict(
            num_modules=4,
            num_branches=3,
            block='BASIC',
            num_blocks=(4, 4, 4),
            num_channels=(32, 64, 128)),
        return_list=True,
        downsample=False,
        use_conv=True,
        final_conv_kernel=1,
    )

    with pytest.raises(AssertionError):
        # HRNet now only support 4 stages
        PoseHighResolutionNet(extra=extra)
    extra['stage4'] = dict(
        num_modules=3,
        num_branches=3,  # should be 4
        block='BASIC',
        num_blocks=(4, 4, 4, 4),
        num_channels=(32, 64, 128, 256))

    with pytest.raises(AssertionError):
        # len(num_blocks) should equal num_branches
        PoseHighResolutionNet(extra=extra)

    extra['stage4']['num_branches'] = 4

    # Test hrnetv2p_w32
    model = PoseHighResolutionNet(extra=extra)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 256, 256)
    feats = model(imgs)
    assert len(feats) == 4
    assert feats[0].shape == torch.Size([1, 32, 64, 64])
    assert feats[3].shape == torch.Size([1, 256, 8, 8])

    # Test single scale output
    model = PoseHighResolutionNet(extra=extra, multiscale_output=False)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 256, 256)
    feats = model(imgs)
    assert len(feats) == 1
    assert feats[0].shape == torch.Size([1, 32, 64, 64])

    extra['return_list'] = False
    model = PoseHighResolutionNet(extra=extra)
    model.train()

    imgs = torch.randn(1, 3, 256, 256)
    feats = model(imgs)
    assert feats.shape == torch.Size([1, 480, 64, 64])
    extra['use_conv'] = False
    model = PoseHighResolutionNet(extra=extra)
    model.init_weights()
    imgs = torch.randn(1, 3, 256, 256)
    feats = model(imgs)
    assert feats.shape == torch.Size([1, 480, 64, 64])

    extra['downsample'] = True
    model = PoseHighResolutionNet(extra=extra)
    model.init_weights()
    imgs = torch.randn(1, 3, 256, 256)
    feats = model(imgs)
    assert feats.shape == torch.Size([1, 480, 8, 8])

    extra['use_conv'] = True
    model = PoseHighResolutionNet(extra=extra)
    model.init_weights()
    imgs = torch.randn(1, 3, 256, 256)
    feats = model(imgs)
    assert feats.shape == torch.Size([1, 480, 8, 8])
    extra['use_conv'] = False

    model = PoseHighResolutionNet(extra=extra, zero_init_residual=True)
    model.init_weights()

    model.train()
    init_cfg = {type: 'Pretrained'}
    pretrained = '.'
    with pytest.raises(AssertionError):
        #     # len(num_blocks) should equal num_branches
        PoseHighResolutionNet(
            extra=extra, init_cfg=init_cfg, pretrained=pretrained)
    with pytest.raises(TypeError):
        #     # len(num_blocks) should equal num_branches
        PoseHighResolutionNet(extra=extra, pretrained=1)

    PoseHighResolutionNet(extra=extra, pretrained=pretrained)

    extra = dict(
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
            num_channels=(48, 96)),
        stage3=dict(
            num_modules=4,
            num_branches=3,
            block='BASIC',
            num_blocks=(4, 4, 4),
            num_channels=(48, 96, 192)),
        stage4=dict(
            num_modules=3,
            num_branches=4,
            block='BASIC',
            num_blocks=(4, 4, 4, 4),
            num_channels=(48, 96, 192, 384)),
        downsample=True,
        use_conv=True,
        final_conv_kernel=1,
        return_list=False)
    model = PoseHighResolutionNetExpose(extra=extra)
    model.init_weights()
    imgs = torch.randn(1, 3, 256, 256)
    feats = model(imgs)
    assert feats.shape == torch.Size([1, 2048])
