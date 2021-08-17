# optimizer
optimizer = dict(type='Adam', lr=0.0002, weight_decay=0)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[])
runner = dict(type='EpochBasedRunner', max_epochs=50)
