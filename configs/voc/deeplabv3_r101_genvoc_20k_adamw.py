_base_ = './deeplabv3_r50_genvoc_20k_adamw.py'
model = dict(backbone=dict(depth=101))
