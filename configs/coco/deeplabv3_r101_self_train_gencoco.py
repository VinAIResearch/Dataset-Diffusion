_base_ = './deeplabv3_r50_self_train_gencoco.py'
model = dict(backbone=dict(depth=101))
