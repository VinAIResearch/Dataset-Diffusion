_base_ = "./deeplabv3_r50_gencoco_40k_adamw.py"
model = dict(backbone=dict(depth=101))
