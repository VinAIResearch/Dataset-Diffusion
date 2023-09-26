_base_ = [
    "../../mmsegmentation/configs/_base_/models/deeplabv3_r50-d8.py",
    "../../mmsegmentation/configs/_base_/datasets/coco2017.py",
    "../../mmsegmentation/configs/_base_/default_runtime.py",
    "../../mmsegmentation/configs/_base_/schedules/schedule_40k.py",
]

crop_size = (512, 512)
optimizer = dict(type="AdamW", lr=0.0001, weight_decay=0.0001, eps=1e-8, betas=(0.9, 0.999))
optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.1, decay_mult=1.0),
        }
    ),
)
# learning policy
param_scheduler = [dict(type="PolyLR", eta_min=0, power=0.9, begin=0, end=40000, by_epoch=False)]

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="RandomResize", scale=(512, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(
    dataset=dict(
        data_root="data/gen_coco", data_prefix=dict(img_path="image", seg_map_path="mask"), pipeline=train_pipeline
    )
)

val_dataloader = dict(dataset=dict(data_root="data/coco", data_prefix=dict(img_path="val2017", seg_map_path="masks")))

test_dataloader = dict(dataset=dict(data_root="data/coco", data_prefix=dict(img_path="val2017", seg_map_path="masks")))

data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor, decode_head=dict(num_classes=81), auxiliary_head=dict(num_classes=81)
)

train_cfg = dict(type="IterBasedTrainLoop", max_iters=40000, val_interval=4000)
