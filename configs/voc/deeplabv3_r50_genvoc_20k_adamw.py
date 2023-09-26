_base_ = [
    "../../mmsegmentation/configs/_base_/models/deeplabv3_r50-d8.py",
    "../../mmsegmentation/configs/_base_/datasets/pascal_voc12.py",
    "../../mmsegmentation/configs/_base_/default_runtime.py",
    "../../mmsegmentation/configs/_base_/schedules/schedule_20k.py",
]

train_dataloader = dict(
    dataset=dict(data_root="data/gen_voc", data_prefix=dict(img_path="image", seg_map_path="mask"))
)

val_dataloader = dict(
    dataset=dict(
        data_root="data/VOCdevkit/VOC2012",
        data_prefix=dict(img_path="JPEGImages", seg_map_path="SegmentationClass"),
        ann_file="ImageSets/Segmentation/val.txt",
    )
)

test_dataloader = val_dataloader

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
param_scheduler = [dict(type="PolyLR", eta_min=0, power=0.9, begin=0, end=20000, by_epoch=False)]
train_cfg = dict(type="IterBasedTrainLoop", max_iters=20000, val_interval=1000)
env_cfg = dict(cudnn_benchmark=False)

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor, decode_head=dict(num_classes=21), auxiliary_head=dict(num_classes=21)
)
