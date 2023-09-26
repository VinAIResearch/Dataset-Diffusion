_base_ = [
    "../../mmsegmentation/configs/_base_/models/deeplabv3_r50-d8.py",
    "../../mmsegmentation/configs/_base_/datasets/pascal_voc12.py",
    "../../mmsegmentation/configs/_base_/default_runtime.py",
    "../../mmsegmentation/configs/_base_/schedules/schedule_20k.py",
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(2048, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
test_dataloader = dict(
    dataset=dict(
        data_root="data/gen_voc", data_prefix=dict(img_path="image", seg_map_path="mask"), pipeline=test_pipeline
    )
)
env_cfg = dict(cudnn_benchmark=False)
test_evaluator = dict(format_only=True)

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    backbone=dict(depth=101),
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=21),
    auxiliary_head=dict(num_classes=21),
)
