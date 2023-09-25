GPUS=2
sh mmsegmentation/tools/dist_train.sh configs/coco/deeplabv3_r101_gencoco_40k_adamw.py $GPUS

# Self-training
sh mmsegmentation/tools/dist_test.sh configs/coco/deeplabv3_r101_self_train_gencoco.py work_dirs/deeplabv3_r101_gencoco_40k_adamw/iter_40000.pth $GPUS --out data/gen_coco/mask_self_train/
sh mmsegmentation/tools/dist_train.sh configs/coco/deeplabv3_r101_gencoco_40k_adamw.py $GPUS --cfg-options train_dataloader.dataset.seg_map_path=mask_self_train
