GPUS=2
sh mmsegmentation/tools/dist_train.sh configs/voc/deeplabv3_r101_genvoc_20k_adamw.py $GPUS

# Self-training
sh mmsegmentation/tools/dist_test.sh configs/voc/deeplabv3_r101_self_train_genvoc.py work_dirs/deeplabv3_r101_genvoc_20k_adamw/iter_20000.pth $GPUS --out data/gen_voc/mask_self_train/
sh mmsegmentation/tools/dist_train.sh configs/voc/deeplabv3_r101_genvoc_20k_adamw.py $GPUS --cfg-options train_dataloader.dataset.seg_map_path=mask_self_train
