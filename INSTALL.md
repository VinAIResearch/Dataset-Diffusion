## Install MMSegmentation
* Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```
* Install MMSegmentation
```shell
cd mmsegmentation
pip install -v -e .
```

* Optional: [MMDetection](https://github.com/open-mmlab/mmdetection) for training Mask2Former model.
```shell
mim install mmdet
```

## Datasets preparation for training segmenters
This dataset preparation is inspired by [MMSegmentation datasets structure](https://github.com/open-mmlab/mmsegmentation/blob/e64548fda0221ad708f5da29dc907e51a644c345/docs/en/user_guides/2_dataset_prepare.md)

It is recommended to symlink the dataset root to `data`.
You need to structure the `data` folder as follows:
```none
data
├── VOCdevkit
│   ├── VOC2012
│   │   ├── JPEGImages
│   │   ├── ImageSets
│   │   │   ├── Segmentation
│   │   ├── SegmentationClass
├── coco
│   ├── train2017
│   ├── val2017
│   ├── masks
```

### Pascal VOC
Pascal VOC 2012 could be downloaded from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)


### COCO
The COCO2017 dataset could be downloaded from [here](https://cocodataset.org/#download). 

The semantic segmentation annotations for the MS COCO dataset could be downloaded from [here](https://drive.google.com/file/d/1pRE9SEYkZKVg0Rgz2pi9tg48j7GlinPV/view?usp=sharing)

## Usage
Prefer to [README](README.md)