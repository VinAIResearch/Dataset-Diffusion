#   Definition of classes which are existed in the dataset
#   For the order of classes, please refer to 
#   https://github.com/open-mmlab/mmsegmentation/tree/main/mmseg/datasets

classes = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

COMPOUND_NOUNS = [
    'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'sports ball','baseball bat', 'baseball glove',
    'tennis racket', 'wine glass', 'potted plant', 'dining table', 'cell phone',
]

palette = [[0, 0, 0], [0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
            [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
            [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
            [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
            [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
            [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
            [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160],
            [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
            [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128],
            [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
            [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128],
            [128, 128, 64], [192, 0, 32], [128, 96, 0], [128, 0, 192],
            [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160],
            [64, 96, 0], [0, 128, 192], [0, 128, 160], [192, 224, 0],
            [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
            [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160],
            [64, 32, 128], [128, 192, 192], [0, 0, 160], [192, 160, 128],
            [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128],
            [64, 128, 96], [64, 160, 0], [0, 64, 0], [192, 128, 224],
            [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0],
]
palette = [value for color in palette for value in color]


def get_indices(classes, prompts):
    indices, class_labels = [], []
    
    for prompt in prompts:
        curr_indices, curr_labels = [], [0]
        index = 0
        class_prompt = prompt['class_prompt']
        prompt = prompt['caption']
        while class_prompt != "":
            for i, cls in enumerate(classes):
                if class_prompt.startswith(cls) and ((cls == class_prompt) or (class_prompt[len(cls)] == " ")):
                    if cls in COMPOUND_NOUNS:
                        curr_indices.append([index + i for i in range(len(cls.split(" ")))])
                    else:
                        curr_indices.append(index)
                    curr_labels.append(i)
                    index += len(cls.split(" "))
                    class_prompt = class_prompt[len(cls):].strip()

        indices.append(curr_indices)
        class_labels.append(curr_labels)

    return indices, class_labels
