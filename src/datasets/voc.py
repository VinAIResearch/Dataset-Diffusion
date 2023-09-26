#   Definition of classes which are existed in the dataset
#   For the order of classes, please refer to
#   https://github.com/open-mmlab/mmsegmentation/tree/main/mmseg/datasets

from nltk.tokenize import word_tokenize


classes = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

palette = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]
palette = [value for color in palette for value in color]


COMPOUND_NOUNS = {
    "diningtable": ["dining table"],
    "pottedplant": [
        "potted plant",
    ],
    "tvmonitor": [
        "tv monitor",
    ],
}


def handle_compound_nouns(prompt):
    for k, v in COMPOUND_NOUNS.items():
        for compound_noun in v:
            if compound_noun in prompt:
                prompt = prompt.replace(compound_noun, k)
    return prompt


def get_indices_class_prompt(classes, prompt):
    prompt = handle_compound_nouns(prompt)
    tokens = word_tokenize(prompt)
    curr_indices, curr_labels = [], [0]
    num_compound_nouns = 0
    for j, token in enumerate(tokens):
        for c_idx, c in enumerate(classes):
            if token == c:
                if token in COMPOUND_NOUNS:
                    curr_indices.append(list(range(j + num_compound_nouns, j + num_compound_nouns + 2)))
                    num_compound_nouns += 1
                else:
                    curr_indices.append(j + num_compound_nouns)
                curr_labels.append(c_idx)

    return curr_indices, curr_labels


def get_indices(classes, prompts):
    indices, class_labels = [], []
    for i, prompt in enumerate(prompts):
        class_prompt = prompt["class_prompt"]
        prompt = prompt["caption"]
        curr_indices, curr_labels = get_indices_class_prompt(classes, class_prompt)
        if len(curr_indices) != 0:
            indices.append(curr_indices)
            class_labels.append(curr_labels)

    return indices, class_labels
