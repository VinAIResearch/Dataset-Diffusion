import argparse
import json
import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from src.attn_processor import (
    AttentionStoreClassPrompts,
    StoredAttnClassPromptsProcessor,
    aggregate_attention,
    register_attention_control,
)
from src.pipeline_class_prompts import StableDiffusionClassPromptsPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Diffusion")
    parser.add_argument("--work-dir", help="the dir to save the synthetic dataset")
    parser.add_argument("--sd-path", help="stable diffusion path")
    parser.add_argument("--json-path", default="data/prompts/voc_prompts.json")
    parser.add_argument("--data-type", choices=["voc", "coco"], default="voc")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--self-res", default=32, type=int)
    parser.add_argument("--cross-res", default=16, type=int)
    parser.add_argument("--threshold", default=0.6, type=float)
    parser.add_argument("--uncertainty-threshold", default=0.5, type=float)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1111)
    args = parser.parse_args()

    return args


def main(args):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    os.makedirs(f"{args.work_dir}/image", exist_ok=True)
    os.makedirs(f"{args.work_dir}/mask", exist_ok=True)

    #   we need two version for `get_indices` to handle the compound class names of 2 datasets
    #   e.g. `diningtable` is a VOC class, while `dining table` is a COCO class.
    if args.data_type == "voc":
        from src.datasets.voc import classes, get_indices, palette
    elif args.data_type == "coco":
        from src.datasets.coco import classes, get_indices, palette
    else:
        raise ValueError(f"Data type: {args.data_type} is not supported. Currently support: `voc`, `coco`.")
    self_res = args.self_res
    cross_res = args.cross_res
    pipe = StableDiffusionClassPromptsPipeline.from_pretrained(args.sd_path, torch_dtype=torch.float16).to(device)
    pipe.enable_attention_slicing()

    #   define the range of timesteps that we want to extract the self/cross attention maps
    #   0 < start < end < num_inference_steps
    controller = AttentionStoreClassPrompts(start=0, end=100)
    register_attention_control(pipe, controller, StoredAttnClassPromptsProcessor)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    with open(args.json_path) as f:
        prompts = json.load(f)

    indices, labels = get_indices(classes, prompts)
    batch_size = args.batch_size
    start_index = max(0, args.start)
    if args.end == -1:
        end_index = len(prompts)
    else:
        end_index = min(len(prompts), args.end)

    negative_prompt = (
        "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face,"
        + "out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy,"
        + "watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face"
    )
    for i in range(start_index, end_index, batch_size):
        batch = prompts[i : i + batch_size]
        batch_filenames = [x["filename"] for x in batch]
        batch_prompts = [x["caption"] for x in batch]
        batch_class_prompts = [x["class_prompt"] for x in batch]
        batch_indices = indices[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]

        output = pipe(
            batch_prompts,
            class_prompts=batch_class_prompts,
            negative_prompt=[negative_prompt] * len(batch),
            num_inference_steps=100,
            generator=generator,
            output_type="numpy",
        )

        for j in range(len(batch)):
            base_filename = osp.splitext(batch_filenames[j])[0]
            image = Image.fromarray((output.images[j] * 255).astype(np.uint8))
            image.save(f"{args.work_dir}/image/{base_filename}.jpg")

            w, h = image.size
            self_attention = aggregate_attention(controller, res=self_res, is_cross=False).float()
            cross_attention = aggregate_attention(controller, res=cross_res, is_cross=True).float()
            cross_attention = F.interpolate(cross_attention.permute(0, 3, 1, 2), (self_res, self_res), mode="bicubic")

            if len(batch_indices[j]) > 0:
                affinity_mat = self_attention[j].reshape(self_res**2, self_res**2)
                affinity_mat = torch.matrix_power(affinity_mat, 4)
                outs = []
                for index in batch_indices[j]:
                    if isinstance(index, list):
                        index = [i + 1 for i in index]
                        ca = cross_attention[j][index].mean(dim=0)
                    elif isinstance(index, int):
                        ca = cross_attention[j][index + 1]
                    out = (affinity_mat @ ca.reshape(self_res**2, 1)).reshape(self_res, self_res)
                    out = out - out.min()
                    out = out / out.max()
                    outs.append(out)
                outs = torch.stack(outs)
                outs = F.interpolate(outs.unsqueeze(0), (h, w), mode="bicubic")[0].cpu().numpy()
                outs_max = outs.max(axis=0)
                mask = np.zeros((h, w), dtype=np.uint8)
                valid = outs_max >= args.threshold
                mask[valid] = (outs.argmax(axis=0) + 1)[valid]
                label = np.array(batch_labels[j], dtype=np.uint8)
                mask = label[mask]

                if args.uncertainty_threshold is not None:
                    assert args.uncertainty_threshold < args.threshold
                    ignore_pixels = (outs_max < args.threshold) & (outs_max >= args.uncertainty_threshold)
                    mask[ignore_pixels] = 255

                save_name = f"{args.work_dir}/mask/{base_filename}.png"

                mask = Image.fromarray(mask.astype(np.uint8)).convert("P")
                mask.putpalette(palette)
                mask.save(save_name)

        controller.reset()


if __name__ == "__main__":
    args = parse_args()
    main(args)
