import json, os, zipfile
import numpy as np
from PIL import Image
import torch
from huggingface_hub import hf_hub_download


def load_data():
    os.makedirs("raw_data", exist_ok=True)
    for split in ["train", "test", "valid"]:
        path = hf_hub_download(
            repo_id="keremberke/satellite-building-segmentation",
            filename=f"data/{split}.zip",
            repo_type="dataset",
            local_dir="raw_data"
        )
        with zipfile.ZipFile(path, 'r') as z:
            z.extractall(f"raw_data/{split}")
        print(f"Extracted {split}")


def load_sam():
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    if not os.path.exists("sam_vit_h_4b8939.pth"):
        os.system("wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    sam.to(device)
    return SamAutomaticMaskGenerator(sam)


def compute_iou(mask_a, mask_b):
    intersection = np.sum(np.logical_and(mask_a, mask_b))
    union = np.sum(np.logical_or(mask_a, mask_b))
    return intersection / union if union > 0 else 0.0


def coco_bbox_to_mask(bbox, img_w, img_h):
    x, y, w, h = [int(v) for v in bbox]
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    mask[y:y+h, x:x+w] = 1
    return mask


def process_split(split_name, raw_dir, out_dir, mask_generator, limit=80, iou_threshold=0.3):
    with open(os.path.join(raw_dir, "_annotations.coco.json")) as f:
        coco = json.load(f)

    id_to_img = {img["id"]: img for img in coco["images"]}
    img_to_bboxes = {}
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        img_to_bboxes.setdefault(iid, []).append(ann["bbox"])

    images_out = os.path.join(out_dir, split_name, "images")
    masks_out  = os.path.join(out_dir, split_name, "masks")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(masks_out,  exist_ok=True)

    annotated = [(iid, info) for iid, info in id_to_img.items() if iid in img_to_bboxes][:limit]
    print(f"\nProcessing {split_name}: {len(annotated)} images")

    saved = 0
    for idx, (iid, img_info) in enumerate(annotated):
        img_path = os.path.join(raw_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue
        print(f"  [{idx+1}/{len(annotated)}] {img_info['file_name']}")
        try:
            image_np = np.array(Image.open(img_path).convert("RGB"))
            h, w = image_np.shape[:2]
            sam_masks = mask_generator.generate(image_np)
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            for bbox in img_to_bboxes[iid]:
                label_mask = coco_bbox_to_mask(bbox, w, h)
                for sm in sam_masks:
                    if compute_iou(sm["segmentation"].astype(np.uint8), label_mask) > iou_threshold:
                        combined_mask = np.logical_or(combined_mask, sm["segmentation"]).astype(np.uint8)
            Image.fromarray(image_np).save(os.path.join(images_out, f"{saved:04d}.png"))
            Image.fromarray(combined_mask * 255).save(os.path.join(masks_out, f"{saved:04d}.png"))
            saved += 1
        except Exception as e:
            print(f"  Error: {e}")

    print(f"  Saved {saved} pairs")


if __name__ == "__main__":
    load_data()
    mask_generator = load_sam()
    process_split("train", "raw_data/train", "dataset", mask_generator, limit=80)
    process_split("val",   "raw_data/valid", "dataset", mask_generator, limit=20)
    process_split("test",  "raw_data/test",  "dataset", mask_generator, limit=20)