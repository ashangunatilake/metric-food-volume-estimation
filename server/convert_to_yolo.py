import os
import cv2
import numpy as np
from tqdm import tqdm

# -------- CONFIG --------
SRC_ROOT = "FoodSeg103"
DST_ROOT = "foodseg_yolo"

IMG_TRAIN = os.path.join(SRC_ROOT, "Images/img_dir/train")
MASK_TRAIN = os.path.join(SRC_ROOT, "Images/ann_dir/train")

IMG_VAL = os.path.join(SRC_ROOT, "Images/img_dir/test")
MASK_VAL = os.path.join(SRC_ROOT, "Images/ann_dir/test")

# ------------------------

def ensure_dirs():
    for split in ["train", "val"]:
        os.makedirs(f"{DST_ROOT}/images/{split}", exist_ok=True)
        os.makedirs(f"{DST_ROOT}/labels/{split}", exist_ok=True)

def process_split(img_dir, mask_dir, split):
    images = os.listdir(img_dir)

    for img_name in tqdm(images, desc=f"Processing {split}"):
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name.replace(".jpg", ".png"))

        if not os.path.exists(mask_path):
            continue

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        label_lines = []

        for cnt in contours:
            if cv2.contourArea(cnt) < 100:  # ignore tiny regions
                continue

            cnt = cnt.squeeze()
            if len(cnt.shape) != 2:
                continue

            # Normalize coordinates
            polygon = []
            for x, y in cnt:
                polygon.append(x / w)
                polygon.append(y / h)

            line = "0 " + " ".join(map(str, polygon))
            label_lines.append(line)

        if not label_lines:
            continue

        # Save image
        dst_img = f"{DST_ROOT}/images/{split}/{img_name}"
        cv2.imwrite(dst_img, img)

        # Save label
        label_name = img_name.replace(".jpg", ".txt")
        dst_lbl = f"{DST_ROOT}/labels/{split}/{label_name}"
        with open(dst_lbl, "w") as f:
            f.write("\n".join(label_lines))


def write_yaml():
    yaml_text = f"""
path: {DST_ROOT}
train: images/train
val: images/val

names:
  0: food
"""
    with open(f"{DST_ROOT}/data.yaml", "w") as f:
        f.write(yaml_text.strip())


if __name__ == "__main__":
    ensure_dirs()
    process_split(IMG_TRAIN, MASK_TRAIN, "train")
    process_split(IMG_VAL, MASK_VAL, "val")
    write_yaml()
    print("✅ Conversion complete")
