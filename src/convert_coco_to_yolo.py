import json
import shutil
from pathlib import Path

DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
ANN_DIR = DATA_DIR / "annotations"
LABELS_DIR = DATA_DIR / "labels"

splits = ["train", "val", "test"]

def convert_split(split):
    ann_file = ANN_DIR / f"{split}.json"
    if not ann_file.exists():
        print(f"Annotation file not found: {ann_file}")
        return

    print(f"\nConverting {split}...")

    with open(ann_file, "r") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}

    # YOLO label output folder
    (LABELS_DIR / split).mkdir(exist_ok=True, parents=True)
    (DATA_DIR / "images" / split).mkdir(exist_ok=True, parents=True)

    # Create blank label files
    for img in data["images"]:
        label_path = LABELS_DIR / split / (img["file_name"].replace(".jpg", ".txt"))
        label_path.touch()

    # Convert each annotation
    for ann in data["annotations"]:
        img = images[ann["image_id"]]

        w = img["width"]
        h = img["height"]

        # COCO bbox format: [x, y, width, height]
        x, y, bw, bh = ann["bbox"]

        # YOLO format: class cx cy w h (normalized)
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h

        label_file = LABELS_DIR / split / (img["file_name"].replace(".jpg", ".txt"))
        with open(label_file, "a") as lf:
            lf.write(f"{ann['category_id']} {cx} {cy} {nw} {nh}\n")

    # Move images into split folders
    for img in data["images"]:
        src = IMAGES_DIR / img["file_name"]
        dst = IMAGES_DIR / split / img["file_name"]
        if src.exists():
            shutil.copy(src, dst)

    print(f"Finished {split} conversion.")

for split in splits:
    convert_split(split)

print("\nAll splits converted successfully!")

