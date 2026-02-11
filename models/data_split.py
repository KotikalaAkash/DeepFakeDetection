import os
import random
import shutil

SOURCE_DIR = "C:\\Users\\RGUKT\\Desktop\\deepfake\\CelebDF(v2)"
DEST_DIR = "C:\\Users\\RGUKT\\Desktop\\deepfake\\CelebDF_final"

SPLIT_RATIO = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

classes = os.listdir(SOURCE_DIR)

for cls in classes:
    cls_path = os.path.join(SOURCE_DIR, cls)
    files = os.listdir(cls_path)
    random.shuffle(files)

    total = len(files)
    train_end = int(SPLIT_RATIO["train"] * total)
    val_end = train_end + int(SPLIT_RATIO["val"] * total)

    splits = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }

    for split, split_files in splits.items():
        split_dir = os.path.join(DEST_DIR, split, cls)
        os.makedirs(split_dir, exist_ok=True)

        for f in split_files:
            shutil.copy(
                os.path.join(cls_path, f),
                os.path.join(split_dir, f)
            )

print(" Dataset split completed!")