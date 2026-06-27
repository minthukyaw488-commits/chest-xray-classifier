import os
import shutil
import random
from pathlib import Path

random.seed(42)  # reproducibility

SRC_TRAIN = Path("data/chest_xray/train")
DST_VAL = Path("data/chest_xray/val_new")
VAL_RATIO = 0.15  # 15% of train goes to val

# Create new val folder
DST_VAL.mkdir(parents=True, exist_ok=True)

for class_name in ["NORMAL", "PNEUMONIA"]:
    src_dir = SRC_TRAIN / class_name
    dst_dir = DST_VAL / class_name
    dst_dir.mkdir(exist_ok=True)
    
    images = list(src_dir.glob("*.jpeg")) + list(src_dir.glob("*.jpg"))
    random.shuffle(images)
    
    n_val = int(len(images) * VAL_RATIO)
    val_images = images[:n_val]
    
    print(f"{class_name}: moving {n_val} / {len(images)} images to val")
    
    for img in val_images:
        shutil.move(str(img), str(dst_dir / img.name))

print("\n✅ New validation set created at data/chest_xray/val_new")
print(f"Train: {len(list((SRC_TRAIN/'NORMAL').glob('*')))} NORMAL + {len(list((SRC_TRAIN/'PNEUMONIA').glob('*')))} PNEUMONIA")
print(f"Val:   {len(list((DST_VAL/'NORMAL').glob('*')))} NORMAL + {len(list((DST_VAL/'PNEUMONIA').glob('*')))} PNEUMONIA")