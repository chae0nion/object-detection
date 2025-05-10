import os
import numpy as np
import pandas as pd

def read_pcd_ascii_with_real_intensity(pcd_path, z_shift=1.1):
    with open(pcd_path, 'r') as f:
        lines = f.readlines()

    start_idx = next(i for i, line in enumerate(lines) if line.strip().upper() == "DATA ASCII") + 1
    points = []
    for line in lines[start_idx:]:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        x, y, z, intensity = map(float, parts[:4])
        x = y
        y = -x
        z += z_shift
        intensity /= 255.0
        points.append([x, y, z, intensity])
    return np.array(points, dtype=np.float32)

# 경로 설정
input_root = "baseline"
output_root = "./output_dataset"

velodyne_dir = os.path.join(output_root, "velodyne")
label_dir = os.path.join(output_root, "labels")
os.makedirs(velodyne_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

# GT 좌표 딕셔너리로 로딩
gt_df = pd.read_csv("baseline/gt_centers.csv")
gt_dict = {}
for _, row in gt_df.iterrows():
    if "4m" in row["filename"]:
        gt_dict["4m"] = (row["center_x"], row["center_y"], row["center_z"])
    elif "7m" in row["filename"]:
        gt_dict["7m"] = (row["center_x"], row["center_y"], row["center_z"])
    elif "10m" in row["filename"]:
        gt_dict["10m"] = (row["center_x"], row["center_y"], row["center_z"])

distances = ["4m", "7m", "10m"]
index = 0
for distance in distances:
    dist_folder = os.path.join(input_root, distance)
    for fname in sorted(os.listdir(dist_folder)):
        if not fname.endswith(".pcd"):
            continue

        pcd_path = os.path.join(dist_folder, fname)
        bin_name = f"{index:06d}.bin"
        label_name = f"{index:06d}.txt"

        try:
            points = read_pcd_ascii_with_real_intensity(pcd_path)
            points.tofile(os.path.join(velodyne_dir, bin_name))

            if distance in gt_dict:
                x, y, z = gt_dict[distance]
                dx, dz, dy, yaw = 2.0, 1.5, 4.0, 0.0
                label_line = f"{x:.2f} {y:.2f} {z:.2f} {dx:.2f} {dy:.2f} {dz:.2f} {yaw:.2f} Vehicle"

                with open(os.path.join(label_dir, label_name), 'w') as f:
                    f.write(label_line)

            print(f"[{index:06d}] 변환 완료")

        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

        index += 1

print(f"\n✅ 총 {index}개 변환 완료! → {velodyne_dir}, {label_dir}")

import os
import random

input_dir = "./output_dataset/velodyne"
imageset_dir = "./output_dataset/ImageSets"
os.makedirs(imageset_dir, exist_ok=True)

# 거리별 인덱스 범위 지정
dist_ranges = {
    "4m": range(0, 100),
    "7m": range(100, 200),
    "10m": range(200, 300),
}

train_ids = []
val_ids = []

for dist, indices in dist_ranges.items():
    ids = [f"{i:06d}" for i in indices]
    random.seed(42)
    random.shuffle(ids)
    train_ids.extend(ids[:80])
    val_ids.extend(ids[80:])

# 저장
with open(os.path.join(imageset_dir, 'train.txt'), 'w') as f:
    f.write('\n'.join(sorted(train_ids)))
with open(os.path.join(imageset_dir, 'val.txt'), 'w') as f:
    f.write('\n'.join(sorted(val_ids)))

print(f"✅ 거리별 균등 분할 완료: train={len(train_ids)}, val={len(val_ids)}")


