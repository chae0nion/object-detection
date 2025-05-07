import os
import pandas as pd
import random

# (1) summary.csv 로드
summary_path = r"C:\Users\lisat\Desktop\pcd\training\summary.csv"
df = pd.read_csv(summary_path)

# (2) 분할 결과 저장 경로
imageset_dir = os.path.join(os.path.dirname(summary_path), "ImageSets")
os.makedirs(imageset_dir, exist_ok=True)

# (3) split 목록 초기화
splits = {"train": [], "val": [], "test": []}

# (4) 재질 × 거리 조합으로 groupby
for (material, distance), group in df.groupby(["material", "distance"]):
    group = group.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    total = len(group)

    n_train = min(6, total)
    n_val = min(2, total - n_train)
    n_test = total - n_train - n_val

    splits["train"].extend(group.iloc[:n_train]["index"])
    splits["val"].extend(group.iloc[n_train:n_train + n_val]["index"])
    splits["test"].extend(group.iloc[n_train + n_val:]["index"])

# (5) .txt 파일로 저장
for split_name, id_list in splits.items():
    with open(os.path.join(imageset_dir, f"{split_name}.txt"), "w") as f:
        for idx in sorted(id_list):
            f.write(f"{idx}\n")

print("✅ train/val/test split 완료! ImageSets 폴더에 저장됨.")
