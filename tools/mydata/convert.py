import os
import numpy as np
import pandas as pd

# (1) PCD 파일에서 [x, y, z, intensity] 읽기
def read_pcd_ascii_with_intensity(pcd_path, z_shift = 1.1):
    with open(pcd_path, 'r') as f:
        lines = f.readlines()

    start_idx = next(i for i, line in enumerate(lines) if line.strip().upper() == "DATA ASCII") + 1
    points = []
    for line in lines[start_idx:]:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        x, y, z, intensity = map(float, parts[:4])
        intensity = intensity / 255.0
        z += z_shift
        points.append([x, y, z, intensity])

    return np.array(points, dtype=np.float32)

# (2) 경로 설정
input_root = r"C:\Users\lisat\Desktop\pcd"  # 4m, 7m, 10m 폴더가 이 안에 있어야 함
output_root = r"C:\Users\lisat\Desktop\pcd\training"
velodyne_dir = os.path.join(output_root, "velodyne")
label_dir = os.path.join(output_root, "label_2")
os.makedirs(velodyne_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

# (3) 설정
materials = ["nothing", "card", "mirror", "velvet", "spot"]
distances = ["4m", "7m", "10m"]
entries = []
index = 0

# (4) 변환 루프
for distance in distances:
    dist_folder = os.path.join(input_root, distance)
    for fname in sorted(os.listdir(dist_folder)):
        if not fname.endswith(".pcd"):
            continue

        pcd_path = os.path.join(dist_folder, fname)
        material = next((m for m in materials if m in fname.lower()), "unknown")

        try:
            points = read_pcd_ascii_with_intensity(pcd_path)

            # 저장 파일명
            bin_name = f"{index:06d}.bin"
            txt_name = f"{index:06d}.txt"

            # (5) .bin 저장
            points.tofile(os.path.join(velodyne_dir, bin_name))

            # (6) .txt 라벨 생성
            label_path = os.path.join(label_dir, txt_name)
            with open(label_path, "w") as f:
                f.write("Car 0.00 0 0.00 0 0 50 50 1.5 1.5 3.5 10.0 5.0 -1.2 1.57\n")

            # (7) summary 기록
            entries.append({
                "index": f"{index:06d}",
                "original_file": fname,
                "material": material,
                "distance": distance
            })

            print(f"[{index:06d}] 변환 완료: {fname}")
            index += 1

        except Exception as e:
            print(f"[ERROR] {fname} 변환 실패: {e}")

# (8) summary.csv 저장
summary_path = os.path.join(output_root, "summary.csv")
pd.DataFrame(entries).to_csv(summary_path, index=False)
print(f"\n✅ 총 {index}개 변환 완료! summary.csv 저장됨 → {summary_path}")
