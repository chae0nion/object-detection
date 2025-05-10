import os
import numpy as np

def read_pcd_ascii_with_real_intensity(pcd_path, z_shift=1.1):
    with open(pcd_path, 'r') as f:
        lines = f.readlines()

    # 헤더 파싱
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().upper() == "DATA ASCII":
            start_idx = i + 1
            break

    # 포인트 파싱
    points = []
    for line in lines[start_idx:]:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        x, y, z, intensity = map(float, parts[:4])
        z += z_shift
        intensity = intensity / 255.0
        points.append([x, y, z, intensity])

    return np.array(points, dtype=np.float32)

# (2) 경로 설정
input_root = r"./"  # '4m', '7m', '10m' 폴더가 이 안에 있어야 함
output_root = r"pcd_bin_split"
os.makedirs(output_root, exist_ok=True)

# (3) 설정
distances = ["4m", "7m", "10m"]
index = 0

# (4) 변환 루프
for distance in distances:
    dist_folder = os.path.join(input_root, distance)
    dist_output = os.path.join(output_root, distance)
    os.makedirs(dist_output, exist_ok=True)

    for fname in sorted(os.listdir(dist_folder)):
        if not fname.endswith(".pcd"):
            continue

        pcd_path = os.path.join(dist_folder, fname)

        try:
            points = read_pcd_ascii_with_real_intensity(pcd_path)

            bin_name = f"{index:06d}.bin"
            points.tofile(os.path.join(dist_output, bin_name))

            print(f"[{index:06d}] 변환 완료 → {distance}/{bin_name}")
            index += 1

        except Exception as e:
            print(f"[ERROR] {fname} 변환 실패: {e}")

print(f"\n✅ 총 {index}개 변환 완료! 각 거리 폴더에 저장됨 → {output_root}")

