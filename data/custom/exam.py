import os

label_dir = './labels'  # 너의 실제 label 경로로 수정
empty_files = []

for fname in sorted(os.listdir(label_dir)):
    if not fname.endswith('.txt'):
        continue
    path = os.path.join(label_dir, fname)

    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]  # 빈 줄 제거

    if len(lines) == 0:
        empty_files.append(fname)
    else:
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 8:
                empty_files.append(fname)
                break  # 하나라도 이상하면 break

print(f"🚨 GT 없는 라벨 파일 {len(empty_files)}개 발견:")
for f in empty_files:
    print(f" - {f}")
