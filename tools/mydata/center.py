import pandas as pd
import os

input_dir = r"./data"  # .csv들이 있는 폴더
output_path = os.path.join(input_dir, "gt_centers.csv")

z_shift = 1.1
entries = []

for fname in os.listdir(input_dir):
    if fname.endswith(".csv"):
        fpath = os.path.join(input_dir, fname)
        df = pd.read_csv(fpath)

        # 컬럼 이름 자동 매핑
        x_col = 'Points_0'
        y_col = 'Points_1'
        z_col = 'Points_2'

        # Z 보정
        df[z_col] = df[z_col] + z_shift

        # 중심 좌표 계산
        x, y, z = df[[x_col, y_col, z_col]].mean().values

        entries.append({
            'file': fname.replace('.csv', ''),
            'x': round(x, 3),
            'y': round(y, 3),
            'z': round(z, 3)
        })

pd.DataFrame(entries).to_csv(output_path, index=False)
print(f"✅ Z보정 적용된 중심좌표 저장 완료 → {output_path}")

