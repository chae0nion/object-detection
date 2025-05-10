import pandas as pd
import glob
import os

# CSV 파일들이 있는 폴더 경로
csv_folder = 'csvs'  # 예: './gt_csvs/'
output_path = 'gt_centers.csv'

results = []

for csv_file in sorted(glob.glob(os.path.join(csv_folder, '*.csv'))):
    df = pd.read_csv(csv_file)

    center_x = df['Points_0'].mean()
    center_y = df['Points_1'].mean()
    center_z = df['Points_2'].mean() + 1.1  # z는 1.1씩 올림

    # 파일 이름에서 확장자 제거 (예: car_4m.csv → car_4m)
    filename = os.path.splitext(os.path.basename(csv_file))[0]

    results.append({
        'filename': filename,
        'center_x': center_x,
        'center_y': center_y,
        'center_z': center_z
    })

# 결과 저장
gt_df = pd.DataFrame(results)
gt_df.to_csv(output_path, index=False)

print(f'완료: {output_path}에 저장됨')

