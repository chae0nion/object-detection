import argparse
import glob
import os
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V

    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V

    OPEN3D_FLAG = False

import numpy as np
import torch
import pandas as pd

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training,
                         root_path=root_path, logger=logger)
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


MATERIALS = ['card', 'mirror', 'nothing', 'spot', 'velvet']


def infer_material(index):
    local_index = index % 100  # 거리 내 인덱스 (0~99)
    return MATERIALS[local_index // 20]  # 20개 단위로 재질 구분


def parse_config():
    parser = argparse.ArgumentParser(description='OpenPCDet Demo')
    parser.add_argument('--cfg_file', type=str, required=True, help='Model config file')
    parser.add_argument('--data_path', type=str, required=True, help='Point cloud file or directory')
    parser.add_argument('--ckpt', type=str, required=True, help='Pretrained model path')
    parser.add_argument('--gt_csv', type=str, required=True, help='Path to GT centers CSV file')
    parser.add_argument('--ext', type=str, default='.npy', help='Point cloud file extension (.bin or .npy)')
    return parser.parse_args()


def main():
    args = parse_config()
    logger = common_utils.create_logger()
    distances = ['4m', '7m', '10m']
    global cfg

    Path("demo_output").mkdir(parents=True, exist_ok=True)

    all_metrics = []  # CSV 저장용 리스트

    for folder in distances:
        print(f"\n=== Processing {folder} ===")

        cfg = cfg_from_yaml_file(args.cfg_file, cfg)

        data_path = Path(args.data_path) / folder
        demo_dataset = DemoDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, False, data_path, None, args.ext)

        model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), demo_dataset)
        model.load_params_from_file(args.ckpt, logger=logger, to_cpu=True)
        model.cuda().eval()

        gt_df = pd.read_csv(args.gt_csv)
        gt_row = gt_df[gt_df['filename'].str.contains(folder)].iloc[0]
        gt_boxes = np.array([[gt_row['center_x'], gt_row['center_y'], gt_row['center_z'], 2.0, 4.0, 1.5, 0.0]],
                            dtype=np.float32)

        with torch.no_grad():
            for idx, data_dict in enumerate(demo_dataset):
                data_dict = demo_dataset.collate_batch([data_dict])
                load_data_to_gpu(data_dict)
                pred_dicts, _ = model.forward(data_dict)

                pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
                pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
                pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()

                material = infer_material(idx)
                output_dir = Path("demo_output") / folder / material
                output_dir.mkdir(parents=True, exist_ok=True)

                np.save(output_dir / f"pred_{idx:06d}.npy", pred_dicts[0])
                image_path = output_dir / f"{idx:06d}.png"

                V.draw_scenes(
                    points=data_dict['points'][:, 1:],
                    ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'],
                    ref_labels=pred_dicts[0]['pred_labels'],
                    gt_boxes=gt_boxes,
                    save_to_file=True,
                    output_path=str(image_path)
                )

                # 결과 저장용 row 생성
                # 결과 저장용 row 생성
                for i, box in enumerate(pred_boxes):
                    if pred_scores[i] >= 0.5:  # 신뢰도 필터링
                        row = {
                            'frame': idx,
                            'material': material,
                            'distance': folder,
                            'box_id': i,
                            'x': box[0], 'y': box[1], 'z': box[2],
                            'dx': box[3], 'dy': box[4], 'dz': box[5],
                            'heading': box[6],
                            'score': float(pred_scores[i]),
                            'label': int(pred_labels[i])
                        }
                        all_metrics.append(row)

    # CSV로 결과 저장
    summary_df = pd.DataFrame(all_metrics)
    summary_df.to_csv("demo_output/material_detection_summary.csv", index=False)
    print("✅ 결과 CSV 저장 완료 → demo_output/material_detection_summary.csv")


if __name__ == '__main__':
    main()

