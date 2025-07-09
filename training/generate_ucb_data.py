import copy
import os
import json

import cv2
import torch
import numpy as np
import time
import csv
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional

from super_resolution.infer import Inferrer, generate_sr_patch
from config import device_name, config
from utils.client_metric import decode_video, decode_bytes_to_numpy
from utils.utils import roi_center_to_xyxy, ffmpeg_tensor_to_bytes

from skimage.metrics import structural_similarity as ssim

device = torch.device(device_name)
dnn_pth_list = config["sr_models"]

csv_header = [
        'folder', 'segment', 'chunk_frame', 'model_idx', 'sr_size',
        'patch_gen_time', 'sr_time', 'encode_time', 'send_size', 'ssim', 'roi_ssim'
    ]

chunk_frame = 60

# 初始化超分模型
class SimulatedSR:
    def __init__(self):
        torch.manual_seed(42)
        self.inferrer = Inferrer(dnn_pth_list)
        self.benchmark_coeffs = self.inferrer.run_benchmark()

    def perform(self, tensors: torch.Tensor, roi_xyxy: np.ndarray, SR_size: int, action: int):
        st = time.perf_counter()
        tensor_for_SR = generate_sr_patch(tensors, roi_xyxy)
        patch_gen_time = time.perf_counter() - st
        st = time.perf_counter()
        sr = self.inferrer.super_resolution(tensor_for_SR, SR_size, action)
        sr_time = time.perf_counter() - st
        return sr, patch_gen_time, sr_time


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    return ssim(img1, img2, multichannel=True, data_range=255, channel_axis=0)

# 主处理函数
def process_folder(root_path: str, folder: str, sr: SimulatedSR) -> List[Dict[str, Any]]:
    """处理单个文件夹中的所有数据"""
    stats = []

    # 解析文件夹结构
    json_path = os.path.join(root_path, f"{folder}.json")
    data_folder = os.path.join(root_path, folder)
    video_720p = None
    for file in os.listdir(data_folder):
        if file.startswith("720p") and file.endswith(".mp4"):
            video_720p = file
    assert video_720p, "720p video is not found"

    # 加载JSON数据
    with open(json_path, 'r') as f:
        annotation_data = json.load(f)

    # 加载720p视频
    ndarray_720p, fps = decode_video(os.path.join(data_folder, video_720p)) # NCHW, RGB

    init_path = os.path.join(data_folder, "init-stream0.m4s")
    with open(init_path, 'rb') as f:
        init_data = f.read()

    # 处理每个分片 (1-12)
    for seg_idx in tqdm(range(1, 13), desc="Processing segments"):
        with open(os.path.join(data_folder, f"chunk-stream0-{seg_idx:05d}.m4s"), 'rb') as f:
            chunk_data = f.read()

        video_bytes = init_data + chunk_data
        ndarray_180p, fps_ = decode_bytes_to_numpy(video_bytes)
        assert fps == fps_

        scaled = np.zeros((ndarray_180p.shape[0], *(ndarray_720p.shape[1:])))
        for i in range(ndarray_180p.shape[0]):
            scaled[i] = cv2.resize(ndarray_180p[i].transpose(1, 2, 0), (ndarray_720p.shape[3], ndarray_720p.shape[2]),
                            interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)

        tensors_div_255 = torch.from_numpy(ndarray_180p).float().div(255)
        T, C, H_lr, W_lr = tensors_div_255.shape

        roi_center = []
        for idx in range(T):
            global_idx = (seg_idx - 1) * chunk_frame + idx
            frame_anno = annotation_data['annotations'][str(global_idx)]
            x, y, w, h = frame_anno
            center_x = x + w // 2
            center_y = y + h // 2
            roi_center.append((center_x, center_y))

        for sr_size in [60, 70, 80, 90, 100, 110, 120]:
            roi_list = roi_center_to_xyxy(roi_center, sr_size, (ndarray_180p.shape[2], ndarray_180p.shape[3]))
            for action in range(len(dnn_pth_list)):
                sr_patch, patch_gen_time, sr_time = sr.perform(tensors_div_255, roi_list, sr_size, action)

                st = time.perf_counter()
                reencode = ffmpeg_tensor_to_bytes(sr_patch, fps, 'generator')
                encode_time = time.perf_counter() - st

                sr_numpy, fps = decode_bytes_to_numpy(reencode)

                # metrics
                avg_ssim = 0
                avg_ssim_roi = 0
                for idx in range(sr_numpy.shape[0]):
                    global_idx = (seg_idx - 1) * chunk_frame + idx
                    x1_lr, y1_lr, x2_lr, y2_lr = roi_list[idx]
                    x1, y1 = x1_lr * 4, y1_lr * 4
                    x2, y2 = x2_lr * 4, y2_lr * 4
                    assert sr_numpy.shape[2] == y2 - y1, sr_numpy.shape[3] == x2 - x1

                    scaled_copy = copy.copy(scaled[idx])
                    scaled_copy[:, y1:y2, x1:x2] = sr_numpy[idx]
                    gt = ndarray_720p[global_idx]
                    avg_ssim += compute_ssim(gt, scaled_copy)

                    x, y, w, h = annotation_data['annotations'][str(global_idx)]
                    x1, x2 = x * 4, (x + w) * 4
                    y1, y2 = y * 4, (y + h) * 4
                    gt_roi = gt[:, y1:y2, x1:x2]
                    scaled_roi = scaled_copy[:, y1:y2, x1:x2]
                    avg_ssim_roi += compute_ssim(gt_roi, scaled_roi)
                avg_ssim /= sr_numpy.shape[0]
                avg_ssim_roi /= sr_numpy.shape[0]

                stat_dict = {'folder': data_folder,
                             'segment': seg_idx,
                             'chunk_frame': chunk_frame,
                             'model_idx': action,
                             'sr_size': sr_size,
                             'patch_gen_time': patch_gen_time,
                             'sr_time': sr_time,
                             'encode_time': encode_time,
                             'send_size': 4 + len(video_bytes) + 9 + len(reencode) + 8 * sr_numpy.shape[0],
                             'avg_ssim': avg_ssim,
                             'avg_ssim_roi': avg_ssim_roi}
                stats.append(stat_dict)
    return stats

def main():
    # 初始化超分系统
    sr_system = SimulatedSR()

    # 配置路径
    root_dir = r"../#semiauto-roi-labeler/180P-annotated/5.9"
    output_csv = "bandit_metrics.csv"

    data_list = []

    # 处理每个文件夹
    for folder_name in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_name}")
            data_list.append(process_folder(root_dir, folder_name, sr_system))

    with open(output_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        for data in data_list:
            row = []
            for item in csv_header:
                row.append(data[item])
            writer.writerow(row)


if __name__ == "__main__":
    main()