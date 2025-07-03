import decord
import os
import numpy as np
import cv2
from typing import List, Tuple

import torch
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from config import config
gpu_id = config["gpu_id"]
device_name = f"cuda:{gpu_id}" if (torch.cuda.is_available() and gpu_id < torch.cuda.device_count()) else "cpu"

def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 10 * np.log10(max_pixel**2 / mse)

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    return ssim(img1, img2, multichannel=True, data_range=255, channel_axis=0)

def decode_video(filename: str) -> Tuple[np.ndarray, float]:
    video_reader = decord.VideoReader(
        filename,
        ctx=decord.cpu(0) if device_name == "cpu" else decord.gpu(int(device_name.split(":")[1])),
        num_threads=4,  # 多线程加速
    )
    frames = video_reader.get_batch(range(len(video_reader))).asnumpy()  # 转为numpy数组
    frames = frames.transpose(0, 3, 1, 2)
    fps = video_reader.get_avg_fps()
    print(filename, "gt_shape:", frames.shape)
    return frames, fps

def decode_bytes_to_numpy(bytes_data: bytes) -> Tuple[np.ndarray, float]:
    with open("/dev/shm/decord_temp.tmp", "wb") as f:
        f.write(bytes_data)
    frames, fps = decode_video("/dev/shm/decord_temp.tmp")
    try:
        os.remove("/dev/shm/decord_temp.tmp")
    except FileNotFoundError:
        pass
    return frames, fps

def load_gt_roi(w_gt: int, h_gt: int, gt_roi_folder: str) -> List[Tuple[int, int, int, int]]:
    gt_roi_list = []
    roi_files = sorted(os.listdir(gt_roi_folder))  # 确保文件顺序正确
    for fname in roi_files:
        with open(os.path.join(gt_roi_folder, fname)) as f:
            parts = list(map(float, f.readline().split()))
            # 解析YOLO格式标注
            xc, yc = parts[1] * w_gt, parts[2] * h_gt
            w, h = parts[3] * w_gt, parts[4] * h_gt
            x1 = int(xc - w / 2)
            y1 = int(yc - h / 2)
            x2 = int(xc + w / 2)
            y2 = int(yc + h / 2)
            # 边界约束
            # print(x1, y1, x2, y2)
            gt_roi_list.append((x1, y1, x2, y2))
    return gt_roi_list

def metric_frames(gt_frames: np.ndarray, lr_frames: np.ndarray, roi_frames: np.ndarray,
                  detect_roi: List[Tuple[int, int, int, int]],
                  gt_roi: List[Tuple[int, int, int, int]],
                  idx: int=0) -> List[Tuple[float, float, float, float]]:
    # 生成放大后的帧
    B, C, H_gt, W_gt = gt_frames.shape
    metrics = []

    print(gt_frames.shape, lr_frames.shape, roi_frames.shape)
    folder = f"client_result"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 创建视频编码器
    video_path = f"{folder}/roi_highlighted_{idx}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (W_gt, H_gt))

    for i in range(B):
        # 双三次插值放大
        lr = lr_frames[i].transpose(1, 2, 0)
        scaled = cv2.resize(lr, (W_gt, H_gt), interpolation=cv2.INTER_CUBIC)
        scaled = scaled.transpose(2, 0, 1)  # HWC -> CHW

        # ROI坐标转换
        x1_lr, y1_lr, x2_lr, y2_lr = detect_roi[i]
        x1, y1 = x1_lr * 4, y1_lr * 4
        x2, y2 = x2_lr * 4, y2_lr * 4
        assert roi_frames.shape[2] == y2 - y1, roi_frames.shape[3] == x2 - x1
        if i % 60 == 0:
            cv2.imwrite(f"{folder}/scaled_no_replace{i}.png", scaled.transpose(1, 2, 0)[:, :, ::-1])
        scaled[:, y1:y2, x1:x2] = roi_frames[i]

        gt = gt_frames[i]

        # 转换为OpenCV格式 (HWC, BGR)
        scaled_viz = scaled.transpose(1, 2, 0)[:, :, ::-1].copy()  # RGB to BGR
        scaled_viz = np.ascontiguousarray(scaled_viz, dtype=np.uint8)
        cv2.rectangle(scaled_viz, (x1, y1), (x2, y2), (0, 255, 0), 2) # 绘制检测ROI (绿色框)
        x1_gt, y1_gt, x2_gt, y2_gt = gt_roi[i]
        cv2.rectangle(scaled_viz, (x1_gt, y1_gt), (x2_gt, y2_gt), (0, 0, 255), 2) # 绘制真实ROI (红色框)
        # 添加文字说明
        cv2.putText(scaled_viz, f"Frame: {i}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(scaled_viz, "Detected ROI", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(scaled_viz, "Ground Truth ROI", (x1_gt, y1_gt-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        # 写入视频帧
        video_writer.write(scaled_viz)

        # 全图指标
        full_psnr = compute_psnr(scaled, gt)
        full_ssim = compute_ssim(scaled, gt)

        # ROI指标
        # print(gt_roi[i])
        x1, y1, x2, y2 = gt_roi[i]
        gt_roi_frame = gt[:, y1:y2, x1:x2]
        scaled_roi_frame = scaled[:, y1:y2, x1:x2]

        if gt_roi_frame.size == 0 or scaled_roi_frame.size == 0:
            roi_psnr = roi_ssim = 0.0
            print(f"Alert: gt_roi.size: {gt_roi_frame.size}, scaled_roi.size: {scaled_roi_frame.size}")
        else:
            roi_psnr = compute_psnr(scaled_roi_frame, gt_roi_frame)
            roi_ssim = compute_ssim(scaled_roi_frame, gt_roi_frame)

        metrics.append(((idx - 1) * 120 + i, full_psnr, full_ssim, roi_psnr, roi_ssim))
        print(f"Frame {(idx - 1) * 120 + i}: Full PSNR: {full_psnr:.2f}, Full SSIM: {full_ssim:.4f}, "
              f"RoI PSNR: {roi_psnr:.2f}, RoI SSIM: {roi_ssim:.4f}")

    video_writer.release()
    print(f"Saved ROI visualization video to: {video_path}")
    return metrics

def metric_normal(gt_frames: np.ndarray, sr_frames: np.ndarray,
                  gt_roi: List[Tuple[int, int, int, int]],
                  idx: int=0) -> List[Tuple[float, float, float, float]]:
    # 生成放大后的帧
    B, C, H_gt, W_gt = gt_frames.shape
    metrics = []

    folder = f"client_result/{idx}"
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in range(B):
        gt = gt_frames[i]
        sr = sr_frames[i]

        # 全图指标
        full_psnr = compute_psnr(sr, gt)
        full_ssim = compute_ssim(sr, gt)

        # ROI指标
        x1, y1, x2, y2 = gt_roi[i]
        gt_roi_frame = gt[:, y1:y2, x1:x2]
        scaled_roi_frame = sr[:, y1:y2, x1:x2]

        if gt_roi_frame.size == 0 or scaled_roi_frame.size == 0:
            roi_psnr = roi_ssim = 0.0
            print(f"Alert: gt_roi.size: {gt_roi_frame.size}, scaled_roi.size: {scaled_roi_frame.size}")
        else:
            roi_psnr = compute_psnr(scaled_roi_frame, gt_roi_frame)
            roi_ssim = compute_ssim(scaled_roi_frame, gt_roi_frame)

        metrics.append(((idx - 1) * 120 + i, full_psnr, full_ssim, roi_psnr, roi_ssim))
        print(f"Frame {(idx - 1) * 120 + i}: Full PSNR: {full_psnr:.2f}, Full SSIM: {full_ssim:.4f}, "
              f"RoI PSNR: {roi_psnr:.2f}, RoI SSIM: {roi_ssim:.4f}")

        # if i % 40 == 0:
        #     cv2.imwrite(f"{folder}/sr_{i}.png", sr.transpose(1, 2, 0)[:, :, ::-1])
        #     cv2.imwrite(f"{folder}/sr_roi_{i}.png", scaled_roi_frame.transpose(1, 2, 0)[:, :, ::-1])
        #     cv2.imwrite(f"{folder}/gt_{i}.png", gt.transpose(1, 2, 0)[:, :, ::-1])
        #     cv2.imwrite(f"{folder}/gt_roi_{i}.png", gt_roi_frame.transpose(1, 2, 0)[:, :, ::-1])

    return metrics

def save_metric(metric_res, filename: str):
    full_psnr = []
    full_ssim = []
    roi_psnr = []
    roi_ssim = []
    for i in range(len(metric_res)):
        full_psnr.append(metric_res[i][1])
        full_ssim.append(metric_res[i][2])
        roi_psnr.append(metric_res[i][3])
        roi_ssim.append(metric_res[i][4])
    full_psnr = sum(full_psnr) / len(full_psnr)
    full_ssim = sum(full_ssim) / len(full_ssim)
    roi_psnr = sum(roi_psnr) / len(roi_psnr)
    roi_ssim = sum(roi_ssim) / len(roi_ssim)
    print(f"Full PSNR: {full_psnr:.2f}, Full SSIM: {full_ssim:.4f}, "
          f"RoI PSNR: {roi_psnr:.2f}, RoI SSIM: {roi_ssim:.4f}")
    metric_res.append(("avg", full_psnr, full_ssim, roi_psnr, roi_ssim))
    df = pd.DataFrame(list(metric_res), columns=['frame', 'full_psnr', 'full_ssim', 'roi_psnr', 'roi_ssim'])
    df.to_csv(filename, index=False)