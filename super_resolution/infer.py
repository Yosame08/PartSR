from .span_arch import SPAN
from basicsr.archs.edsr_arch import EDSR
from basicsr.archs.basicvsr_arch import BasicVSR

from typing import Optional, Tuple, List
from collections import OrderedDict

import os
import tqdm
import torch
import torch.nn.functional as F
import time
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

do_batch = True
from config import device_name, config, vsr_models
device = torch.device(device_name)
dnn_pth_list = config["sr_models"]
prefixes = ["super_resolution/models/", "models/"]

# 并行的超分辨率处理函数
def mp_server_sr(sr_queue, pipe_conn, result_dict, queue_lock, dnn_queue_counts):
    torch.manual_seed(42)
    inferrer = Inferrer(dnn_pth_list)
    # 将 run_benchmark 的结果发送回主进程
    pipe_conn.send(inferrer.run_benchmark())
    pipe_conn.close()  # 关闭管道连接
    while True:
        identifier, tensors, roi_list, action = sr_queue.get()
        if tensors is None:
            break
        result = inferrer.super_resolution(tensors, roi_list, action=action, replacement=False)
        result_dict[identifier] = result
        with queue_lock:
            dnn_queue_counts[action] -= 1

def mp_client_sr(sr_queue, result_dict):
    inferrer = Inferrer(dnn_pth_list)
    while True:
        identifier, tensors, roi_list, action = sr_queue.get()
        if tensors is None:
            break
        result = inferrer.super_resolution(tensors, roi_list, action=action, replacement=False)
        result_dict[identifier] = result

def measure_time(model: torch.nn.Module, w, h, vsr_model, warmup=3):
    """测量给定尺寸下的推理时间"""
    dummy_input = torch.randn(1, 3, h, w).to(device)
    if vsr_model:
        dummy_input = dummy_input.unsqueeze(0)  # (1, 1, 3, h, w)

    test_time = 10
    tot_time = 0
    max_time = 0
    min_time = float('inf')

    for _ in range(warmup):
        __ = model(dummy_input)

    for _ in range(test_time):
        start_time = time.perf_counter()
        __ = model(dummy_input)
        elapsed_time = time.perf_counter() - start_time
        if elapsed_time > max_time:
            max_time = elapsed_time
        if elapsed_time < min_time:
            min_time = elapsed_time
        tot_time += elapsed_time
    
    print(f"Model output shape: {__.shape}")
    return (tot_time - max_time - min_time) / (test_time - 2)

def load_basicsr_model(checkpoint):
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "params" in checkpoint:  # BasicSR 保存的权重文件
        state_dict = checkpoint["params"]
    else:
        state_dict = checkpoint

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

class Inferrer:
    def __init__(self, model_names: List[str]):
        self.models: Optional[List[torch.nn.Module]] = []
        for name in model_names:
            chosen = ""
            for prefix in prefixes:
                path = os.path.join(prefix, name)
                print(path)
                if os.path.exists(path):
                    chosen = path
                    break
            name = name.lower()
            if name.startswith("edsr"):
                if 'm' in name[4:]:
                    model = EDSR(3, 3, 64, 16)
                elif 'l' in name[4:]:
                    model = EDSR(3, 3, 256, 32, res_scale=0.1)
                else:
                    raise NotImplementedError(f"Cannot resolve EDSR type (M/L): {name}")
            elif name.startswith("span"):
                if "ch48" in name:
                    feature_channels = 48
                elif "ch52" in name:
                    feature_channels = 52
                else:
                    raise NotImplementedError(f"Cannot resolve span channels (ch48/ch52): {name}")
                if "x2" in name:
                    scale = 2
                elif "x4" in name:
                    scale = 4
                else:
                    raise NotImplementedError(f"Cannot resolve span scale (x2/x4): {name}")
                model = SPAN(3, 3, feature_channels, scale)
            elif name.startswith("basicvsr"):
                model = BasicVSR(64, 30, "super_resolution/spynet_sintel_final-3d2a1287.pth")
            else:
                raise NotImplementedError(f"Cannot resolve model type: {name}")

            if chosen == "":
                self.models.append(None)
                logger.warning(f"Cannot find model file: {name}")
            else:
                state_dict = torch.load(chosen, map_location="cpu", weights_only=True)
                print(model.load_state_dict(load_basicsr_model(state_dict), strict=True))
                model = model.to(device)
                for param in model.parameters():
                    param.requires_grad = False
                self.models.append(model)
        
    def run_benchmark(self) -> List[float]:
        print("running benchmark...")
        min_size = 80
        # max_size = 400
        max_size = 96
        step = 8
        sizes = range(min_size, max_size + 1, step)
        ret = []
        for idx, model in enumerate(self.models):
            x = []
            y = []
            time.sleep(0.01)
            for h in tqdm.tqdm(sizes):
                # for w in sizes:
                with torch.no_grad():
                    elapsed_time = measure_time(model, h, h, vsr_models[idx])
                x.append(h * h)
                y.append(elapsed_time)
                # print(f"size: {h}, time: {elapsed_time:.4f}s")
            linear_fit = np.polyfit(x, y, 1)  # 一次多项式拟合

            slope, intercept = linear_fit
            y_pred = np.polyval(linear_fit, x)
            residuals = y - y_pred
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            r_value = np.sqrt(r_squared)
            print(f"{type(model)}: y={slope:.4e}x+{intercept:.4e}, r = {r_value}")
            ret.append(float(slope))
        return ret
    
    def super_resolution(self, tensors: torch.Tensor, roi_list: np.ndarray, action: int, replacement: bool=False, pad: int=5) -> torch.Tensor:
        """
        逐样本处理不同尺寸ROI的优化版本

        Args:
            tensors: 输入图像批次 (B, C, H, W)
            roi_list: 每张图像的ROI坐标 [(x1, y1, x2, y2), ...]
            action: 选择进行超分辨率处理的模型
            replacement: 是否将原图像上采样后替换对应区域

        Returns:
            融合后的高分辨率图像 (B, C, H*4, W*4)
        """
        B, C, H, W = tensors.shape
        scale_factor = 4
        SR_size = roi_list[0][2] - roi_list[0][0]  # 计算 ROI 的宽度
        pad_size = SR_size + pad * 2

        # 阶段2: 裁剪 ROI 区域
        lr_patches = []
        positions = []
        for batch_idx in range(B):
            x1, y1, x2, y2 = roi_list[batch_idx]
            
            x1_exp = max(0, x1 - pad) 
            y1_exp = max(0, y1 - pad)  
            x2_exp = min(W, x2 + pad)
            y2_exp = min(H, y2 + pad)
            patch = tensors[batch_idx, :, y1_exp:y2_exp, x1_exp:x2_exp]  # (C, H, W)
            pad_left = pad - (x1 - x1_exp)
            pad_right = pad - (x2_exp - x2)
            pad_top = pad - (y1 - y1_exp)
            pad_bottom = pad - (y2_exp - y2)
            padded_patch = torch.nn.functional.pad(patch, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
            
            # lr_patch = tensors[batch_idx, :, y1:y2, x1:x2].unsqueeze(0)  # (1, C, h, w)
            lr_patch = padded_patch.unsqueeze(0)
            lr_patches.append(lr_patch)
            positions.append((x1, y1, x2, y2))
            if lr_patch.shape[-2:] != (pad_size, pad_size):
                print("SR input shape error:", x1, y1, x2, y2, lr_patch.shape, pad)

        # 阶段3: 批量超分推理
        lr_patches = torch.cat(lr_patches, dim=0).to(device)
        
        with torch.no_grad():
            if (not vsr_models[action]) and do_batch:
                assert B % 4 == 0, f"Batch size {B} is not divisible by 4"
                batch_size = B // 4  # (B, C, H*4, W*4)
                sr_patches = []
                for i in range(0, B, batch_size):
                    batch_lr_patches = lr_patches[i:i + batch_size]
                    sr_batch_patches = self.models[action](batch_lr_patches)
                    # print(batch_lr_patches[0, 0, 0, 0:10])
                    # print(sr_batch_patches[0, 0, 0, 0:10])
                    sr_patches.append(sr_batch_patches)
                sr_patches = torch.cat(sr_patches, dim=0)
                # print(sr_patches[0][0][0])
            else:
                if vsr_models[action]:
                    lr_patches = lr_patches.unsqueeze(0)
                sr_patches = self.models[action](lr_patches) 
            # print(sr_patches.shape)
            
        if vsr_models[action]:
            sr_patches = sr_patches.squeeze(0)
            
        assert sr_patches.shape[2] == pad_size * scale_factor and sr_patches.shape[3] == pad_size * scale_factor
        
        pos_low = pad * scale_factor
        pos_high = (pad + SR_size) * scale_factor
        sr_patches = sr_patches[:, :, pos_low:pos_high, pos_low:pos_high]  # (B, C, H*4, W*4)
        print(sr_patches.shape)
            
        if not replacement:
            # 如果不替换，直接返回超分结果
            return sr_patches.cpu()

        # 阶段4: 替换到基础图像
        if replacement:
            with torch.no_grad():
                hr_base = F.interpolate(tensors.to(device), scale_factor=scale_factor,
                                        mode='bicubic', align_corners=False)
        
        for idx, (x1, y1, x2, y2) in enumerate(positions):
            hr_x1 = x1 * scale_factor
            hr_y1 = y1 * scale_factor
            hr_x2 = x2 * scale_factor
            hr_y2 = y2 * scale_factor

            # 尺寸对齐检查与调整
            target_h = hr_y2 - hr_y1
            target_w = hr_x2 - hr_x1
            sr_patch = sr_patches[idx:idx + 1]  # 取出当前的超分结果
            if sr_patch.shape[-2:] != (target_h, target_w):
                sr_patch = F.interpolate(sr_patch,
                                        size=(target_h, target_w),
                                        mode='bicubic',
                                        align_corners=False)

            # 替换到基础图像
            hr_base[idx, :, hr_y1:hr_y2, hr_x1:hr_x2] = sr_patch.squeeze(0)

        return hr_base.cpu()

