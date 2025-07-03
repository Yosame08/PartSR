from .detector import Detector
from .tracker import track, init_tracker
from typing import List, Tuple
import numpy as np
import torch
import math

dynamicity_threshold = 1
alpha = 1
beta = 1


def get_inside(block_interval: Tuple[int, int], roi_interval: Tuple[int, int]) -> int:
    left_or_top = max(block_interval[0], roi_interval[0])
    right_or_bottom = min(block_interval[1], roi_interval[1])
    return 0 if left_or_top >= right_or_bottom else right_or_bottom - left_or_top


def accumulate(mv, x: int, y: int, w: int, h: int) -> float:
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    tot = 0
    for item in mv:
        source, w, h, src_x, src_y, dst_x, dst_y, motion_x, motion_y, motion_scale = item
        x_inside = get_inside((dst_x, dst_x + w), (x1, x2))
        y_inside = get_inside((dst_y, dst_y + h), (y1, y2))
        area = x_inside * y_inside
        if area > 0:
            module = math.sqrt((motion_x / motion_scale) ** 2 + (motion_y / motion_scale) ** 2)
            tot += module * area
    return tot


class RoIExtractor:
    def __init__(self, tensors: torch.Tensor, ndarray: np.ndarray,
                 types: List[str], mvs, framerate: float,
                 residual_list: np.ndarray, detector: Detector):
        self.tensors = tensors
        self.ndarray = ndarray
        self.types = types
        self.mvs = mvs
        self.framerate = framerate
        self.residual_list = residual_list
        self.detector = detector

        self.dynamicity = 0
        self.tracker = None
        self.diag_length = math.sqrt(tensors.shape[2] ** 2 + tensors.shape[3] ** 2)
        self.video_size = tensors.shape[2] * tensors.shape[3]

        self.anchor_num = 0

    def _new_anchor(self, idx: int) -> Tuple[int, int]:
        self.anchor_num += 1
        import time
        bg = time.time()
        x1, y1, x2, y2 = self.detector(self.tensors[idx])
        # print(f"{idx}, detect time: {time.time() - bg:.3f}s, result: ({x1}, {y1}, {x2}, {y2})")
        w, h = x2 - x1, y2 - y1
        self.tracker = init_tracker(self.ndarray[idx], (x1, y1, w, h))
        self.dynamicity = 0
        return x1 + w // 2, y1 + h // 2

    def _temporal_reuse(self, idx: int) -> Tuple[bool, int, int]:  # bool -> whether reusing succeeded
        success, x, y, w, h = track(self.tracker, self.ndarray[idx])
        # print(f"{idx}, track result: {success}, ({x}, {y}, {w}, {h})")
        if not success:
            return False, 0, 0
        motion = (accumulate(self.mvs[idx], x, y, w, h)
                  * self.framerate / (self.video_size - w * h)) / self.diag_length
        self.dynamicity += alpha * motion + beta * self.residual_list[idx]
        # print(f"dynamicity for {idx}: {self.dynamicity:.3f}, motion: {motion:.3f}, residual: {self.residual_list[idx]:.3f}")
        return (True, x + w // 2, y + h // 2) if self.dynamicity < dynamicity_threshold else (False, 0, 0)

    def run(self) -> List[Tuple[int, int]]:
        roi_list = []
        for idx in range(self.tensors.shape[0]):
            if self.tracker is None or self.types[idx] == 'I':
                x, y = self._new_anchor(idx)
            else:
                success, x, y = self._temporal_reuse(idx)
                if not success:
                    x, y = self._new_anchor(idx)
            roi_list.append((int(x), int(y)))
        print(f"anchor num: {self.anchor_num}")
        return roi_list

