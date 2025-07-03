from typing import Tuple, Optional
import cv2
import numpy as np

def init_tracker(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> cv2.Tracker:
    tracker = cv2.TrackerKCF_create()
    tracker.init(frame, roi)
    return tracker

def track(tracker: cv2.Tracker, frame: np.ndarray) -> Tuple[bool, int, int, int, int]:
    success, box = tracker.update(frame)
    if success:
        x, y, w, h = [int(v) for v in box]
        return success, x, y, w, h
    else:
        return False, 0, 0, 0, 0

if __name__ == "__main__":
    print(type(cv2.TrackerKCF_create()))
