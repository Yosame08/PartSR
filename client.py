import os

import numpy as np
import torch
import json
import struct
import time

from flask import Flask, request, make_response
from flask_cors import cross_origin
from torch.multiprocessing import Manager
from typing import Optional

from super_resolution.infer import mp_client_sr
from utils.client_metric import load_gt_roi, decode_video, decode_bytes_to_numpy, metric_frames, save_metric, \
    metric_normal

from utils.mp4_bin_editor import update_reencode_metadata
from utils.utils import ffmpeg_tensor_to_bytes

# 在应用初始化时创建Manager和共享字典
header: Optional[bytes] = None
app = Flask(__name__)

template_file = 'plant'

lr_name = f'/home/guest/server_folder/{template_file}/180p-sell-{template_file}.mp4'
gt_name = f'/home/guest/server_folder/{template_file}/720p-sell-{template_file}.mp4'
gt_roi_folder = f'/home/guest/server_folder/{template_file}/{template_file}_annotate/labels'

save_file = f"client_result/model-test-ours-{template_file}.csv"

gt_full_frames, gt_fps = decode_video(gt_name)
lr_full_frames, lr_fps = decode_video(lr_name)
assert gt_fps == lr_fps and gt_full_frames.shape[0] == lr_full_frames.shape[0]
gt_roi = load_gt_roi(gt_full_frames.shape[3], gt_full_frames.shape[2], gt_roi_folder)
part_len = 120

nxt_id = 1


def metric_clip(idx: int, gt_frames: np.ndarray, lr_frames: np.ndarray,
                roi_frames: np.ndarray, detect_roi: list):
    global nxt_id
    while nxt_id < idx:
        time.sleep(0.5)
    if nxt_id == idx:
        metric_res.extend(metric_frames(gt_frames, lr_frames, roi_frames,
                                        detect_roi, gt_roi[(idx-1)*part_len:idx*part_len], idx))
    nxt_id += 1
    if nxt_id == 7:
        save_metric(metric_res, save_file)

def resolve_request(request):
    metadata = request.form.get('metadata')
    if metadata:
        metadata = json.loads(metadata)
        idx = metadata.get('idx')
    else:
        raise RuntimeError("No metadata provided")

    file = request.files.get('file')
    if not file:
        raise RuntimeError("No file provided")
    return idx, file.read()

@app.route('/header', methods=['POST'])
@cross_origin()
def upload_header():
    try:
        idx, byte = resolve_request(request)
        global header
        header = byte
        return "OK", 200
    except RuntimeError as e:
        return e, 400

@app.route('/sr', methods=['POST'])
@cross_origin()
def upload_stream():
    try:
        idx, byte = resolve_request(request)
    except RuntimeError as e:
        print(f"Error resolving request: {e}")
        return e, 400
    gt_frames = gt_full_frames[(idx - 1) * part_len:idx * part_len]
    lr_frames = lr_full_frames[(idx - 1) * part_len:idx * part_len]

    lr_length = struct.unpack('>I', byte[:4])[0]
    lr_bytes = byte[4:4+lr_length]
    lr_frames, framerate = decode_bytes_to_numpy(lr_bytes)
    frame_count = lr_frames.shape[0]

    off = 4 + lr_length + 1
    SR_size, action = struct.unpack('>II', byte[off:off+8])
    off += 8
    detect_roi = []
    for f in range(frame_count):
        x1, y1 = struct.unpack(">II", byte[off:off+8])
        detect_roi.append((x1, y1, x1 + SR_size, y1 + SR_size))
        off += 8

    identifier = str(idx)
    app.config['sr_queue'].put((identifier, torch.from_numpy(lr_frames).float().div(255), np.array(detect_roi), action))
    while identifier not in result_dict:
        time.sleep(0.02)  # Wait for the SR process to finish
    tensor = result_dict.pop(identifier)

    reencode = ffmpeg_tensor_to_bytes(tensor, framerate, identifier)
    final_video = update_reencode_metadata(lr_bytes, reencode)

    roi_frames, _ = decode_bytes_to_numpy(final_video)
    metric_clip(idx, gt_frames, lr_frames, roi_frames, detect_roi)
    return "OK", 200

@app.route('/metric', methods=['POST'])
@cross_origin()
def metric():
    try:
        idx, byte = resolve_request(request)
    except RuntimeError as e:
        print(f"Error resolving request: {e}")
        return e, 400
    gt_frames = gt_full_frames[(idx - 1) * part_len:idx * part_len]
    lr_frames = lr_full_frames[(idx - 1) * part_len:idx * part_len]

    SR_size, roi_length = struct.unpack('>II', byte[0:8])
    roi_bytes = byte[8:8+roi_length]
    roi_frames, fps = decode_bytes_to_numpy(roi_bytes)
    frame_count = roi_frames.shape[0]
    detect_roi = []
    bytes_offset = 8 + roi_length
    for f in range(frame_count):
        x1, y1 = struct.unpack(">II", byte[bytes_offset:bytes_offset+8])
        detect_roi.append((x1, y1, x1 + SR_size, y1 + SR_size))
        # print(f"{f}, detect_roi: {detect_roi[-1]}")
        bytes_offset = bytes_offset + 8

    import cv2
    cv2.imwrite(f"client_result/lr0.png", lr_frames[0].transpose(1, 2, 0)[:, :, ::-1])
    cv2.imwrite(f"client_result/roi0.png", roi_frames[0].transpose(1, 2, 0)[:, :, ::-1])
    cv2.imwrite(f"client_result/gt0.png", gt_frames[0].transpose(1, 2, 0)[:, :, ::-1])
    assert lr_frames.shape[0] == roi_frames.shape[0]

    metric_clip(idx, gt_frames, lr_frames, roi_frames, detect_roi)
    return "OK", 200

@app.route('/normal', methods=['POST'])
@cross_origin()
def normal_metric():
    try:
        idx, byte = resolve_request(request)
    except RuntimeError as e:
        print(f"Error resolving request: {e}")
        return e, 400
    save_dir = f"./client_result"
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/{idx}.mp4", "wb") as f:
        f.write(byte)
        print(f"Saved part {idx} to {save_dir}/{idx}.mp4")

    frames, fps = decode_bytes_to_numpy(byte)
    if not frames.shape[1:] == gt_full_frames.shape[1:]:
        print(frames.shape, gt_full_frames.shape)
        raise RuntimeError("Frame shape mismatch")

    global nxt_id
    while nxt_id < idx:
        time.sleep(0.4)
    if nxt_id == idx:
        metric_res.extend(metric_normal(gt_full_frames[(idx-1)*part_len:idx*part_len], frames,
                                        gt_roi[(idx-1)*part_len:idx*part_len], idx))
    nxt_id += 1
    if nxt_id == 7:
        save_metric(metric_res, f"client_result/regenhance-{template_file}.csv")
    return "OK", 200

def main():
    from torch import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    torch.multiprocessing.set_sharing_strategy('file_descriptor')
    sr_queue = mp.Queue()
    process_b = mp.Process(target=mp_client_sr, args=(sr_queue, result_dict))
    process_b.start()
    try:
        app.config['sr_queue'] = sr_queue
        app.run(host='0.0.0.0', port=5555)
    except Exception as e:
        print(f"An error occurred: {e}")
        sr_queue.put((None, None))  # 发送终止信号给子进程
        process_b.join()  # 等待子进程结束
        process_b.terminate()  # 强制终止子进程

if __name__ == '__main__':
    manager = Manager()
    result_dict = manager.dict()
    metric_res = manager.list()
    main()

