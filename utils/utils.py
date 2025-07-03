import os
import subprocess
import torch
import numpy as np
import cv2

from typing import Tuple, List
from mvextractor.videocap import VideoCap

from config import config

ffmpeg = config["ffmpeg"]
ffmpeg_residual = config["ffmpeg_residual"]

def del_if_exist(filename: str):
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass

def roi_center_to_xyxy(roi_center: List[Tuple[int, int]], size: int, frame_shape: Tuple[int, int]) -> np.ndarray:
    """
    :param roi_center: [(x, y), (x, y), ...]
    :param size: size for SR
    :param frame_shape: (H, W)
    """
    assert frame_shape[0] >= size and frame_shape[1] >= size, "Frame size must be larger than ROI size"
    xyxy = []
    half_size = size // 2
    for idx in range(len(roi_center)):
        x_center, y_center = roi_center[idx]
        def check_bound(center: int, lim: int) -> Tuple[int, int]:
            val1 = center - half_size if center >= half_size else 0
            if val1 + size > lim:
                dif = val1 + size - lim
                val1 -= dif
                val2 = lim
            else:
                val2 = val1 + size
            return val1, val2
        x1, x2 = check_bound(x_center, frame_shape[1])
        y1, y2 = check_bound(y_center, frame_shape[0])
        xyxy.append([x1, y1, x2, y2])
    return np.array(xyxy)

def decode_residual(filename: str, frame_num: int) -> np.ndarray:
    cmd = [
        ffmpeg_residual, "-loglevel", "error",
        "-i", filename, "-f", "null", "-"
    ]
    process = subprocess.Popen(
        cmd,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )

    mb_data = b''
    while True:
        new_data = process.stderr.read(4096)
        if not new_data:
            break
        mb_data += new_data
    assert mb_data != b'', "residual decoding returns nothing"

    ptr = 0
    residual_list = []
    while ptr < len(mb_data):
        frame_idx = int.from_bytes(mb_data[ptr:ptr+4], 'little')
        mb_width = int.from_bytes(mb_data[ptr+4:ptr+8], 'little')
        mb_height = int.from_bytes(mb_data[ptr+8:ptr+12], 'little')
        mb_frame_size = 12 + mb_width * mb_height
        if 0 <= frame_idx < frame_num:
            area_ops = np.frombuffer(mb_data[ptr+12:ptr+mb_frame_size], dtype=np.uint8).reshape((mb_width, mb_height))
            residual_list.append(area_ops.sum())
        ptr += mb_frame_size
    return np.array(residual_list)

def decode_mv_residual(video_bytes: bytes, identifier: str, limit: int=0)\
        -> Tuple[np.ndarray, float, List[List], List[str], np.ndarray]:
    tmp_name = f"/dev/shm/{identifier}.mp4"
    with open(tmp_name, "wb") as f:
        f.write(video_bytes)

    cv2_video = cv2.VideoCapture(tmp_name)
    framerate = cv2_video.get(cv2.CAP_PROP_FPS)
    cv2_video.release()

    cap = VideoCap() # BGR
    ret = cap.open(tmp_name)
    if not ret:
        raise RuntimeError(f"Could not open video", len(video_bytes))

    step = 0
    all_frames = []
    all_motion_vectors = []
    all_types = []

    # continuously read and display video frames and motion vectors
    while True:
        ret, frame, motion_vectors, frame_type, timestamp = cap.read()
        if not ret:
            print(f"decoded {step} frames")
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换操作
        all_frames.append(frame)
        all_motion_vectors.append(motion_vectors)
        all_types.append(frame_type)

        step += 1
        if step == limit:
            print(f"decoded {step} frames (limit)")
            break

    cap.release()
    all_frames = np.array(all_frames)
    assert all_frames.shape[1] % 4 == 0 and all_frames.shape[2] % 4 == 0
    video_area = all_frames.shape[1] * all_frames.shape[2]
    residual_list = decode_residual(tmp_name, all_frames.shape[0]) / video_area # pixel average
    del_if_exist(tmp_name)
    return all_frames, framerate, all_motion_vectors, all_types, residual_list

def ffmpeg_tensor_to_bytes(frames_tensor: torch.Tensor, framerate: float, identifier: str) -> bytes:
    N, C, H, W = frames_tensor.shape
    ff_out_name = f'/dev/shm/{identifier}'
    command = [
        ffmpeg, '-y', '-loglevel', 'error',  # 使用传入的ffmpeg路径
        '-f', 'rawvideo',  # 输入为原始视频数据
        '-pix_fmt', 'rgb24',  # 输入像素格式
        '-s', f'{W}x{H}',  # 帧尺寸
        '-r', str(framerate),  # 输入帧率
        '-i', '-',  # 从标准输入读取数据

        '-c:v', 'libx264',
        '-crf', '18',
        '-preset', 'slow',

        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-f', 'dash',
        '-single_file', '1',
        ff_out_name
    ]

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 将所有帧数据转换为字节流
    frames_np = frames_tensor.mul(255).byte().cpu().numpy()  # [N,C,H,W] uint8
    frames_np = np.transpose(frames_np, (0, 2, 3, 1))  # NCHW -> NHWC
    input_bytes = frames_np.tobytes()                        # 直接转为字节流
    _, stderr_data = process.communicate(input=bytes(input_bytes))
    if process.returncode != 0:
        print(stderr_data.decode('utf-8'))

    ff_out_vid = f"/dev/shm/{identifier}-stream0.mp4"
    with open(ff_out_vid, "rb") as f:
        video = f.read()
    del_if_exist(ff_out_name)
    del_if_exist(ff_out_vid)
    return video

def show_frame(tensor_frame):
    # 将CHW Tensor转为HWC numpy数组
    if type(tensor_frame) == np.ndarray:
        tensor_frame = torch.from_numpy(tensor_frame)
    if tensor_frame.shape[0] == 3:
        tensor_frame = tensor_frame.permute(1, 2, 0)
    frame_np = tensor_frame.numpy().astype(int)  # BCHW -> HWC
    plt.imshow(frame_np)
    plt.axis('off')
    plt.savefig('frame.png', bbox_inches='tight', pad_inches=0)
    plt.show()

def test_mv(stream, stat_lim):
    def decode_mv_stream(a, b, c):
        raise NotImplementedError("decode mv stream is modified")
    bg = time.time()
    tensor_mv, mvs, _, bitrate, framerate = decode_mv_stream(stream, "Temp", stat_lim)
    print(_)
    print(f"{time.time() - bg}")
    print(bitrate, framerate)

    width = tensor_mv.shape[3]
    height = tensor_mv.shape[2]
    print(width, height)

    x = [i / framerate for i in range(stat_lim)]
    y = []
    accu = []
    for i in range(len(mvs)):
        m = 0
        for item in mvs[i]:
            source, w, h, src_x, src_y, dst_x, dst_y, motion_x, motion_y, motion_scale = item
            module = math.sqrt((motion_x / motion_scale) ** 2 + (motion_y / motion_scale) ** 2)
            m += module * (w * h)
        m = m * framerate / ((width * height) ** 1.5)
        y.append(m)
        if m == 0:
            accu.append(0)
        else:
            accu.append(accu[-1] + m)
    plt.figure(figsize=(8, 8), dpi=200)
    plt.subplot(211)
    plt.plot(x, y, linewidth=0.5)
    plt.xlabel("sec")
    plt.ylabel("mv module")
    plt.title('mv module delta')
    plt.xlim(0, stat_lim // framerate)
    plt.ylim(0, 0.3)
    plt.xticks(np.arange(0, stat_lim // framerate + 1, 2))
    plt.grid()
    plt.subplot(212)
    plt.plot(x, accu, linewidth=0.5)
    plt.xlabel("sec")
    plt.ylabel("mv accumulate")
    plt.title('accumulate trend')
    plt.xlim(0, stat_lim // framerate)
    plt.ylim(0, 2)
    plt.xticks(np.arange(0, stat_lim // framerate + 1, 2))
    plt.grid()
    plt.show()

def test_residual(stream, stat_lim):
    tensor_mv, framerate, mvs, types, residual_list = decode_mv_residual(stream, "Temp", stat_lim)
    print(residual_list)
    x = np.arange(len(residual_list))
    plt.figure(figsize=(6, 6), dpi=200)
    plt.plot(x, residual_list)
    plt.grid()
    plt.show()

if __name__ == "__main__":
    import time, math
    import matplotlib.pyplot as plt
    # header = "/home/guest/bbb_30fps/bbb_30fps_320x180_200k/bbb_30fps_320x180_200k_0.m4v"
    video_filename = "/home/guest/server_folder/slipper/180p-sell-slipper.mp4"
    stat_lim = 600
    # 把filename读取成bytes
    with open(video_filename, 'rb') as vf:
        stream = vf.read()
    test_residual(stream, stat_lim)