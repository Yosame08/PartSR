import copy
import random
import struct
from typing import Optional, Tuple, List

import requests
import torch
import time

from multiprocessing import Manager, Pipe
from flask import Flask, request, make_response
from flask_cors import cross_origin
from torchvision import transforms

from classifier.classifier import Classifier
from extractor.extractor import RoIExtractor
from extractor.detector import Detector
from scheduler.scheduler import Scheduler
from super_resolution.infer import dnn_pth_list, mp_server_sr
from utils.mp4_bin_editor import update_reencode_metadata
from utils.utils import decode_mv_residual, ffmpeg_tensor_to_bytes, roi_center_to_xyxy

# 存放视频流的目标服务器地址
from config import config, device_name
device = torch.device(device_name)
source_server = config["source_server"]
app = Flask(__name__)

# 初始化所有可用的SR模型
dnn_number = len(dnn_pth_list)
dnn_latency_param: Optional[List] = None

classifier: Optional[Classifier] = None
scheduler: Optional[Scheduler] = None
detector: Optional[Detector] = None


def initialize_models():
    global classifier, scheduler, detector
    classifier = Classifier("models/classifier.pth", num_classes=2)
    scheduler = Scheduler("")
    detector = Detector("extractor/best_rpn.pth")

def part_sr(identifier: str, video: bytes) -> bytes:
    if identifier not in id_status:
        raise RuntimeError("Found no header")
    print(f'[selective_sr] input length = {len(video)}')
    status = id_status[identifier]
    video_bytes = status["header"] + video

    # 1. 解码为tensor，提取运动向量（没有除以255和没有normalize）
    bg = print("(1)[decode]", end=" ") or time.perf_counter()
    ndarray, framerate, mvs, types, residual_list = decode_mv_residual(video_bytes, identifier)
    tensors_div_255 = torch.from_numpy(ndarray.transpose(0, 3, 1, 2)).float().div(255) # NHWC → NCHW
    tensors_normalized = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensors_div_255)
    decode_time = time.perf_counter() - bg
    print(f"time: {decode_time:.3f}s, shape: {ndarray.shape}")

    # 2. 若该视频没有被分类过，则进行分类
    classify_time = 0
    if "class" not in status:
        bg = print("(2)[classify]", end=" ") or time.perf_counter()
        classified = classifier(tensors_normalized)
        status["class"] = classified
        classify_time = time.perf_counter() - bg
        print(f"time = {classify_time:.3f}s, classified: {classified}")

    # 3. 提取roi
    bg = print("(3)[extract RoI]", end=" ") or time.perf_counter()
    extractor = RoIExtractor(tensors_normalized, ndarray, types, mvs, framerate, residual_list, detector)
    roi_array = extractor.run()
    extract_time = time.perf_counter() - bg
    print(f"time = {extract_time:.3f}s")  # , roi: {roi_array}")

    # 4. 根据数据流的特征，选择合适的SR模型
    bg = print("(4)[select action]", end=" ") or time.perf_counter()
    # context: (1)video size (2)user bandwith/buffer length (3)model queue (4)model latency k
    user = []  # user.extend(status["bandwidth"], status["buffer_length"])
    with queue_lock:
        q = copy.copy(dnn_queue_counts)
    # action = scheduler.infer(ndarray.shape[2:], user, q, dnn_latency_param)
    action = 0
    SR_size = min(ndarray.shape[1], ndarray.shape[2]) // 2
    # print("roi_array", roi_array)
    roi_xyxy = roi_center_to_xyxy(roi_array, SR_size, (ndarray.shape[1], ndarray.shape[2]))
    # print(f"roi_xyxy: {roi_xyxy}")
    time.sleep(random.uniform(0.03, 0.1))  # 模拟调度延迟
    schedule_time = time.perf_counter() - bg
    print(f"time = {schedule_time:.3f}s, action: {action}, SR_size: {SR_size}")

    if action < dnn_number:  # 5. 如果是在服务器超分
        bg = print("(5)[super resolution]", end=" ") or time.perf_counter()
        with queue_lock:
            dnn_queue_counts[action] += 1
        app.config['sr_queue'].put((identifier, tensors_div_255, roi_xyxy, action))
        while identifier not in result_dict:
            time.sleep(0.01)  # Wait for the SR process to finish
        tensor = result_dict.pop(identifier)
        sr_time = time.perf_counter() - bg
        print(f"time = {sr_time:.3f}s, tensor: {tensor.shape}")

        bg = print("(6)[re-encode]", end=" ") or time.perf_counter()
        reencode = ffmpeg_tensor_to_bytes(tensor, framerate, identifier)
        final_video = update_reencode_metadata(video_bytes, reencode)
        reencode_time = time.perf_counter() - bg
        print(f"time = {reencode_time:.3f}s, patch size: {len(final_video)}")

        response_data = struct.pack('>BII', int(0), SR_size, len(final_video)) + final_video
        for i in range(ndarray.shape[0]):
            response_data += struct.pack('>II', roi_xyxy[i][0], roi_xyxy[i][1])
        print(f"request finished. stream size {len(video)}, sr size {len(final_video)}")
        print(f"{decode_time:.3f}s, {classify_time:.3f}s, {extract_time:.3f}s, "
              f"{schedule_time:.3f}s, {sr_time:.3f}s, {reencode_time:.3f}s")
    else:  # 5. 如果是在客户端超分
        response_data = struct.pack('>BII', int(1), SR_size, action - dnn_number)
        for i in range(ndarray.shape[0]):
            response_data += struct.pack('>II', roi_xyxy[i][0], roi_xyxy[i][1])
    return response_data


def check_idx(path: str) -> Optional[Tuple[int, str]]:
    split = path.split('/')
    identifier = split[0]
    filename = split[-1]
    if filename.startswith('bbb'):
        parts = filename.split('_')
        idx = parts[-1][:-4]
        identifier = "_".join(parts[:-1])
        return int(idx), identifier
    elif 'stream0' in filename:
        idx = 0 if 'init' in filename else int(filename.split('-')[-1][:-4])
        # identifier = "sell-stream0"
        return idx, identifier
    return None

output_stat = {}

@app.route('/<path:path>', methods=['GET'])
@cross_origin()
def index(path):
    # bbb 的文件名格式：bbb_30fps_320x180_200k_0.m4v
    # sell 的文件名格式：init-stream0.m4s或者chunk-stream0-00001.m4s
    resp = requests.get(f"{source_server}/{path}", params=request.args)
    file = check_idx(path)
    if file:  # 返回处理后的二进制数据
        idx, identifier = file
        if idx == 0:  # 如果是视频头（第一个分片）
            identifier_dict = manager.dict()
            identifier_dict["header"] = resp.content
            id_status[identifier] = identifier_dict
            processed_data = resp.content
        else:
            delay_bg = time.perf_counter()
            processed_data = struct.pack(">I", len(resp.content)) + resp.content + part_sr(identifier,
                                                                                           resp.content)
            delay_time = time.perf_counter() - delay_bg
            print(f"[idx={idx}] delay time = {delay_time:.3f}s, send size = {len(processed_data)}\n##########")
            global output_stat
            if identifier not in output_stat:
                output_stat[identifier] = [1, [(delay_time, len(processed_data))]]
            else:
                output_stat[identifier][1].append((delay_time, len(processed_data)))
                output_stat[identifier][0] += 1
            if output_stat[identifier][0] == 6:
                print(identifier)
                for item in output_stat[identifier][1]:
                    print(f"{item[0]:.3f},{item[1]}")
        flask_response = make_response(processed_data)
        flask_response.headers['Content-Type'] = 'application/octet-stream'
        flask_response.headers['Content-Length'] = len(processed_data)
    else:  # 原封不动返回收到的 response
        flask_response = make_response(resp.content)
        flask_response.headers['Content-Type'] = resp.headers.get('Content-Type', 'application/octet-stream')
        flask_response.headers['Content-Length'] = len(resp.content)
    return flask_response


def main():
    from torch import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    torch.multiprocessing.set_sharing_strategy('file_descriptor')
    # 创建 Pipe，用于主进程和子进程通信
    parent_conn, child_conn = Pipe()
    sr_queue = mp.Queue()
    process_b = mp.Process(target=mp_server_sr, args=(sr_queue, child_conn, result_dict, queue_lock, dnn_queue_counts))
    process_b.start()
    try:
        initialize_models()
        global dnn_latency_param
        dnn_latency_param = parent_conn.recv()  # 接收子进程发送的延迟参数
        parent_conn.close()
        app.config['sr_queue'] = sr_queue
        app.run(host='0.0.0.0', port=11451)
    except Exception as e:
        print(f"An error occurred: {e}")
        sr_queue.put((None, None, None, None))  # 发送终止信号给子进程
        process_b.join()  # 等待子进程结束
        process_b.terminate()  # 强制终止子进程


if __name__ == '__main__':
    # from SR_DNN.infer import test_main
    # test_main()  # 测试SR模型的延迟
    # exit(0)
    # 在应用初始化时创建Manager和共享字典
    manager = Manager()
    result_dict = manager.dict()
    id_status = manager.dict()
    queue_lock = manager.Lock()
    dnn_queue_counts = manager.list([0] * dnn_number)
    try:
        main()
    except KeyboardInterrupt as e:
        print("KeyboardInterrupt")
        exit(1)

