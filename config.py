import os
import yaml
import torch
from typing import Dict

def parse_device_name(conf: Dict) -> str:
    gpu_id = conf["gpu_id"]
    device_name = f"cuda:{gpu_id}" if (torch.cuda.is_available() and 0 <= gpu_id < torch.cuda.device_count()) else "cpu"
    print(f"Specified device: {device_name}")
    return device_name

config_path = "config.yaml" if os.path.exists("config.yaml") else "../config.yaml"
with open(config_path, 'r') as cf:
    config = yaml.load(cf, Loader=yaml.FullLoader)
device_name = parse_device_name(config)
vsr_models = config['vsr_models']
