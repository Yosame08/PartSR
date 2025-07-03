from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from tqdm import tqdm
import torchvision.transforms.v2.functional as F
from torchvision.transforms import InterpolationMode, Normalize
from sklearn.model_selection import train_test_split
import gc
import os
import glob
import time
import random
import torch
import torch.nn as nn
import decord

from .classifier import VideoClassifier
from config import config

from config import device_name
device = torch.device(device_name)

def parse_video_groups(data_dir):
    """
    解析数据集目录，按 "类别-视频名-画质" 分组视频片段
    返回:
        video_groups: { "class-video-quality": [所有片段路径] }
        class_to_idx: { "class": 类别索引 }
    """
    video_groups = defaultdict(list)
    class_names = set()

    for vid_type in os.listdir(data_dir):
        video_paths = glob.glob(os.path.join(os.path.join(data_dir, vid_type), "*.mp4"))
        # 解析文件名并分组
        for path in video_paths:
            filename = os.path.basename(path)
            parts = filename.split("-")
            if len(parts) < 4:
                continue  # 跳过不合法文件名
            class_name, video_name, quality, _ = parts[:4]
            assert class_name == vid_type
            group_key = f"{class_name}-{video_name}-{quality}"
            video_groups[group_key].append(path)
            class_names.add(class_name)

        # 生成类别索引映射
        class_to_idx = {cls: idx for idx, cls in enumerate(sorted(class_names))}
    return video_groups, class_to_idx

def split_video_groups(video_groups, class_to_idx, test_ratio=0.2, seed=42):
    """
    按视频组划分数据集，保证同一视频的不同片段在同一集合
    返回:
        (train_paths, train_labels), (test_paths, test_labels)
    """
    # 按类别分层抽样
    groups_per_class = defaultdict(list)
    for group_key in video_groups:
        class_name = group_key.split("-")[0]
        groups_per_class[class_name].append(group_key)

    train_groups, test_groups = [], []
    for cls, groups in groups_per_class.items():
        # 按固定比例划分每个类别的视频组
        cls_train, cls_test = train_test_split(
            groups, test_size=test_ratio, random_state=seed
        )
        train_groups.extend(cls_train)
        test_groups.extend(cls_test)

    # 收集所有片段路径和标签
    def _collect_data(groups):
        paths, labels = [], []
        for group_key in groups:
            class_name = group_key.split("-")[0]
            label = class_to_idx[class_name]
            paths.extend(video_groups[group_key])
            labels.extend([label] * len(video_groups[group_key]))
        return paths, labels

    return _collect_data(train_groups), _collect_data(test_groups)


class GroupedVideoDataset(Dataset):
    def __init__(self, video_paths, labels, target_size=(180, 320), classify_sample_rate=10, test=False):
        self.video_data = []
        self.labels = labels
        self.target_h, self.target_w = target_size
        self.classify_sample_rate = classify_sample_rate
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.test = test

        # 预加载视频（不进行归一化）
        for path in tqdm(video_paths, desc="Loading videos"):
            frames = self._extract_frames(path)
            self.video_data.append(frames)

    def _extract_frames(self, path):
        """返回 0-1 范围的未归一化数据"""
        vr = decord.VideoReader(path, ctx=decord.cpu(0))
        total_frames = len(vr)
        start = random.randint(0, self.classify_sample_rate - 1)
        indices = list(range(start, total_frames, self.classify_sample_rate))
        frames = vr.get_batch(indices).asnumpy()  # (N, H, W, C)
        frames = torch.from_numpy(frames.transpose(0, 3, 1, 2))  # (N, C, H, W)
        return frames.float() / 255.0  # 仅缩放至 [0,1]

    def _augment_frames(self, frames):
        """应用时空一致性数据增强"""
        # 随机水平翻转
        if torch.rand(1) < 0.5:
            frames = F.horizontal_flip(frames)

        # 随机亮度/对比度调整
        brightness = random.uniform(0.85, 1.15)
        contrast = random.uniform(0.85, 1.15)
        frames = F.adjust_brightness(frames, brightness)
        frames = F.adjust_contrast(frames, contrast)

        # 动态缩放和裁剪
        H, W = frames.shape[-2:]
        scale = max(self.target_h / H, self.target_w / W)
        new_H, new_W = int(H * scale), int(W * scale)
        frames = F.resize(frames, [new_H, new_W], InterpolationMode.BICUBIC)

        # 随机裁剪
        if self.test:
            # 测试时中心裁剪
            top = (new_H - self.target_h) // 2
            left = (new_W - self.target_w) // 2
        else:
            top = torch.randint(0, new_H - self.target_h + 1, (1,)).item()
            left = torch.randint(0, new_W - self.target_w + 1, (1,)).item()

        return F.crop(frames, top, left, self.target_h, self.target_w)

    def __len__(self):
        return len(self.video_data)

    def __getitem__(self, idx):
        frames = self.video_data[idx]
        label = self.labels[idx]
        frames = self._augment_frames(frames)
        return self.normalize(frames), label  # 最后归一化 ✅

def train():
    torch.manual_seed(42)
    # 步骤1: 解析数据集
    video_groups, class_to_idx = parse_video_groups("../../datasets/MyData_Classify")
    print(f"Classes: {len(class_to_idx)}")
    # 步骤2: 划分数据集
    (train_paths, train_labels), (test_paths, test_labels) = split_video_groups(video_groups, class_to_idx)
    print(f"Train: {len(train_paths)} samples, Test: {len(test_paths)} samples")

    print("Reading train dataset")
    train_dataset = GroupedVideoDataset(train_paths, train_labels, (config["video_width"], config["video_height"]), config["classify_sample_rate"], test=False)
    print("Reading test dataset")
    test_dataset = GroupedVideoDataset(test_paths, test_labels, (config["video_width"], config["video_height"]), config["classify_sample_rate"], test=True)  # 测试时可用不同transform

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = VideoClassifier(num_classes=len(class_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    best_loss = float('inf')  # 初始化最佳loss为正无穷

    for epoch in range(30):
        model.train()
        tot_loss = 0
        # idx = 1
        for frames, labels in tqdm(train_loader):
            # for frames, labels in train_loader:
            frames = frames.to(device)
            labels = labels.to(device)
            outputs = model(frames)
            loss = criterion(outputs, labels)
            tot_loss += loss.item()
            # loss = loss / 3
            loss.backward()
            # if idx % 3 == 0:
            optimizer.step()
            optimizer.zero_grad()
            # idx += 1
        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        tot_time = 0
        with torch.no_grad():
            for frames, labels in tqdm(test_loader):
                frames = frames.to(device)
                labels = labels.to(device)
                bg = time.time()
                outputs = model(frames)
                tot_time += time.time() - bg
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        del frames, labels, outputs
        torch.cuda.empty_cache()
        gc.collect()

        with open("log.txt", "a") as f:
            f.write(f"{epoch} Test Accuracy: {100 * correct / total:.2f}%\n")
        avg_loss = tot_loss / len(train_loader)
        print(
            f"epoch {epoch}, loss: {avg_loss:.4f}, Test Acc: {100 * correct / total:.2f}%, Avg Infer Time: {tot_time / len(test_loader):.3f}s")
        if avg_loss < best_loss and avg_loss < 0.4:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"best{avg_loss:.3f}.pth")
            best_loss = avg_loss
            print(f"New best model saved at epoch {epoch} with loss {avg_loss:.3f}")


if __name__ == "__main__":
    train()
