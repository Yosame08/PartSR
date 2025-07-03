import os
import random
import torch
import tqdm
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.patches as patches

from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from PIL import Image
from detector import RoIRPN

import sys

sys.path.append("..")
from config import device_name

device = torch.device(device_name)

class AddGaussianNoise(torch.nn.Module):
    """
    自定义高斯噪声变换 (修改版)
    参数：
    - mean: 噪声均值 (默认0)
    - std_range: 噪声标准差范围 (元组, 例: (0.01, 0.03))
    """

    def __init__(self, mean=0.0, std_range=(0.01, 0.03), apply_prob=0.5):
        super().__init__()
        self.mean = mean
        self.std_range = std_range
        self.apply_prob = apply_prob

    def forward(self, img):
        # 如果输入是PIL图像，转换为浮点张量
        if isinstance(img, Image.Image):
            img = F.to_image(img)  # 转换为 uint8 张量 (C, H, W)
            img = F.to_dtype(img, dtype=torch.float32, scale=True)  # 转换为 float32 并归一化到 [0, 1]

        if not self.training or random.random() > self.apply_prob:
            return img

        std = torch.empty(1).uniform_(*self.std_range).item()
        noise = torch.randn_like(img) * std + self.mean
        noisy_img = torch.clamp(img + noise, 0.0, 1.0)
        return noisy_img


# 定义归一化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = v2.Compose([
    # 在PIL图像上进行的增强
    v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1, hue=0.05),
    v2.RandomApply([v2.GaussianBlur(kernel_size=(3, 5))], p=0.3),
    AddGaussianNoise(std_range=(0.001, 0.01), apply_prob=0.5),  # 添加高斯噪声
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

val_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),  # 转换为 float32 并归一化到 [0, 1]
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

except_folder = ['sportsfit', 'toytrain']


def parse_label(label_path, orig_w, orig_h):
    """解析YOLO格式标签并转换坐标"""
    with open(label_path, 'r') as f:
        line = f.readline().strip()
        _, xc, yc, w, h = map(float, line.split())

    xc *= orig_w
    yc *= orig_h
    w *= orig_w
    h *= orig_h

    x1 = int(xc - w / 2)
    y1 = int(yc - h / 2)
    x2 = int(xc + w / 2)
    y2 = int(yc + h / 2)

    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > orig_w:
        x2 = orig_w
    if y2 > orig_h:
        y2 = orig_h

    return torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return tensor * std + mean


def save_enhancement(image: torch.Tensor):
    save_dir = "./saved_samples"
    os.makedirs(save_dir, exist_ok=True)
    random_id = random.randint(1000, 9999)  # 4位随机数
    img_path = os.path.join(save_dir, f"sample_{random_id}.png")

    # 转换张量格式 (C,H,W) -> (H,W,C) 并确保数值范围正确
    image = denormalize(image)
    img_np = image.permute(1, 2, 0).cpu().numpy()
    if img_np.max() <= 1.0:  # 检查是否归一化到[0,1]
        img_np = (img_np * 255).astype("uint8")
    # 创建画布绘制边界框
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_np)
    ax.axis("off")
    # 保存图像并关闭绘图
    plt.savefig(img_path, bbox_inches="tight", pad_inches=0)
    plt.close()


class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, jump=1):
        self.transform = transform
        self.cached_images = []  # 存储预处理后的图像张量
        self.cached_boxes = []  # 存储预处理后的边界框坐标
        self.original_shapes = []  # 存储原始图像尺寸（用于可视化还原）

        # 遍历数据集并缓存所有数据
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if subdir in except_folder or not os.path.isdir(subdir_path):
                continue
            
            img_dir = os.path.join(subdir_path, "images")
            skip = False
            while not os.path.exists(img_dir):
                subsub = os.listdir(subdir_path)
                found = False
                for subsubdir in subsub:
                    if os.path.isdir(os.path.join(subdir_path, subsubdir)) and not subsubdir in except_folder:
                        found = True
                        subdir_path = os.path.join(subdir_path, subsubdir)
                        break
                if not found:
                    print(f"No valid subdirectory found in {subdir_path}, skipping.")
                    skip = True
                    break
                img_dir = os.path.join(subdir_path, "images")
            if skip:
                continue
            label_dir = os.path.join(subdir_path, "labels")

            # 按文件名排序确保图片和标签对齐
            print(f"Processing subdirectory: {img_dir}")
            img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png") or f.endswith(".jpg")])
            for idx in range(0, len(img_files), jump):
                img_file = img_files[idx]
                base_name = os.path.splitext(img_file)[0]
                img_path = os.path.join(img_dir, img_file)
                label_path = os.path.join(label_dir, f"{base_name}.txt")

                if os.path.exists(label_path):
                    # 加载并预处理图像
                    image = Image.open(img_path).convert("RGB")
                    orig_w, orig_h = image.size

                    # 解析并预处理边界框
                    box = parse_label(label_path, orig_w, orig_h)

                    # 缓存处理后的数据
                    self.cached_images.append(image)
                    self.cached_boxes.append(box)
                    self.original_shapes.append(torch.Tensor([orig_h, orig_w]))

    def __len__(self):
        return len(self.cached_images)

    def __getitem__(self, idx):
        """直接从内存获取预处理好的数据"""
        image = self.cached_images[idx]
        boxes = self.cached_boxes[idx]

        # 应用动态增强（如果有）
        if self.transform:
            image = self.transform(image)

        # if random.random() <= 0.002:
        #    save_enhancement(image)

        return image, {
            "boxes": boxes,
            "labels": torch.ones((1,), dtype=torch.int64),
        }


def collate_fn(batch):
    return tuple(zip(*batch))

def visualize_boxes(image, epoch, proposal, image_path):
    x1, y1, x2, y2 = proposal
    if epoch == 4 or (epoch + 1) % 8 == 0:
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        dir_name = f"visualize/epoch_{epoch + 1}"
        os.makedirs(dir_name, exist_ok=True)  # 创建目录
        file_name = os.path.splitext(os.path.basename(image_path))[0]  # [0]去掉扩展名
        plt.savefig(f"{dir_name}/{file_name}.png")
        plt.close(fig)  # 明确关闭这个特定图形

def train_eval(model: RoIRPN, image_path: str, epoch: int):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)

    # 创建预处理流水线
    image_tensor = val_transform(image).to(device)

    with torch.no_grad():
        proposal = model(image_tensor.unsqueeze(0))[0]['boxes']  # 添加batch维度
    # Extract the first proposal (x1, y1, x2, y2)
    if len(proposal) == 0:
        print("No proposals found.")
        return
    proposal = proposal[0].cpu().numpy().astype(int)
    print(f"Recommended Region: {proposal}")
    visualize_boxes(image, epoch, proposal, image_path)

def calculate_iou(box1, box2):
    """
    计算两个矩形框的IoU（交并比）
    参数格式: (x1, y1, x2, y2) 其中 (x1,y1) 是左上角，(x2,y2) 是右下角
    """
    # 计算交集区域坐标
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    # 检查是否有交集
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # 计算交集面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # 计算各自面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集面积
    union_area = box1_area + box2_area - intersection_area
    
    # 计算IoU
    iou = intersection_area / union_area
    return iou

def val_rpn(model: RoIRPN, val_data_loader: DataLoader):
    model.eval()
    missed = 0
    average_iou = 0.0
    average_dist = 0.0
    propose_area = 0.0
    ground_truth_area = 0.0
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_data_loader):
            proposal = model(torch.stack(images).to(device))
            for j in range(len(images)):
                proposal_boxes = proposal[j]['boxes']
                if len(proposal_boxes) > 0:
                    proposal_box = proposal_boxes[0].cpu().numpy().astype(int)
                    ground_truth_box = targets[j]['boxes'][0].numpy().astype(int)
                    iou = calculate_iou(proposal_box, ground_truth_box)
                    average_iou += iou
                    proposal_center = np.array([(proposal_box[0] + proposal_box[2]) / 2, (proposal_box[1] + proposal_box[3]) / 2])
                    target_center = np.array([(ground_truth_box[0] + ground_truth_box[2]) / 2, (ground_truth_box[1] + ground_truth_box[3]) / 2])
                    dist = np.linalg.norm(proposal_center - target_center)
                    average_dist += dist
                    propose_area += (proposal_box[2] - proposal_box[0]) * (proposal_box[3] - proposal_box[1])
                    ground_truth_area += (ground_truth_box[2] - ground_truth_box[0]) * (ground_truth_box[3] - ground_truth_box[1])
                else:
                    missed += 1
    average_iou /= (len(val_dataset) - missed)
    average_dist /= (len(val_dataset) - missed)
    propose_area /= (len(val_dataset) - missed)
    ground_truth_area /= (len(val_dataset) - missed)
    print(f"Validation - Average IoU: {average_iou:.4f}, Average Distance: {average_dist:.4f}, Missed Proposals: {missed}, Propose Area: {propose_area:.2f}, Ground Truth Area: {ground_truth_area:.2f}")

def train_rpn(model: RoIRPN, train_dataset: YOLODataset, val_dataset: YOLODataset, num_epochs: int=256, batch_size: int=128):
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    val_rpn(model, val_data_loader)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1 / batch_size, momentum=0.9, nesterov=True, weight_decay=1e-6)
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.75)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    model.freeze_layers(4)
    for epoch in range(num_epochs):
        total_loss = 0
        total_rpn_loss = 0
        total_roi_loss = 0
        model.train()
        if epoch == 16:
            model.freeze_layers(3)
        elif epoch == 32:
            model.freeze_layers(2)
        elif epoch == 64:
            model.freeze_layers(1)
        optimizer.zero_grad()
        for i, (images, targets) in enumerate(tqdm.tqdm(train_data_loader)):
            images = torch.stack(images).to(device)
            # print("train", images[0])
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images, targets)
            # print(outputs)
            weight_losses, rpn_loss, roi_loss = model.calc_losses(outputs)
            total_loss += weight_losses.item()  # 仅用于输出
            total_rpn_loss += rpn_loss.item()
            total_roi_loss += roi_loss.item()
            weight_losses.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪防止梯度爆炸
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        print(f"Epoch {epoch + 1} Loss: {total_loss:.4f}, RPN Loss: {total_rpn_loss:.4f}, ROI Loss: {total_roi_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.7f}")

        val_rpn(model, val_data_loader)
        train_eval(model, "visualize/frame_000019.jpg", epoch)
        train_eval(model, "visualize/frame_000043.jpg", epoch)
        if (epoch + 1) % 8 == 0:
            # 保存模型
            torch.save(model.state_dict(), f"visualize/epoch_{epoch + 1}/rpn_epoch_{epoch + 1}.pth")
            print(f"Model saved at epoch {epoch + 1}")


if __name__ == "__main__":
    train_dataset = YOLODataset(root_dir="../../vid_source_server/files", transform=train_transform, jump=1)
    # train_dataset = YOLODataset(root_dir="../../roi_datasets", transform=train_transform)
    val_dataset = YOLODataset(root_dir="../../vid_source_server/files", transform=val_transform, jump=10)
    model = RoIRPN().to(device)
    train_rpn(model, train_dataset, val_dataset)
    torch.save(model.state_dict(), "final_rpn_model.pth")
