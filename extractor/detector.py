from typing import List, Optional, Dict

import torch
from torch import nn

import torchvision
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.models.detection.image_list import ImageList

from torchvision.ops import box_iou, generalized_box_iou
from torchvision.models.detection import FasterRCNN

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.append("..")
from config import device_name
device = torch.device(device_name)

class RoIRPN(FasterRCNN):
    def __init__(self, min_size=180, max_size=320):
        # 创建带有FPN的ResNet50骨干网络
        backbone = resnet_fpn_backbone(
            backbone_name='resnet34',
            weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1,
            trainable_layers=3
        )

        # 确保骨干网络有out_channels属性
        if not hasattr(backbone, 'out_channels'):
            backbone.out_channels = 256  # FPN输出的通道数

        # 锚点生成器配置（针对小尺寸图像优化）
        anchor_sizes = ((16, 32, 48), (32, 64, 96), (48, 72, 96), (64, 90, 128), (72, 96, 128))  # 每个特征图级别一个尺寸
        aspect_ratios = ((0.5, 0.66, 1.0, 1.5, 2.0),) * len(anchor_sizes)  # 每个特征图级别相同的宽高比
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )

        # ROI对齐池化
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )

        # 初始化父类
        super().__init__(
            backbone,
            num_classes=2,  # 背景+前景
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=min_size,
            max_size=max_size
        )

        self.rpn.nms_thresh = 0.7

        self.rpn._pre_nms_top_n = {
            "training": 2000, 
            "testing": 1000
        }

        self.rpn._post_nms_top_n = {
            "training": 2000, 
            "testing": 1000
        }

    def freeze_layers(self, trainable_layers=3):
        """冻结指定层数以下的骨干网络层"""
        for name, param in self.backbone.body.named_parameters():
            if 'layer' in name:
                # 提取层号 (e.g., 'layer1', 'layer2', etc.)
                layer_num = int(name.split('.')[0][-1])
                param.requires_grad_(layer_num >= trainable_layers)
                if layer_num >= trainable_layers:
                    print(f"Training layer: {name}")
            else:
                # 非层参数（如stem）默认冻结
                param.requires_grad_(False)

        # RPN和检测头始终训练
        for param in self.rpn.parameters():
            param.requires_grad_(True)
        for param in self.roi_heads.parameters():
            param.requires_grad_(True)

    def forward(self, images, targets=None):
        # 确保输入是列表形式
        if isinstance(images, torch.Tensor):
            images = [images] if images.dim() == 3 else list(images)

        # 训练模式
        if self.training and targets is not None:
            return super().forward(images, targets)
        # 推理模式
        else:
            outputs = super().forward(images)
            processed_outputs = []

            for output in outputs:
                if len(output['boxes']) > 0:
                    best_idx = torch.argmax(output['scores'])
                    processed_output = {
                        'boxes': output['boxes'][best_idx].unsqueeze(0),
                        'scores': output['scores'][best_idx].unsqueeze(0),
                        'labels': output['labels'][best_idx].unsqueeze(0)
                    }
                else:
                    h, w = images[0].shape[-2:]
                    processed_output = {
                        'boxes': torch.tensor([[0, 0, w, h]], dtype=torch.float32, device=images[0].device),
                        'scores': torch.tensor([0.0], device=images[0].device),
                        'labels': torch.tensor([0], dtype=torch.int64, device=images[0].device)
                    }
                processed_outputs.append(processed_output)

            return processed_outputs

    def calc_losses(self, loss_dict):
        """计算总损失"""
        rpn_loss = loss_dict.get('loss_objectness', 0) + loss_dict.get('loss_rpn_box_reg', 0)
        roi_loss = loss_dict.get('loss_classifier', 0) + loss_dict.get('loss_box_reg', 0)

        return rpn_loss + roi_loss, rpn_loss, roi_loss

class Detector:
    def __init__(self, model_path: str="extractor/best_rpn.pth"):
        self.model = RoIRPN()

        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict)
            logger.info("Detector model parameters initialized successfully")
        except FileNotFoundError:
            logger.warning(f"Detector model file {model_path} not found, using default weights")

        self.model = self.model.to(device)

        # 冻结梯度计算
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.model.eval()

        # 验证前向传播
        test_input = torch.randn(1, 3, 224, 224).to(device)  # 假设1帧输入
        with torch.no_grad():
            _ = self.model(test_input)
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        执行目标检测
        参数：
        - image: 形状为 [channels, height, width] 的float32张量
                 数值范围建议为[0, 1]（与训练时预处理一致）
        返回：
        - 检测结果，包含边界框、置信度和标签
        """
        with torch.no_grad():
            proposal = self.model(image.to(device))
            proposal = proposal[0]['boxes'][0].cpu().numpy().astype(int)
        return proposal

