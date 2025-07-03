from torchvision.models import resnet34, resnet50, resnet101
from torchvision.models.resnet import ResNet34_Weights, ResNet50_Weights, ResNet101_Weights
import torch.nn as nn
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import device_name
device = torch.device(device_name)

class VideoClassifier(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=2):
        super().__init__()

        self.resnet = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-1])
        for param in self.resnet.parameters():  # 冻结前6层
            param.requires_grad = False

        self.gru = nn.GRU(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0  # 多层时加Dropout
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, num_classes)
        )

    def unfreeze(self, begin, end):
        for param in self.resnet[begin:end].parameters():
            param.requires_grad = True

    def forward(self, x):
        num_frames = x.shape[0]
        features = self.resnet(x)  # resnet -> features = (frames, 2048, 1, 1)
        features = features.view(num_frames, -1)  # features = (frames, 2048)
        _, hidden = self.gru(features)  # hidden = (4, hidden_size)

        # 拼接双向 GRU 的最后一层隐藏状态
        hidden = torch.cat((hidden[-2], hidden[-1])) if self.gru.bidirectional else hidden[-1]
        return self.fc(hidden)

class Classifier:
    def __init__(self, model_path: str, num_classes: int):
        self.model = VideoClassifier(num_classes=num_classes)

        # 加载权重
        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict)
            logger.info("classifier model parameters initialized successfully")
        except FileNotFoundError:
            logger.warning(f"classifier model file {model_path} not found, using default weights")

        self.model = self.model.to(device)

        # 冻结梯度计算
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.model.eval()

        # 验证前向传播
        test_input = torch.randn(16, 3, 224, 224).to(device)  # 假设16帧输入
        with torch.no_grad():
            _ = self.model(test_input)

    def __call__(self, input_tensor: torch.Tensor) -> int:
        if input_tensor.dim() != 4 or input_tensor.shape[1] != 3:
            raise ValueError(f"输入维度应为4 (NCHW), got {input_tensor.dim()}")

        with torch.no_grad():
            input_tensor = input_tensor.to(device)
            outputs = self.model(input_tensor)
            _, predicted = torch.max(outputs, 0)
            return predicted.item()

