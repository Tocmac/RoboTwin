import torch
import torchvision
from torchsummary import summary

# 加载模型
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# 查看参数量
summary(model, input_size=(3, 224, 224))