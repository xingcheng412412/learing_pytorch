import torch
import torchvision
from torch import nn
from torchvision.models import VGG16_Weights

vgg16 = torchvision.models.vgg16(weights=None)
# 保存方式1
torch.save(vgg16, "./data/vgg16_method1.pth")

# 保存方式2(官方推荐)
torch.save(vgg16.state_dict(), "./data/vgg16_method2.pth")

# 陷阱
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)

    def forward(self, x):
        x = self.conv1
        return x

tudui = Tudui()
torch.save(tudui, "./data/tudui_method.pth")