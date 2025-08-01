import torch
import torchvision.models
from torch import nn

# 方式1-》 保存方式1，加载模型
#model = torch.load("./data/vgg16_method1.pth",  weights_only=False)
#print(model)

# 方式2-》 保存方式2，加载模型
vgg16 = torchvision.models.vgg16(weights=None)
vgg16.load_state_dict(torch.load("./data/vgg16_method2.pth"))
#model = torch.load("./data/vgg16_method2.pth")
print(vgg16)

# 陷阱1
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)

    def forward(self, x):
        x = self.conv1
        return x

model = torch.load("./data/tudui_method.pth", weights_only=False)
print(model)