import torch
import torchvision
from torch import nn
from torch.ao.nn.qat import Conv2d
from torch.utils.data import DataLoader
import torch.ao.quantization as quant
from torch.utils.tensorboard import SummaryWriter

# 数据集加载
dataset = torchvision.datasets.CIFAR10(
    "./dataset",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True  # 确保下载数据集
)
dataloader = DataLoader(dataset, batch_size=64)

# 模型定义
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.qconfig = quant.get_default_qat_qconfig('fbgemm')
        self.conv1 = Conv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=3,
            stride=1,
            padding=0,
            qconfig=self.qconfig
        )

    def forward(self, x):
        return self.conv1(x)

# 实例化并准备QAT
tudui = Tudui()
tudui = quant.prepare_qat(tudui.train())  # 只需调用一次

writer = SummaryWriter("../logs")

step = 0
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    print(imgs.shape)
    print(output.shape)
    writer.add_images("input", imgs, step)
    torch.reshape(output,(-1, 3, 30, 30))
    writer.add_images("output", imgs, step)


    step = step + 1

writer.close()