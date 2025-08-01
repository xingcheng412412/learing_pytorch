import torch
import torchvision.transforms
from PIL import Image
from model import *
image_path = "./images/dog.png"
image = Image.open(image_path)
print(image)
image = image.convert('RGB')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)
image = torch.reshape(image, (1, 3, 32, 32))
model = torch.load("./data/tudui_0.pth", weights_only=False)
print(model)
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))
