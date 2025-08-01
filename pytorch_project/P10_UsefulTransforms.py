from  torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

img = Image.open("dataset/train/ants_image/5650366_e22b7e1065.jpg")
print(img)