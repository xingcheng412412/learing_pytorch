from  torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
# tensor的数据类型
# 通过 transform.ToTensor去解决两个问题
# 2. 为什么我们需要Tensor数据类型

# 绝对路径 C:\Users\向依玲\Desktop\pytorch_project\dataset\train\ants_image\5650366_e22b7e1065.jpg
# 相对路径 dataset/train/ants_image/6240329_72c01e663e.jpg
img_path = "dataset/train/ants_image/6240329_72c01e663e.jpg"
img = Image.open(img_path)
print(img)

writer = SummaryWriter("logs")

# Totensor使用
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
writer.add_image("Tensor_img", tensor_img)

# Normalize使用
print(tensor_img[0][0][0])
trans_norm = transforms.Normalize([1, 3, 5], [3, 2, 1])
img_norm = trans_norm(tensor_img)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm, 1)

# Resize使用
print(img.size)
trans_resize = transforms.Resize((512, 512))
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL -> totensor -> img_resize tensor
img_resize = tensor_trans(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize)

# Compose -resize - 2
trans_resize_2 = transforms.Resize(512)
# PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2, tensor_trans])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

#RandomCrop
trans_random = transforms.RandomCrop(256)
trans_compose_2 = transforms.Compose([trans_random, tensor_trans])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()