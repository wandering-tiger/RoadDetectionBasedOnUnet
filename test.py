import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import numpy as np
import os

# 定义测试数据集类
class TestDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_filenames = os.listdir(image_folder)
        self.image_filenames = sorted(self.image_filenames)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_filenames[idx])

        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image,img_name
    

# 定义UNet模型
class double_conv2d_bn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,strides=1,padding=1):
        super(double_conv2d_bn,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,
                               kernel_size=kernel_size,
                              stride = strides,padding=padding,bias=True)
        self.conv2 = nn.Conv2d(out_channels,out_channels,
                              kernel_size = kernel_size,
                              stride = strides,padding=padding,bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class deconv2d_bn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=2,strides=2):
        super(deconv2d_bn,self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels,out_channels,
                                        kernel_size = kernel_size,
                                       stride = strides,bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out

class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
#         self.layer1_conv = double_conv2d_bn(3,64)
#         self.layer2_conv = double_conv2d_bn(64,128)
#         self.layer3_conv = double_conv2d_bn(128,256)
#         self.layer4_conv = double_conv2d_bn(256,512)
#         self.layer5_conv = double_conv2d_bn(512,1024)
#         self.layer6_conv = double_conv2d_bn(1024,512)
#         self.layer7_conv = double_conv2d_bn(512,256)
#         self.layer8_conv = double_conv2d_bn(256,128)
#         self.layer9_conv = double_conv2d_bn(128,64)
#         self.layer10_conv = nn.Conv2d(64,3,kernel_size=3,
#                                      stride=1,padding=1,bias=True)

#         self.deconv1 = deconv2d_bn(1024,512)
#         self.deconv2 = deconv2d_bn(512,256)
#         self.deconv3 = deconv2d_bn(256,128)
#         self.deconv4 = deconv2d_bn(128,64)
        self.layer1_conv = double_conv2d_bn(3,8)
        self.layer2_conv = double_conv2d_bn(8,16)
        self.layer3_conv = double_conv2d_bn(16,32)
        self.layer4_conv = double_conv2d_bn(32,64)
        self.layer5_conv = double_conv2d_bn(64,128)
        self.layer6_conv = double_conv2d_bn(128,64)
        self.layer7_conv = double_conv2d_bn(64,32)
        self.layer8_conv = double_conv2d_bn(32,16)
        self.layer9_conv = double_conv2d_bn(16,8)
        self.layer10_conv = nn.Conv2d(8,3,kernel_size=3,
                                     stride=1,padding=1,bias=True)
        
        self.deconv1 = deconv2d_bn(128,64)
        self.deconv2 = deconv2d_bn(64,32)
        self.deconv3 = deconv2d_bn(32,16)
        self.deconv4 = deconv2d_bn(16,8)

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        conv1 = self.layer1_conv(x)
        pool1 = F.max_pool2d(conv1,2)

        conv2 = self.layer2_conv(pool1)
        pool2 = F.max_pool2d(conv2,2)

        conv3 = self.layer3_conv(pool2)
        pool3 = F.max_pool2d(conv3,2)

        conv4 = self.layer4_conv(pool3)
        pool4 = F.max_pool2d(conv4,2)

        conv5 = self.layer5_conv(pool4)

        convt1 = self.deconv1(conv5)
        concat1 = torch.cat([convt1,conv4],dim=1)
        conv6 = self.layer6_conv(concat1)

        convt2 = self.deconv2(conv6)
        concat2 = torch.cat([convt2,conv3],dim=1)
        conv7 = self.layer7_conv(concat2)

        convt3 = self.deconv3(conv7)
        concat3 = torch.cat([convt3,conv2],dim=1)
        conv8 = self.layer8_conv(concat3)

        convt4 = self.deconv4(conv8)
        concat4 = torch.cat([convt4,conv1],dim=1)
        conv9 = self.layer9_conv(concat4)
        outp = self.layer10_conv(conv9)
        outp = self.sigmoid(outp)
        return outp

# 定义保存图片函数
def save_output_image(output, img_name, output_dir, input_size):
    img_name = os.path.basename(img_name)  # 获取图片文件名
    output_path = os.path.join(output_dir, img_name)
    output_image = transforms.ToPILImage()(output.squeeze(0).cpu())  # 将张量转换为 PIL 图像

    # 将输出调整大小
    output = torch.nn.functional.interpolate(output.unsqueeze(0), size=input_size, mode='bilinear', align_corners=False).squeeze(0)
    
    # 将输出转换为 PIL 图像
    output_image = transforms.ToPILImage()(output.cpu())
    
    output_image.save(output_path)

# 定义测试参数
batch_size = 16
output_dir = '/kaggle/working/test_outputs'  # 保存输出图片的目录

# 准备测试数据
test_transform = transforms.Compose([
    transforms.Resize((1216, 352)),  # 调整图像大小
    transforms.ToTensor()
])

test_dataset = TestDataset(image_folder='/kaggle/input/dataset-road/data_road/testing/image_2', transform=test_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 加载已训练好的模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Unet().to(device)
model.load_state_dict(torch.load('/kaggle/input/semantic-segmentation-model/semantic_segmentation_model.pth'))
model.eval()

# 进行推断并保存输出图片
with torch.no_grad():
    for data in test_loader:
        inputs, img_names = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        for output, img_name in zip(outputs, img_names):
#             save_output_image(output, img_name, output_dir)
            save_output_image(output, img_name, output_dir,  (352, 1216))

print('Finished Testing and Saved Output Images')
