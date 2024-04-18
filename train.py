import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import os
from PIL import Image

# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.image_filenames = os.listdir(image_folder)
        self.label_filenames = os.listdir(label_folder)
        self.image_filenames = sorted(self.image_filenames)
        self.label_filenames = sorted(self.label_filenames)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_filenames[idx])
        label_name = os.path.join(self.label_folder, self.label_filenames[idx])

        image = Image.open(img_name)
        label = Image.open(label_name)
#         label = Image.open(label_name).convert('L')  # 将标签图像转换为灰度图像
        
#         save_path = "/kaggle/working/um_lane_L.png"  # 替换为实际的保存路径
#         img = Image.open(label_name).convert('L')
#         img.save(save_path)
#         print("saved")

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

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

# 设置训练参数
batch_size = 16
learning_rate = 0.01
epochs = 10

# 准备数据
transform = transforms.Compose([
    transforms.Resize((1216, 352)),  # 调整图像大小
#     transforms.Grayscale(num_output_channels=1),  # 将图像转换为单通道灰度图像
    transforms.ToTensor()
])

dataset = CustomDataset(image_folder='/kaggle/input/dataset-road/data_road/training/image_2',
             label_folder='/kaggle/input/dataset-road/data_road/training/gt_image_2',
             transform=transform)

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Unet().to(device)
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(epochs):
    # 训练阶段
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)  # 将数据移动到 GPU 上
#         print("inputs",inputs.shape)
        optimizer.zero_grad()
        outputs = model(inputs)
#         print("outputs",outputs.shape)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(running_loss)
        if i % 10 == 9:
            print('[Epoch %d, Batch %5d] Train Loss: %.3f' %(epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    # 验证阶段
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data[0].to(device), data[1].to(device)  # 将数据移动到 GPU 上
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)

            val_loss += loss.item()

    print('[Epoch %d] Validation Loss: %.3f' %(epoch + 1, val_loss / len(val_loader)))

# 保存模型权重
torch.save(model.state_dict(), '/kaggle/working/semantic_segmentation_model.pth')
print('Finished Training')
