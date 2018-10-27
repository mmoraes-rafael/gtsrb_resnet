import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3_drop = nn.Dropout2d(p=0.2)
        self.conv3_bn = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv6_drop = nn.Dropout2d(p=0.2)
        self.conv6_bn = nn.BatchNorm2d(64)

        self.conv7 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv7_bn = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv8_bn = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv9_drop = nn.Dropout2d(p=0.2)
        self.conv9_bn = nn.BatchNorm2d(128)

        self.conv10 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv10_bn = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv11_bn = nn.BatchNorm2d(256)
        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv12_drop = nn.Dropout2d(p=0.2)
        self.conv12_bn = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256 * 6 * 6 , 2048)
        self.fc1_bn = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, nclasses)

    def forward(self, inp):
        res = F.relu(self.conv1_bn(self.conv1(inp)))
        x =   F.relu(self.conv2_bn(self.conv2(res)))
        x =   self.conv3_drop(self.conv3(x))
        block1_out = F.relu(self.conv3_bn(F.max_pool2d(x + res, 2)))    

        res = F.relu(self.conv4_bn(self.conv4(block1_out)))
        x =   F.relu(self.conv5_bn(self.conv5(res)))
        x = self.conv6_drop(self.conv6(x))
        block2_out = F.relu(self.conv6_bn(F.max_pool2d(x + res, 2)))

        res = F.relu(self.conv7_bn(self.conv7(block2_out)))
        x =   F.relu(self.conv8_bn(self.conv8(res)))
        x = self.conv9_drop(self.conv9(x))
        block3_out = F.relu(self.conv9_bn(F.max_pool2d(x + res, 2)))

        res = F.relu(self.conv10_bn(self.conv10(block3_out)))
        x =   F.relu(self.conv11_bn(self.conv11(res)))
        x = F.relu(self.conv12_bn(self.conv12_drop(self.conv12(x + res))))
   
        x = x.view(-1, 256 * 6 * 6)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.dropout(x, training=self.training, p=0.2)
        x = self.fc2(x)
        return F.log_softmax(x)
