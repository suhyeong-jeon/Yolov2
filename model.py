import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
from util.network import WeightLoader

torch.manual_seed(0)

model_paths = {
    'darknet19': 'https://s3.ap-northeast-2.amazonaws.com/deepbaksuvision/darknet19-deepBakSu-e1b3ec1e.pth'
}

class GlobalAveragePool2d(nn.Module):
    def __init__(self):
        super(GlobalAveragePool2d, self).__init__()

    def forward(self, x):
        # print(x.size())
        x = F.avg_pool2d(x, x.size()[2:])
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        return x

class DarkNet19(nn.Module):
    def __init__(self, pretrained=False):
        super(DarkNet19, self).__init__()
        self.features_1 = self.create_conv_layer()
        self.features_2 = self.create_conv_layer_2()
        self.classifier = self.create_classifier()

        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_paths['darknet19'], progress=True))

    def forward(self, x):
        x = self.features_1(x)
        x = self.features_2(x)
        print(x.shape)
        x = self.classifier(x)
        print(x.shape)

        return x

    def create_conv_layer(self):
        self.features = nn.Sequential(OrderedDict([
            ('layer1', nn.Sequential(OrderedDict([
                ('conv1_1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn1_1', nn.BatchNorm2d(32)),
                ('leaky1_1', nn.LeakyReLU(0.1, inplace=True)),
                ('maxpool1', nn.MaxPool2d(2, 2))
            ]))),
            ('layer2', nn.Sequential(OrderedDict([
                ('conv2_1', nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn2_1', nn.BatchNorm2d(64)),
                ('leaky2_1', nn.LeakyReLU(0.1, inplace=True)),
                ('maxpool2', nn.MaxPool2d(2, 2))
            ]))),
            ('layer3', nn.Sequential(OrderedDict([
                ('conv3_1', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn3_1', nn.BatchNorm2d(128)),
                ('leaky3_1', nn.LeakyReLU(0.1, inplace=True)),
                ('conv3_2', nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)),
                ('bn3_2', nn.BatchNorm2d(64)),
                ('leaky3_2', nn.LeakyReLU(0.1, inplace=True)),
                ('conv3_3', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn3_3', nn.BatchNorm2d(128)),
                ('leaky3_3', nn.LeakyReLU(0.1, inplace=True)),
                ('maxpool3', nn.MaxPool2d(2, 2))
            ]))),
            ('layer4', nn.Sequential(OrderedDict([
                ('conv4_1', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn4_1', nn.BatchNorm2d(256)),
                ('leaky4_1', nn.LeakyReLU(0.1, inplace=True)),
                ('conv4_2', nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)),
                ('bn4_2', nn.BatchNorm2d(128)),
                ('leaky4_2', nn.LeakyReLU(0.1, inplace=True)),
                ('conv4_3', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn4_3', nn.BatchNorm2d(256)),
                ('leaky4_3', nn.LeakyReLU(0.1, inplace=True)),
                ('maxpool4', nn.MaxPool2d(2, 2))
            ]))),
            ('layer5', nn.Sequential(OrderedDict([
                ('conv5_1', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn5_1', nn.BatchNorm2d(512)),
                ('leaky5_1', nn.LeakyReLU(0.1, inplace=True)),
                ('conv5_2', nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)),
                ('bn5_2', nn.BatchNorm2d(256)),
                ('leaky5_2', nn.LeakyReLU(0.1, inplace=True)),
                ('conv5_3', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn5_3', nn.BatchNorm2d(512)),
                ('leaky5_3', nn.LeakyReLU(0.1, inplace=True)),
                ('conv5_4', nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)), # padding=1?
                ('bn5_4', nn.BatchNorm2d(256)),
                ('leaky5_4', nn.LeakyReLU(0.1, inplace=True)),
                ('conv5_5', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn5_5', nn.BatchNorm2d(512)),
                ('leaky5_5', nn.LeakyReLU(0.1, inplace=True)),
                # ('maxpool5', nn.MaxPool2d(2, 2))
            ]))),
        ]))

        return self.features

    def create_conv_layer_2(self):
        self.features = nn.Sequential(OrderedDict([
            ('layer6', nn.Sequential(OrderedDict([
                ('maxpool5', nn.MaxPool2d(2, 2)),
                ('conv6_1',
                 nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn6_1', nn.BatchNorm2d(1024)),
                ('leaky6_1', nn.LeakyReLU(0.1, inplace=True)),
                ('conv6_2',
                 nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)),
                ('bn6_2', nn.BatchNorm2d(512)),
                ('leaky6_3', nn.LeakyReLU(0.1, inplace=True)),
                ('conv6_3',
                 nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn6_3', nn.BatchNorm2d(1024)),
                ('leaky6_3', nn.LeakyReLU(0.1, inplace=True)),
                ('conv6_4',
                 nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)),
                # padding=1?
                ('bn6_4', nn.BatchNorm2d(512)),
                ('leaky6_4', nn.LeakyReLU(0.1, inplace=True)),
                ('conv6_5',
                 nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn6_5', nn.BatchNorm2d(1024)),
                ('leaky6_5', nn.LeakyReLU(0.1, inplace=True))
            ])))
        ]))

        return self.features

    def create_classifier(self):
        self.classifier = nn.Sequential(OrderedDict([
            ('conv7_1', nn.Conv2d(in_channels=1024, out_channels=1000, kernel_size=1, stride=1, padding=0, bias=False)), # detection task에서는 사용되지 않음
            ('globalavgpool', GlobalAveragePool2d()),
            ('softmax', nn.Softmax(dim=1))
        ]))

        return self.classifier

    def load_weights(self, weights_file):
        weights_loader = WeightLoader()
        weights_loader.load(self, weights_file)


if __name__ =='__main__':
    DarkNet19 = DarkNet19()
    # ResNet.load_state_dict(torch.load('./resnet101-cd907fc2.pth'))
    x = torch.randn((2, 3, 416, 416))
    DarkNet19(x)