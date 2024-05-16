import torch
import torch.nn as nn
from model import DarkNet19
import torch.nn.functional as F
from loss import build_target, yolo_loss
from torch.autograd import Variable


# https://github.com/tztztztztz/yolov2.pytorch/blob/master/yolov2.py 참조

# Fine-Grained Features
class ReorgLayer(nn.Module): # 512 * 26 * 26 크기의 feature map -> 2048 * 13 * 13 크기의 feature map으로 변형
    def __init__(self, stride=2):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        B, C, H, W = x.data.size()
        # print(B, C, H, W)
        ws = self.stride
        hs = self.stride
        x = x.view(B, C, int(H / hs), hs, int(W / ws), ws).transpose(3, 4).contiguous()
        x = x.view(B, C, int(H / hs * W / ws), hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, int(H / hs), int(W / ws)).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, int(H / hs), int(W / ws)) # torch.Size([1, 2048, 13, 13])
        # print(f'reorg shape : {x.shape}')
        return x

class Yolov2(nn.Module):
    num_classes=20
    num_anchors = 5
    def __init__(self, classes=None, weights_file=False):
        super(Yolov2, self).__init__()
        if classes:
            self.num_classes = len(classes)

        # darknet backbone - 마지막 conv layer는 사용하지 않음
        darknet19 = DarkNet19()

        if weights_file:
            print('load pretrained weight from {}'.format(weights_file))
            darknet19.load_weights(weights_file)
            print('pretrained weight loaded!')


        self.conv1 = darknet19.create_conv_layer()
        self.conv2 = darknet19.create_conv_layer_2()

        #detection layers
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False), # padding = int(kernel_size-1)/2
        #     nn.BatchNorm2d(1024),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(1024),
        #     nn.LeakyReLU(0.1, inplace=True)
        # )
        self.downsampler = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(in_channels=3072, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(1024),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv2d(1024, (5 + self.num_classes) * self.num_anchors, kernel_size=1, stride=1, padding=0)
        # )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=3072, out_channels=3072, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3072),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=3072, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, (5 + self.num_classes) * self.num_anchors, kernel_size=1, stride=1, padding=0)
        )
        self.reorg = ReorgLayer()


    def forward(self, x, gt_boxes=None, gt_classes=None, num_boxes=None, training=False):
        x = self.conv1(x) # Feature extraction by Darknet-19
        shortcut = self.reorg(x) # 512 * 26 * 26 크기의 feature map을 512 * 13 * 13 크기의 4개로 변형한 다음 2048 * 13 * 13 크기의 feature map으로 변형
        x = self.conv2(x) # Feature extraction by Darknet-19, output : 1024 * 13 * 13
        # print(f'feature shape : {x.shape}')
        # x = self.conv3(x)
        x = torch.cat([shortcut, x], dim=1) # 2048 * 13 * 13 + 1024 * 13 * 13 -> 3072 * 13 * 13

        # 최종 feature map 얻음.
        out = self.conv4(x) # torch.Size([1, 125, 13, 13]) -> 각 grid cell별로 5개의 bounding box가 20개의 classes score와
                            # confidence, x, y, w, h를 예측해서 channel의 수는 125
        # print(f'최종 결과 : {out.shape}')

        batch_size, _, h, w = out.size()

        # out.permute(0, 2, 3, 1) -> (1, 13, 13, 125)
        # (1, 13, 13, 125).view(batch_size, h * w * self.num_anchors, 5 + self.num_classes) -> (1, 13*13*5, 5+20)
        # 즉 (1, 13*13*5, 5+20)은 최종 feature map의 5개의 anchor box별로 classes score과 confidence, x, y, w, h의 정보가 들어있음.
        out = out.permute(0, 2, 3, 1).contiguous().view(batch_size, h * w * self.num_anchors, 5 + self.num_classes)

        # 확률과 bounding box의 x, y좌표를 0~1사이의 값으로 나타내기 위해 sigmoid 사용
        xy_pred = torch.sigmoid(out[:, :, 0:2]) # 25개의 torch에서 0, 1은 x, y
        conf_pred = torch.sigmoid(out[:, :, 4:5]) # 4는 confidence
        hw_pred = torch.exp(out[:, :, 2:4]) # 2, 3은 h, w
        class_score = out[:, :, 5:] # classes score는 5부터 끝까지
        class_pred = F.softmax(class_score, dim=-1)
        delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)

        if training:
            output_variable = (delta_pred, conf_pred, class_score)
            output_data = [v.data for v in output_variable]
            gt_data = (gt_boxes, gt_classes, num_boxes) # ground truth data
            target_data = build_target(output_data, gt_data, h, w)

            target_variable = [Variable(v) for v in target_data]
            box_loss, iou_loss, class_loss = yolo_loss(output_variable, target_variable)

            return box_loss, iou_loss, class_loss

        return delta_pred, conf_pred, class_pred

if __name__ == '__main__':
    model = Yolov2()
    im = torch.randn(1, 3, 416, 416)
    out = model(im)
    delta_pred, conf_pred, class_pred = out
    print('delta_pred size:', delta_pred.size())
    print('conf_pred size:', conf_pred.size())
    print('class_pred size:', class_pred.size())
