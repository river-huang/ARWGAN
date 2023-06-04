import torch
import torch.nn as nn
from options import HiDDenConfiguration
from model.Dense_block import Bottleneck


class Decoder(nn.Module):

    def conv1(self, in_channel, out_channel):
        return nn.Conv2d(in_channels=in_channel,
                         out_channels=out_channel,
                         stride=1,
                         kernel_size=7,
                         padding=3)

    def conv2(self, in_channel, out_chanenl):
        return nn.Conv2d(in_channels=in_channel,
                         out_channels=out_chanenl,
                         stride=1,
                         kernel_size=3,
                         padding=1)

    def __init__(self, config: HiDDenConfiguration):
        super(Decoder, self).__init__()
        self.channels = config.decoder_channels

        self.first_layer = nn.Sequential(self.conv2(3, self.channels),
                                         nn.BatchNorm2d(self.channels),
                                         nn.LeakyReLU(inplace=True))

        self.second_layer = nn.Sequential(self.conv2(self.channels, self.channels),
                                          nn.BatchNorm2d(self.channels),
                                          nn.LeakyReLU(inplace=True))

        self.third_layer = nn.Sequential(self.conv2(self.channels * 2, self.channels),
                                         nn.BatchNorm2d(self.channels),
                                         nn.LeakyReLU(inplace=True))

        self.fourth_layer = nn.Sequential(self.conv2(self.channels * 3, self.channels),
                                          nn.BatchNorm2d(self.channels),
                                          nn.LeakyReLU(inplace=True))

        self.Dense_block1 = Bottleneck(self.channels, self.channels)
        self.Dense_block2 = Bottleneck(self.channels * 2, self.channels)
        self.Dense_block3 = Bottleneck(self.channels * 3, self.channels)

        self.fivth_layer = nn.Sequential(self.conv2(self.channels, config.message_length),
                                         nn.BatchNorm2d(config.message_length),
                                         nn.ReLU(inplace=True))

        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(config.message_length, config.message_length)

    def forward(self, image_with_wm):
        feature0 = self.first_layer(image_with_wm)
        feature1 = self.second_layer(feature0)
        feature2 = self.third_layer(torch.cat([feature0, feature1], dim=1))
        feature3 = self.fourth_layer(torch.cat([feature0, feature1, feature2], dim=1))
        x = self.fivth_layer(feature3)
        x = self.pooling(x)
        x = self.linear(x.squeeze_(3).squeeze_(2))
        return x
