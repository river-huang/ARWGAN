import torch
import torch.nn as nn
from options import HiDDenConfiguration
from model.Dense_block import Bottleneck


class Encoder(nn.Module):

    def conv1(self, in_channel, out_channel):
        return nn.Conv2d(in_channels=in_channel,
                         out_channels=out_channel,
                         stride=1,
                         kernel_size=7, padding=3)

    def conv2(self, in_channel, out_chanenl):
        return nn.Conv2d(in_channels=in_channel,
                         out_channels=out_chanenl,
                         stride=1,
                         kernel_size=3,
                         padding=1)

    def __init__(self, config: HiDDenConfiguration):
        super(Encoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks

        self.first_layer = nn.Sequential(
            self.conv2(3, self.conv_channels)
        )

        self.second_layer = nn.Sequential(
            self.conv2(self.conv_channels, self.conv_channels),
            nn.BatchNorm2d(self.conv_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.third_layer = nn.Sequential(
            self.conv2(self.conv_channels * 2, self.conv_channels),
            nn.BatchNorm2d(self.conv_channels),
            nn.LeakyReLU(inplace=True),
            self.conv2(self.conv_channels, self.conv_channels),
            nn.BatchNorm2d(self.conv_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.fourth_layer = nn.Sequential(
            self.conv2(self.conv_channels * 3 + config.message_length, self.conv_channels),
            nn.BatchNorm2d(self.conv_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.Dense_block1 = Bottleneck(self.conv_channels + config.message_length, self.conv_channels)
        self.Dense_block2 = Bottleneck(self.conv_channels * 2 + config.message_length, self.conv_channels)
        self.Dense_block3 = Bottleneck(self.conv_channels * 3 + config.message_length, self.conv_channels)
        self.Dense_block_a1 = Bottleneck(self.conv_channels, self.conv_channels)
        self.Dense_block_a2 = Bottleneck(self.conv_channels * 2, self.conv_channels)
        self.Dense_block_a3 = Bottleneck(self.conv_channels * 3, self.conv_channels)

        self.fivth_layer = nn.Sequential(
            nn.BatchNorm2d(self.conv_channels + config.message_length),
            nn.ReLU(inplace=True),
            self.conv2(self.conv_channels + config.message_length, self.conv_channels),
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(inplace=True),
            self.conv2(self.conv_channels, config.message_length),
        )
        self.sixth_layer = nn.Sequential(
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(inplace=True),
            self.conv2(self.conv_channels, self.conv_channels),
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(inplace=True),
            self.conv2(self.conv_channels, config.message_length),
            nn.Softmax(dim=1)
        )
        self.softmax = nn.Sequential(nn.Softmax(dim=1))

        self.final_layer = nn.Sequential(nn.Conv2d(config.message_length, 3, kernel_size=3, padding=1),
                                         )

    def forward(self, image, message):
        H, W = image.size()[2], image.size()[3]

        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)
        expanded_message = expanded_message.expand(-1, -1, H, W)

        feature0 = self.first_layer(image)
        feature1 = self.Dense_block1(torch.cat((feature0, expanded_message), 1), last=True)
        feature2 = self.Dense_block2(torch.cat((feature0, expanded_message, feature1), 1), last=True)
        feature3 = self.Dense_block3(torch.cat((feature0, expanded_message, feature1, feature2), 1), last=True)
        feature3 = self.fivth_layer(torch.cat((feature3, expanded_message), 1))
        feature_attention = self.Dense_block_a3(self.Dense_block_a2(self.Dense_block_a1(feature0)), last=True)
        feature_mask = (self.sixth_layer(feature_attention)) * 30
        feature = feature3 * feature_mask
        im_w = self.final_layer(feature)
        im_w = im_w + image
        return im_w







