import torch
import torch.nn as nn
from torchvision import models
from config import image_size


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bt1 = nn.BatchNorm2d(out_c)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bt2 = nn.BatchNorm2d(out_c)
        self.relu2 = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bt1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bt2(x)
        x = self.relu2(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv_block = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, inputs):
        x = self.conv_block(inputs)
        pool = self.pool(x)
        return x, pool


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_block_1 = EncoderBlock(3, 64)
        self.encoder_block_2 = EncoderBlock(64, 128)
        self.encoder_block_3 = EncoderBlock(128, 256)
        self.encoder_block_4 = EncoderBlock(256, 512)

        self.bottle_neck = nn.Conv2d(512, 1024, kernel_size=1, padding=0)

    def forward(self, inputs):
        x1, pool1 = self.encoder_block_1(inputs)
        x2, pool2 = self.encoder_block_2(pool1)
        x3, pool3 = self.encoder_block_3(pool2)
        x4, pool4 = self.encoder_block_4(pool3)
        bottle_neck = self.bottle_neck(pool4)
        return x1, x2, x3, x4, bottle_neck


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.upsampling = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(out_c+out_c, out_c)

    def forward(self, inputs, skip_conn):
        x = self.upsampling(inputs)
        x = torch.cat([x, skip_conn], 1)
        x = self.conv_block(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.docker_block_1 = DecoderBlock(1024, 512)
        self.docker_block_2 = DecoderBlock(512, 256)
        self.docker_block_3 = DecoderBlock(256, 128)
        self.docker_block_4 = DecoderBlock(128, 64)

        self.decoder_classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, convs):
        conv1, conv2, conv3, conv4, bottle_neck = convs

        d1 = self.docker_block_1(bottle_neck, conv4)
        d2 = self.docker_block_2(d1, conv3)
        d3 = self.docker_block_3(d2, conv2)
        d4 = self.docker_block_4(d3, conv1)
        outputs = self.decoder_classifier(d4)
        return outputs


class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_classes)

    def forward(self, inputs):
        convs = self.encoder(inputs)
        outputs = self.decoder(convs)
        return outputs


""" Test Model """

# input_image = torch.rand((1, 3, image_size[0], image_size[1]))
# model = UNet(num_classes=13)
# # Total parameters and trainable parameters.
# total_params = sum(p.numel() for p in model.parameters())
# print(f"{total_params:,} total parameters.")
# total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"{total_trainable_params:,} training parameters.")
# outputs = model(input_image)
# print(outputs.shape)