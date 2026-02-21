import torch.nn as nn


class ResBlock(nn.Module)

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        identity = x

        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))

        out += identity  
        return self.relu(out)


class I2IM(nn.Module):

    def __init__(self):
        super().__init__()

        self.init_layers()
        self.get_parameters()

    def init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.initial_conv = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.initial_in = nn.InstanceNorm2d(64, affine=True)

        self.res_blocks = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
        )

        self.output_conv = nn.Conv2d(
            64, 3, kernel_size=1, stride=1, padding=0, bias=True
        )

    def get_parameters(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.net_parameters = {"Total": total_num, "Trainable": trainable_num}
        print(f"I2IM (6×ResBlock) - Total: {total_num:,}, Trainable: {trainable_num:,}")

    def forward(self, x):

        h = self.relu(self.initial_in(self.initial_conv(x)))  

        h = self.res_blocks(h)  

        h = self.output_conv(h)  

        h += x

        i2im_out = self.sigmoid(h)

        return i2im_out
