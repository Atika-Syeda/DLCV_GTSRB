import torch
import torch.nn as nn
import torch.nn.functional as F

class GTSRBnet(nn.Module):
    def __init__(
        self,
        n_classes,
        img_ch=1,
        channels=[16, 32, 64, 128, 200],
        device=['cuda' if torch.cuda.is_available() else 'cpu'][0],
        kernel=3,
    ):
        super().__init__()
        self.device = device
        self.channels = channels

        self.Conv = nn.Sequential()
        self.Conv.add_module(
            "conv0",
            convblock(ch_in=img_ch, ch_out=channels[0], kernel_sz=kernel, block=0),
        )
        for k in range(1, len(channels)):
            self.Conv.add_module(
                f"conv{k}",
                convblock(
                    ch_in=channels[k - 1], ch_out=channels[k], kernel_sz=kernel, block=k
                ),
            )

        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(self.channels[-1]*8*8, 350)
        self.fc2 = nn.Linear(350, n_classes)

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
            )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
            )
   
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x, normalize=False, verbose=False):
        # encoding path
        xout = []
        x = self.Conv[0](x)
        xout.append(x)
        for k in range(1, len(self.Conv)):
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            x = self.Conv[k](x)
            xout.append(x)

        # transform the input
        x = self.conv_drop(x)
        x = x.view(-1, self.channels[-1]*8*8)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class convblock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_sz, block=-1):
        super().__init__()
        self.conv = nn.Sequential()
        self.block = block
        if self.block != 0:
            self.conv.add_module("conv_0", batchconv(ch_in, ch_out, kernel_sz))
        else:
            self.conv.add_module("conv_0", batchconv0(ch_in, ch_out, kernel_sz))
        self.conv.add_module("conv_1", batchconv(ch_out, ch_out, kernel_sz))

    def forward(self, x):
        x = self.conv[1](self.conv[0](x))
        return x


def batchconv0(ch_in, ch_out, kernel_sz):
    return nn.Sequential(
        nn.BatchNorm2d(ch_in, eps=1e-5, momentum=0.1),
        nn.Conv2d(ch_in, ch_out, kernel_sz, padding=kernel_sz // 2, bias=False),
    )


def batchconv(ch_in, ch_out, sz):
    return nn.Sequential(
        nn.BatchNorm2d(ch_in, eps=1e-5, momentum=0.1),
        nn.ReLU(inplace=True),
        nn.Conv2d(ch_in, ch_out, sz, padding=sz // 2, bias=False),
    )

