import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
        self, in_channels=2, out_channels=2, features=None
    ):
        super(UNET, self).__init__()
        # Encode Modules
        if features is None:
            features = [64, 128, 256, 512]
        self.ups = nn.ModuleList()
        # Decode Modules
        self.downs = nn.ModuleList()
        # 2x2 Maxpool after DoubleConv
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encode Step
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decode Step
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Output Prediction
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Execute each Encode step
        for down in self.downs:
            x = down(x)
            # Add skip connection
            skip_connections.append(x)
            x = self.pool(x)

        # Execute bottleneck
        x = self.bottleneck(x)

        # Reverse Skip Connection List
        skip_connections = skip_connections[::-1]

        # Execute each Decode step
        for idx in range(0, len(self.ups), 2):
            # Execute ConvTranspose2D on x
            x = self.ups[idx](x)
            # Retrieve skip connection
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            # Combined skip connection and x
            concat_skip = torch.cat((skip_connection, x), dim=1)
            # Execute Double Conv
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)

    assert preds.shape == x.shape


if __name__ == "__main__":
    test()
