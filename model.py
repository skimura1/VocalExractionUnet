import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from transforms import make_filterbanks
from util import batch_normalized


class ResDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
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
        # 2x2 Maxpool after ResDoubleConv
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)

        # Encode Step
        for feature in features:
            self.downs.append(ResDoubleConv(in_channels, feature))
            in_channels = feature

        # Decode Step
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(ResDoubleConv(feature*2, feature))

        # Bottleneck
        self.bottleneck = ResDoubleConv(features[-1], features[-1]*2)

        # Final Conv
        self.final_conv = nn.Sequential(nn.Conv2d(features[0], out_channels, kernel_size=1),
                                        nn.Sigmoid())

        # Final Activation
        # self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        skip_connections = []

        # Execute each Encode step
        for down in self.downs:
            x = down(x)
            # Add skip connection
            skip_connections.append(x)
            x = self.dropout(self.pool(x))

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
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])

            # Combined skip connection and x
            concat_skip = torch.cat((skip_connection, x), dim=1)
            # Execute Double Conv
            x = self.ups[idx + 1](concat_skip)
            x = self.dropout(x)

        return self.final_conv(x)

class Separator(nn.Module):
    def __init__(self, n_fft, hop_length, model, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stft, self.istft = make_filterbanks(
            n_fft=n_fft,
            hop_length=hop_length,
            center=False,
            sample_rate=44100.0,
            device=device
        )
        self.model = model.to(device)
        self.device = device

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x):
        """Performing the separation on audio input"""
        x = x.to(self.device)

        with torch.no_grad():
            mix_spec = self.stft(x)
            norm_spec, _, _ = batch_normalized(mix_spec)
            model_output = self.model(norm_spec)
            estimates = self.istft(model_output)

        return estimates

