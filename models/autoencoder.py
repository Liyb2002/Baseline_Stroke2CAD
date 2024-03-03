import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoencoderEmbed(nn.Module):
    def __init__(self, code_size, x_dim, y_dim, root_feature):
        super(AutoencoderEmbed, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            # Replace with CoordConv if available
            nn.Conv2d(1, root_feature, kernel_size=3, padding=1),
            nn.BatchNorm2d(root_feature),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(root_feature, root_feature * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(root_feature * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(root_feature * 2, root_feature * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(root_feature * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(root_feature * 4, root_feature * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(root_feature * 8),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(root_feature * 8, root_feature * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(root_feature * 16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(root_feature * 16 * (x_dim // 32) * (y_dim // 32), code_size),
            nn.Sigmoid()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(code_size, root_feature * 16 * (x_dim // 32) * (y_dim // 32)),
            nn.Unflatten(1, (root_feature * 16, x_dim // 32, y_dim // 32)),

            nn.Conv2d(root_feature * 16, root_feature * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(root_feature * 16),
            nn.ReLU(),
            nn.ConvTranspose2d(root_feature * 16, root_feature * 16, kernel_size=2, stride=2),

            nn.Conv2d(root_feature * 16, root_feature * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(root_feature * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(root_feature * 8, root_feature * 8, kernel_size=2, stride=2),

            nn.Conv2d(root_feature * 8, root_feature * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(root_feature * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(root_feature * 4, root_feature * 4, kernel_size=2, stride=2),

            nn.Conv2d(root_feature * 4, root_feature * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(root_feature * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(root_feature * 2, root_feature * 2, kernel_size=2, stride=2),

            nn.Conv2d(root_feature * 2, root_feature, kernel_size=3, padding=1),
            nn.BatchNorm2d(root_feature),
            nn.ReLU(),
            nn.ConvTranspose2d(root_feature, root_feature, kernel_size=2, stride=2),

            nn.Conv2d(root_feature, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        code = self.encoder(x)
        recons = self.decoder(code)
        return {'output_code': code, 'output_recons': recons}
