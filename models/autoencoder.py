import torch.nn as nn

class AutoencoderEmbed(nn.Module):
    def __init__(self, code_size, x_dim, y_dim, root_feature):
        super(AutoencoderEmbed, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, root_feature, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(root_feature),

            nn.Conv2d(root_feature, root_feature * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(root_feature * 2),

            nn.Flatten(),
            nn.Linear(root_feature * 2 * (x_dim // 4) * (y_dim // 4), code_size),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(code_size, root_feature * 2 * (x_dim // 4) * (y_dim // 4)),
            nn.ReLU(),
            nn.Unflatten(1, (root_feature * 2, x_dim // 4, y_dim // 4)),

            nn.ConvTranspose2d(root_feature * 2, root_feature, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(root_feature),

            nn.ConvTranspose2d(root_feature, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        code = self.encoder(x)
        reconstructed = self.decoder(code)
        return {'output_recons': reconstructed}
