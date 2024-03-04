import torch.nn as nn
import torch

class AddCoords(nn.Module):
    def __init__(self, x_dim=256, y_dim=256, with_r=False):
        super(AddCoords, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r

    def forward(self, input_tensor):
        batch_size_tensor = input_tensor.shape[0]

        xx_channel = torch.arange(self.y_dim, device=input_tensor.device).repeat(1, self.x_dim, 1)
        yy_channel = torch.arange(self.x_dim, device=input_tensor.device).repeat(1, self.y_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (self.y_dim - 1)
        yy_channel = yy_channel.float() / (self.x_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)

        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret

class CoordConv(nn.Module):
    def __init__(self, x_dim, y_dim, root_feature, with_r=False):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(x_dim=x_dim, y_dim=y_dim, with_r=with_r)
        input_channels = 1 + 2 + (1 if with_r else 0)
        self.conv = nn.Conv2d(input_channels, root_feature, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(root_feature)
        self.relu = nn.ReLU()

    def forward(self, input_tensor):
        ret = self.addcoords(input_tensor)
        ret = self.conv(ret)
        ret = self.bn(ret)
        ret = self.relu(ret)
        return ret

class AutoencoderEmbed(nn.Module):
    def __init__(self, code_size, x_dim, y_dim, root_feature):
        super(AutoencoderEmbed, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            CoordConv(x_dim, y_dim, root_feature),
            nn.MaxPool2d(2,2),

            nn.Conv2d(root_feature, root_feature * 2, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(root_feature * 2, momentum=0.95),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(root_feature * 2, root_feature * 4, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(root_feature * 4, momentum=0.95),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(root_feature * 4, root_feature * 8, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(root_feature * 8, momentum=0.95),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(root_feature * 8, root_feature * 16, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(root_feature * 16, momentum=0.95),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            
            nn.Flatten(),
            nn.Linear(root_feature * 16 * (x_dim // 32) * (y_dim // 32), code_size),
            nn.ReLU()

        )

        # Decoder
        self.decoder = nn.Sequential(

            nn.Linear(code_size, root_feature * 16 * (x_dim // 32) * (y_dim // 32)),
            nn.Unflatten(1, (root_feature * 16, x_dim // 32, y_dim // 32)),

            nn.Conv2d(root_feature * 16, root_feature * 16, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(root_feature * 16),
            nn.ReLU(),
            nn.ConvTranspose2d(root_feature * 16, root_feature * 8, kernel_size=(2,2), stride=2),

            nn.Conv2d(root_feature * 8, root_feature * 8, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(root_feature * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(root_feature * 8, root_feature * 4, kernel_size=(2,2), stride=2),

            nn.Conv2d(root_feature * 4, root_feature * 4, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(root_feature * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(root_feature * 4, root_feature * 2, kernel_size=(2,2), stride=2),

            nn.Conv2d(root_feature * 2, root_feature * 2, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(root_feature * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(root_feature * 2, root_feature, kernel_size=(2,2), stride=2),


            nn.ConvTranspose2d(root_feature, 1, kernel_size=(2,2), stride=2),
            nn.Sigmoid()

        )

    def forward(self, x):
        code = self.encoder(x)
        reconstructed = self.decoder(code)
        return {'output_recons': reconstructed}
