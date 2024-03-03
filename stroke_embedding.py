from models.autoencoder import AutoencoderEmbed
import torch


autoencoder = AutoencoderEmbed(code_size=64, x_dim=256, y_dim=256, root_feature=32)

test_input = torch.rand(1, 1, 256, 256)

# Forward pass
output = autoencoder(test_input)
print(output['output_code'].shape, output['output_recons'].shape)
