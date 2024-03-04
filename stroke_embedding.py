import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from torch.optim import Adam
import os

import preprocessing.preprocess
import preprocessing.io_utils
import models.autoencoder


def train_autoencoder(dataset, num_epochs=1, learning_rate=1e-3, batch_size=32):
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=preprocessing.io_utils.stroke_collate)
    stroke_encoder = models.autoencoder.AutoencoderEmbed(code_size=64, x_dim=256, y_dim=256, root_feature=32).to(device)


    checkpoint_path = os.path.join(preprocessing.io_utils.home_dir, "output", "stroke_encoder", "stroke_encoder" + ".ckpt")
    loaded_model = preprocessing.io_utils.load_model(stroke_encoder, checkpoint_path)
    if loaded_model is not None:
        return loaded_model


    criterion = torch.nn.MSELoss()
    optimizer = Adam(stroke_encoder.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=True)

        for i, (_, stroke_image)  in loop:
            stroke_image = stroke_image.to(device)
            output = stroke_encoder(stroke_image)['output_recons']
            loss = criterion(output, stroke_image)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())
    
    preprocessing.io_utils.save_model(stroke_encoder, "stroke_encoder")

    return stroke_encoder



dataset = preprocessing.preprocess.get_stroke_dataset()
train_autoencoder(dataset)


