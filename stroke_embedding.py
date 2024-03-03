import torch
from torch.utils.data import DataLoader
import torch.optim as optim



from models.autoencoder import AutoencoderEmbed
from preprocessing.preprocess import get_stroke_dataset
from preprocessing.io_utils import stroke_collate


def train_autoencoder(dataset, num_epochs=1, learning_rate=1e-3, batch_size=32):
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=stroke_collate)

    for epoch in range(num_epochs):
        for i, data in enumerate(data_loader):
    #         CAD_Program, stroke_image = data
    #         print("CAD_Program", CAD_Program.shape)
    #         print("stroke_image", stroke_image.shape)
            print("i", i)
            # stroke_image = data['npr_image'].to(device)
            # print("image shape", stroke_image.shape)



dataset = get_stroke_dataset()
train_autoencoder(dataset)


