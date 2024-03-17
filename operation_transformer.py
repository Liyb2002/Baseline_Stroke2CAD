import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import preprocessing.preprocess
import models.stroke_cloud_transformer

def operation_transformer(dataset, model, num_epochs=1, batch_size=2, learning_rate=1e-3):

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=preprocessing.io_utils.stroke_cloud_collate)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for batch in data_loader:
            CAD_Programs, final_edges= batch

            separate_strokes(final_edges) 



def separate_strokes(final_edges):
    print("len", len(final_edges))

stroke_cloud_dataset = preprocessing.preprocess.get_stroke_cloud()
model = models.stroke_cloud_transformer.StrokeToCADModel()
operation_transformer(stroke_cloud_dataset, model)
