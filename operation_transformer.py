import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import preprocessing.preprocess
import models.stroke_cloud_transformer
import data_structure.stroke_class

def operation_transformer(dataset, model, num_epochs=1, batch_size=2, learning_rate=1e-3):

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=preprocessing.io_utils.stroke_cloud_collate)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for batch in data_loader:
            CAD_Programs, final_edges= batch

            separate_strokes(final_edges) 
            break



def separate_strokes(final_edges):

    straight_stroke = []
    curve_strokes = []

    for combined in final_edges:

        for key in combined:
            data_block = combined[key]
            
            per_stroke_keys = []
            for per_stroke_key in data_block:
                per_stroke_keys.append(per_stroke_key)
            
            if len(data_block[per_stroke_keys[0]]) == 2:
                straight_stroke.append(data_structure.stroke_class.StraightLine3D(data_block))



stroke_cloud_dataset = preprocessing.preprocess.get_stroke_cloud()
model = models.stroke_cloud_transformer.StrokeToCADModel()
operation_transformer(stroke_cloud_dataset, model)
