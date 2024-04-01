import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm 

import preprocessing.preprocess
import models.sketch_param_model
import data_structure.stroke_class
import onshape.parse_CAD
import operation_transformer
import preprocessing.stroke_graph

def sketch_param_transformer(dataset, model, num_epochs=3, batch_size=1, learning_rate=1e-3):
    total_size = len(dataset)
    train_size = int(0.8 * total_size) 
    validation_size = total_size - train_size  

    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=preprocessing.io_utils.stroke_cloud_collate)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=preprocessing.io_utils.stroke_cloud_collate)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader):
            CAD_Program_path, final_edges, strokes_dict_path = batch

            stroke_objects = operation_transformer.separate_strokes_keep_order(final_edges) 
            connectivity_matrix = preprocessing.stroke_graph.build_connectivity_matrix(strokes_dict_path, stroke_objects)

            optimizer.zero_grad()
            print("num_strokes", len(stroke_objects))
            output_probabilities = model(stroke_objects, connectivity_matrix)
            print("num_strokes", len(stroke_objects), "output_probabilities", output_probabilities.shape)

            parsed_CAD_program = onshape.parse_CAD.parseCAD(CAD_Program_path)
            
            entity_info = onshape.parse_CAD.sketch_entity(parsed_CAD_program[0]['entities'])

            # print("first sketch", entity_info[0])
            break



stroke_cloud_dataset = preprocessing.preprocess.get_stroke_cloud()
model = models.sketch_param_model.SketchPredictor()

sketch_param_transformer(stroke_cloud_dataset, model)
