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

def sketch_param_transformer(dataset, model, device, num_epochs=1, batch_size=1, learning_rate=1e-3):
    total_size = len(dataset)
    train_size = int(0.8 * total_size) 
    validation_size = total_size - train_size  

    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=preprocessing.io_utils.stroke_cloud_collate)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=preprocessing.io_utils.stroke_cloud_collate)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader):
            CAD_Program_path, final_edges, strokes_dict_path = batch

            stroke_objects = operation_transformer.separate_strokes_keep_order(final_edges)
            for stroke_obj in stroke_objects:
                stroke_obj.to_device(device)
            connectivity_matrix = preprocessing.stroke_graph.build_connectivity_matrix(strokes_dict_path, stroke_objects).to(device)

            optimizer.zero_grad()

            parsed_CAD_program = onshape.parse_CAD.parseCAD(CAD_Program_path)
            entity_info = onshape.parse_CAD.sketch_entity(parsed_CAD_program[0]['entities'])
            gt_labels = preprocessing.stroke_graph.build_gt_label(entity_info[0], stroke_objects)
            gt_labels = gt_labels.to(device)  

            output_probabilities = model(stroke_objects, connectivity_matrix)

            loss = criterion(output_probabilities, gt_labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}')

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in validation_loader:
                CAD_Program_path, final_edges, strokes_dict_path = batch

                stroke_objects = operation_transformer.separate_strokes_keep_order(final_edges,device)
                connectivity_matrix = preprocessing.stroke_graph.build_connectivity_matrix(strokes_dict_path, stroke_objects)

                parsed_CAD_program = onshape.parse_CAD.parseCAD(CAD_Program_path)
                entity_info = onshape.parse_CAD.sketch_entity(parsed_CAD_program[0]['entities'])
                gt_labels = preprocessing.stroke_graph.build_gt_label(entity_info[0], stroke_objects)

                stroke_objects, connectivity_matrix, gt_labels = stroke_objects.to(device), connectivity_matrix.to(device), gt_labels.to(device)

                output_probabilities = model(stroke_objects, connectivity_matrix)

                loss = criterion(output_probabilities, gt_labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(validation_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}')




device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = "cpu"
stroke_cloud_dataset = preprocessing.preprocess.get_stroke_cloud()
model = models.sketch_param_model.SketchPredictor()
model.to(device)


sketch_param_transformer(stroke_cloud_dataset, model, device)
