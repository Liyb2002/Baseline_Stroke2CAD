import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm 
import os

import preprocessing.preprocess
import models.sketch_param_model
import data_structure.stroke_class
import onshape.parse_CAD
import operation_transformer
import preprocessing.stroke_graph

def train_sketch_param_transformer(dataset, device, batch_size=1, learning_rate=5e-4):

    model = models.sketch_param_model.SketchPredictor()
    model.to(device)

    # checkpoint_path = os.path.join(preprocessing.io_utils.home_dir, "output", "SketchPredictor_model", "SketchPredictor_model" + ".ckpt")
    # loaded_model = preprocessing.io_utils.load_model(model, checkpoint_path)
    # if loaded_model is not None:
    #     return loaded_model


    total_size = len(dataset)
    train_size = int(0.8 * total_size) 
    validation_size = total_size - train_size  

    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=preprocessing.io_utils.stroke_cloud_collate)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=preprocessing.io_utils.stroke_cloud_collate)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    epoch = 0
    for i in range (1):
        epoch += 1
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
            gt_labels = gt_labels


            output_probabilities = model(stroke_objects, connectivity_matrix)
            loss = criterion(output_probabilities, gt_labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            break
                    
        avg_train_loss = total_train_loss / len(train_loader)
        print(f'Epoch [{epoch}], Train Loss: {avg_train_loss:.4f}')

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(validation_loader):
                CAD_Program_path, final_edges, strokes_dict_path = batch

                stroke_objects = operation_transformer.separate_strokes_keep_order(final_edges)
                for stroke_obj in stroke_objects:
                    stroke_obj.to_device(device)

                connectivity_matrix = preprocessing.stroke_graph.build_connectivity_matrix(strokes_dict_path, stroke_objects)

                parsed_CAD_program = onshape.parse_CAD.parseCAD(CAD_Program_path)
                entity_info = onshape.parse_CAD.sketch_entity(parsed_CAD_program[0]['entities'])
                gt_labels = preprocessing.stroke_graph.build_gt_label(entity_info[0], stroke_objects)
                gt_labels = gt_labels.to(device)  
                gt_labels = gt_labels

                output_probabilities = model(stroke_objects, connectivity_matrix)

                loss = criterion(output_probabilities, gt_labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(validation_loader)

        if avg_val_loss < 0.08:
            break
        print(f'Epoch [{epoch}], Val Loss: {avg_val_loss:.4f}')

    preprocessing.io_utils.save_model(model, "SketchPredictor_model")
    return model




def run_sketch_param_prediction():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = "cpu"
    stroke_cloud_dataset = preprocessing.preprocess.get_stroke_cloud()

    SketchPredictor_model = train_sketch_param_transformer(stroke_cloud_dataset, device)

    data_loader = DataLoader(stroke_cloud_dataset, batch_size=1, shuffle=True, collate_fn=preprocessing.io_utils.stroke_cloud_collate)
    sampled_batch = next(iter(data_loader))
    CAD_Program_path, final_edges, strokes_dict_path = sampled_batch

    stroke_objects = operation_transformer.separate_strokes_keep_order(final_edges)
    for stroke_obj in stroke_objects:
        stroke_obj.to_device(device)
    connectivity_matrix = preprocessing.stroke_graph.build_connectivity_matrix(strokes_dict_path, stroke_objects).to(device)

    parsed_CAD_program = onshape.parse_CAD.parseCAD(CAD_Program_path)
    entity_info = onshape.parse_CAD.sketch_entity(parsed_CAD_program[0]['entities'])

    SketchPredictor_model.eval()

    with torch.no_grad():
        output_probabilities = SketchPredictor_model(stroke_objects, connectivity_matrix)

    flat_matrix = output_probabilities.flatten()

    top_values, indices = torch.topk(flat_matrix, 10)

    print("top_values", top_values)
    print("indices", indices)

    return top_values, indices


def face_aggregate():
    print("hi")

run_sketch_param_prediction()