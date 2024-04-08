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
import utils.face_aggregate

import models.LineEmbedding

def train_sketch_param_transformer(dataset, device, batch_size=8, learning_rate=5e-4):

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
    Line_Embedding_model = models.LineEmbedding.LineEmbeddingNetwork_nonPos()

    epoch = 0
    for i in range (1):
        epoch += 1
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader):
            _, batch_final_edges, batch_strokes_dict_path = batch

            batch_stroke_objects = operation_transformer.separate_strokes_keep_order(batch_final_edges)

            batch_connectivity_matrix, _, _ = preprocessing.stroke_graph.build_connectivity_matrix(batch_strokes_dict_path, batch_stroke_objects)

            optimizer.zero_grad()

            # parsed_CAD_program = onshape.parse_CAD.parseCAD(CAD_Program_path)
            # sequences = parsed_CAD_program[0]['sequence']
            # entity_info = onshape.parse_CAD.sketch_entity(parsed_CAD_program[0]['entities'])
            # gt_labels = preprocessing.stroke_graph.build_gt_label(entity_info[0], stroke_objects)

            batch_labels = preprocessing.stroke_graph.build_gt_label_from_ID(0, batch_stroke_objects)

            batch_embedding = []
            for stroke_objects in batch_stroke_objects:
                embedding = Line_Embedding_model(stroke_objects)
                batch_embedding.append(embedding)

            batch_stroke_probabilities = model(batch_embedding, batch_connectivity_matrix)

            loss = compute_loss(batch_stroke_probabilities, batch_labels)


            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
                    
        avg_train_loss = total_train_loss / len(train_loader)
        print(f'Epoch [{epoch}], Train Loss: {avg_train_loss:.4f}')

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(validation_loader):
                _, batch_final_edges, batch_strokes_dict_path = batch

                batch_stroke_objects = operation_transformer.separate_strokes_keep_order(batch_final_edges)

                batch_connectivity_matrix, _, _ = preprocessing.stroke_graph.build_connectivity_matrix(batch_strokes_dict_path, batch_stroke_objects)

                # parsed_CAD_program = onshape.parse_CAD.parseCAD(CAD_Program_path)
                # entity_info = onshape.parse_CAD.sketch_entity(parsed_CAD_program[0]['entities'])
                # gt_labels = preprocessing.stroke_graph.build_gt_label(entity_info[0], stroke_objects)

                batch_labels = preprocessing.stroke_graph.build_gt_label_from_ID(0, batch_stroke_objects)

                for stroke_objects in batch_stroke_objects:
                    embedding = Line_Embedding_model(stroke_objects)

                output_probabilities = model(batch_stroke_objects, batch_connectivity_matrix)
                loss = criterion(output_probabilities, batch_labels)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(validation_loader)
        print(f'Epoch [{epoch}], Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < 0.05:
            break

    preprocessing.io_utils.save_model(model, "SketchPredictor_model")
    return model


def compute_loss(batch_stroke_probabilities, batch_labels):
    criterion = nn.BCELoss(reduction='none')

    max_length = max([prob.shape[0] for prob in batch_stroke_probabilities])
    batch_size = len(batch_stroke_probabilities)

    padded_probs = torch.zeros(batch_size, max_length, 1)
    padded_labels = torch.zeros(batch_size, max_length, 1)

    for i in range(batch_size):
        length = batch_stroke_probabilities[i].shape[0]
        padded_probs[i, :length, :] = batch_stroke_probabilities[i]
        padded_labels[i, :length, :] = batch_labels[i]

    mask = (padded_labels != 0).float()

    loss = criterion(padded_probs, padded_labels)
    loss *= mask

    average_loss = loss.sum() / mask.sum()

    return average_loss




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
    connectivity_matrix, raw_connectivity_matrix, plane_dict = preprocessing.stroke_graph.build_connectivity_matrix(strokes_dict_path, stroke_objects)
    connectivity_matrix = connectivity_matrix.to(device)

    parsed_CAD_program = onshape.parse_CAD.parseCAD(CAD_Program_path)
    entity_info = onshape.parse_CAD.sketch_entity(parsed_CAD_program[0]['entities'])

    SketchPredictor_model.eval()

    with torch.no_grad():
        output_probabilities = SketchPredictor_model(stroke_objects, connectivity_matrix)

    flat_matrix = output_probabilities.flatten()

    top_values, top_indices = torch.topk(flat_matrix, 10)

    print("top_values", top_values)
    print("indices", top_indices)

    planes, plane_stroke_ids = utils.face_aggregate.find_planes(top_indices, stroke_objects, raw_connectivity_matrix)

    for (plane, plane_stroke_id) in zip (planes, plane_stroke_ids):
        # preprocessing.stroke_graph.plot_3D(plane)
        print("plane_stroke_id", plane_stroke_id)

        confidence = 0
        for id in plane_stroke_id:
            prob = flat_matrix[id]
            confidence += prob / len(plane_stroke_id)
        
        print("confidence", confidence)
        
    return top_values, top_indices



run_sketch_param_prediction()