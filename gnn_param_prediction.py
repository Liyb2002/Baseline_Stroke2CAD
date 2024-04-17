import preprocessing.io_utils
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm 
import os

import preprocessing.preprocess
import preprocessing.collate_fn
import models.gnn.gnn
import data_structure.stroke_class
import onshape.parse_CAD
import operation_transformer
import preprocessing.stroke_graph
import utils.face_aggregate
import utils.plotting
import build123.protocol

def train_gnn_param_prediction(dataset, device, batch_size=1, learning_rate=5e-4, epochs=50):
    model = models.gnn.gnn.InstanceModule()  # Assume InstanceModule is correctly imported and defined
    model.to(device)

    # checkpoint_path = os.path.join(preprocessing.io_utils.home_dir, "output", "gnn_model_Op", "gnn_model_Op" + ".ckpt")
    # loaded_model = preprocessing.io_utils.load_model(model, checkpoint_path)
    # if loaded_model is not None:
    #     return loaded_model

    
    total_size = len(dataset)
    train_size = int(0.8 * total_size) 
    validation_size = total_size - train_size  

    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=preprocessing.collate_fn.stroke_cloud_collate)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=preprocessing.collate_fn.stroke_cloud_collate)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss() 

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()  
            gnn_graph = batch[0]
            predictions = model(gnn_graph.x_dict, gnn_graph.edge_index_dict)
            # labels = gnn_graph['stroke'].y.to(device).long()
            labels = gnn_graph['stroke'].z.to(device)
            
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in validation_loader:
                gnn_graph = batch[0]
                predictions = model(gnn_graph.x_dict, gnn_graph.edge_index_dict)
                # labels = gnn_graph['stroke'].y.to(device).long()
                labels = gnn_graph['stroke'].z.to(device)
                loss = criterion(predictions, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(validation_loader)

        if avg_val_loss < 0.1:
            break

        print(f"Epoch {epoch + 1}/{epochs} - Validation Loss: {avg_val_loss:.4f}")

    preprocessing.io_utils.save_model(model, "gnn_model_Op")

    return model



def get_class_predictions(predictions):
    _, predicted_classes = torch.max(predictions, dim=1)
    return predicted_classes.unsqueeze(1)

def filter_predictions_by_class(predicted_classes, class_index):
    return (predicted_classes == class_index).nonzero(as_tuple=False).squeeze()

def get_top_strokes_for_label(predictions, label_index, top_k=10):
    if label_index >= predictions.shape[1]:
        raise ValueError("label_index is out of bounds of the prediction matrix second dimension")
    
    label_scores = predictions[:, label_index]
    top_strokes_indices = torch.topk(label_scores, k=top_k, largest=True, sorted=True)[1]
    
    return top_strokes_indices

def run_gnn_param_prediction():
    gnn_cloud_dataset = preprocessing.preprocess.get_gnn_graph()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = 'cpu'
    gnn_Predictor_model = train_gnn_param_prediction(gnn_cloud_dataset, device)

    example_graph = gnn_cloud_dataset[0]
    # utils.plotting.plot_3d_graph_strokes(example_graph)

    x_dict = example_graph.x_dict
    edge_index_dict = example_graph.edge_index_dict

    gnn_Predictor_model.eval()

    with torch.no_grad():
        predictions = gnn_Predictor_model(x_dict, edge_index_dict)
    
    print("predictions", predictions.shape)

    operation_of_interest = 2
    operation_interest_predictions = predictions[:, operation_of_interest]

    _ , top_strokes_indices = torch.topk(operation_interest_predictions, 10)

    gt_mat = example_graph['stroke'].z.to(device)
    gt_operation_interest = gt_mat[:, operation_of_interest]
    _ , gt_top_strokes_indices = torch.topk(gt_operation_interest, 10)
    print("gt_top_strokes_indices", gt_top_strokes_indices)
 


    plane_points_list, plane_stroke_ids = utils.face_aggregate.find_planes_gnn(top_strokes_indices, example_graph)

    # for (plane, plane_stroke_id) in zip (planes, plane_stroke_ids):
        # preprocessing.stroke_graph.plot_3D(plane)
        # print("plane_stroke_id", plane_stroke_id)
        # print("plane", plane)

        # confidence = 0
        # for id in plane_stroke_id:
        #     prob = flat_matrix[id]
        #     confidence += prob / len(plane_stroke_id)
        
        # print("confidence", confidence)
    
    build123.protocol.build_sketch(plane_points_list[0])

    print("unique points for plane", utils.face_aggregate.find_unique_points(plane_points_list[0]))
    print("ids", plane_stroke_ids[0])

    return predictions




run_gnn_param_prediction()