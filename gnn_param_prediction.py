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


def train_gnn_param_prediction(dataset, device, batch_size=1, learning_rate=5e-4, epochs=5):
    model = models.gnn.gnn.InstanceModule()  # Assume InstanceModule is correctly imported and defined
    model.to(device)

    # checkpoint_path = os.path.join(preprocessing.io_utils.home_dir, "output", "gnn_model", "gnn_model" + ".ckpt")
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
    criterion = torch.nn.CrossEntropyLoss()  

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()  
            gnn_graph = batch[0]
            predictions = model(gnn_graph.x_dict, gnn_graph.edge_index_dict)
            # print("predictions", predictions.shape)
            labels = gnn_graph['stroke'].y.to(device).long()
            
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
                labels = gnn_graph['stroke'].y.to(device).long()
                loss = criterion(predictions, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(validation_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Validation Loss: {avg_val_loss:.4f}")

    # preprocessing.io_utils.save_model(model, "gnn_model")

    return model



def get_class_predictions(predictions):
    _, predicted_classes = torch.max(predictions, dim=1)
    return predicted_classes.unsqueeze(1)

def filter_predictions_by_class(predicted_classes, class_index):
    return (predicted_classes == class_index).nonzero(as_tuple=False).squeeze()

def run_gnn_param_prediction():
    gnn_cloud_dataset = preprocessing.preprocess.get_gnn_graph()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = 'cpu'
    gnn_Predictor_model = train_gnn_param_prediction(gnn_cloud_dataset, device)

    example_graph = gnn_cloud_dataset[0]
    x_dict = example_graph.x_dict
    edge_index_dict = example_graph.edge_index_dict

    gnn_Predictor_model.eval()

    with torch.no_grad():
        predictions = gnn_Predictor_model(x_dict, edge_index_dict)
    
    predicted_classes = get_class_predictions(predictions)
    print("Predicted class indices:", predicted_classes)

    class_of_interest = 0
    indices_of_class = filter_predictions_by_class(predicted_classes, class_of_interest)
    print(f"Indices of strokes predicted as class {class_of_interest}:", indices_of_class)

    ground_truth_labels = example_graph['stroke'].y.to(device)
    print("Ground truth labels:", ground_truth_labels)




    return predictions




run_gnn_param_prediction()