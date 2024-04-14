import preprocessing.io_utils
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm 
import os

import preprocessing.preprocess
import preprocessing.collate_fn
import models.sketch_param_model
import data_structure.stroke_class
import onshape.parse_CAD
import operation_transformer
import preprocessing.stroke_graph
import utils.face_aggregate


def train_gnn_param_prediction(dataset, device, batch_size=8, learning_rate=5e-4,):
    model = models.sketch_param_model.SketchPredictor()
    model.to(device)
    
    total_size = len(dataset)
    train_size = int(0.8 * total_size) 
    validation_size = total_size - train_size  

    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=preprocessing.collate_fn.stroke_cloud_collate)


    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    epoch = 0
    for i in range (1):
        epoch += 1
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader):
            gnn_graph = batch
            print("gnn_graph", gnn_graph[0])



def run_gnn_param_prediction():
    gnn_cloud_dataset = preprocessing.preprocess.get_gnn_graph()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = 'cpu'
    gnn_Predictor_model = train_gnn_param_prediction(gnn_cloud_dataset, device)


run_gnn_param_prediction()