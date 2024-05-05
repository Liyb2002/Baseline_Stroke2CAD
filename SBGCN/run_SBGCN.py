
import brep_read
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader

import SBGCN_network


def train_graph_embedding(dataset):

    model = SBGCN_network.FaceEdgeVertexGCN()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 16
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in data_loader:
            graph = batch[0]
            optimizer.zero_grad()

            # Forward pass
            x_t, x_p, x_f, x_e, x_v = model(graph)
            print("x_t", x_t.shape)
            print("x_p", x_p.shape)
            print("x_f", x_f.shape)
            print("x_e", x_e.shape)
            print("x_v", x_v.shape)

        








def run():
    step_path = '../preprocessing/canvas/step_4.step'
    dataset = brep_read.BRep_Dataset(step_path)

    # train_graph_embedding(dataset)


run()