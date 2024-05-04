
import brep_read
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader



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
        








def run():
    step_path = '../preprocessing/canvas/step_4.step'
    dataset = brep_read.BRep_Dataset(step_path)

    # train_graph_embedding(dataset)


run()