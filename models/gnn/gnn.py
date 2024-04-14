import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, EdgeConv
from torch_geometric.data import HeteroData

import models.gnn.basic

class SemanticModule(nn.Module):
    def __init__(self, in_channels = 6, out_channels = 32, num_classes = 7):
        super(SemanticModule, self).__init__()
        self.layers = nn.ModuleList([
            models.gnn.basic.ResidualGeneralHeteroConvBlock(['temp_previous_add', 'intersects_mean'], in_channels, out_channels),
            models.gnn.basic.ResidualGeneralHeteroConvBlock(['temp_previous_add', 'intersects_mean'], out_channels, out_channels)
        ])
        self.classifier = nn.Linear(out_channels, num_classes)  

    def forward(self, x_dict, edge_index_dict):
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
        x = x_dict['stroke']
        return self.classifier(x)


class InstanceModule(nn.Module):
    def __init__(self, in_channels = 6, out_channels = 32):
        super(InstanceModule, self).__init__()
        self.encoder = SemanticModule(in_channels, out_channels)
        self.decoder = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, 1)  # Binary output for edge existence
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.encoder(x_dict, edge_index_dict)
        x = x_dict['stroke']
        return torch.sigmoid(self.decoder(x))
