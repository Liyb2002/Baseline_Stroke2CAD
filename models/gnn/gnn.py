import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, EdgeConv
from torch_geometric.data import HeteroData

import models.gnn.basic

class SemanticModule(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=32, mlp_channels=[64, 32], num_classes = 10):
        super(SemanticModule, self).__init__()
        self.local_head = models.gnn.basic.GeneralHeteroConv(['temp_previous_add', 'intersects_mean'], in_channels, hidden_channels)


        self.layers = nn.ModuleList([
            models.gnn.basic.ResidualGeneralHeteroConvBlock(['temp_previous_add', 'intersects_mean'], hidden_channels, hidden_channels),
            models.gnn.basic.ResidualGeneralHeteroConvBlock(['temp_previous_add', 'intersects_mean'], hidden_channels, hidden_channels),
            models.gnn.basic.ResidualGeneralHeteroConvBlock(['temp_previous_add', 'intersects_mean'], hidden_channels, hidden_channels),
            models.gnn.basic.ResidualGeneralHeteroConvBlock(['temp_previous_add', 'intersects_mean'], hidden_channels, mlp_channels[0])
        ])

        self.mlp = models.gnn.basic.MLPLinear(mlp_channels)

    def forward(self, x_dict, edge_index_dict):

        x_dict = self.local_head(x_dict, edge_index_dict)

        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
        x = self.mlp(x_dict['stroke'])

        return x


class InstanceModule(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=32, mlp_channels= [64, 32]):
        super(InstanceModule, self).__init__()
        num_classes = 10
        
        in_features_decoder = mlp_channels[-1]

        self.encoder = SemanticModule(in_channels, hidden_channels, mlp_channels, num_classes)
        self.decoder = nn.Sequential(
            nn.Linear(in_features_decoder, hidden_channels),  
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, num_classes)  
        )

    def forward(self, x_dict, edge_index_dict):
        features = self.encoder(x_dict, edge_index_dict)
        return torch.sigmoid(self.decoder(features))  
