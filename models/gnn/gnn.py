import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, EdgeConv, SAGEConv
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
        self.net1 = SAGEConv(hidden_channels, hidden_channels)  # Adjusted for hidden_channels output from encoder
        self.net2 = SAGEConv(hidden_channels, num_classes)      # Output the number of classes

    def forward(self, x_dict, edge_index_dict):
        x = self.encoder(x_dict, edge_index_dict) # Assuming 'stroke' is the feature matrix
        
        # Apply Graph SAGE Convolutions, assuming edge_index_dict contains the correct edge indices for each layer
        edge_index = edge_index_dict[('stroke', 'intersects', 'stroke')]

        # Apply Graph SAGE Convolutions, using the specified edge indices
        x = F.relu(self.net1(x, edge_index))  # Apply first SAGEConv with ReLU activation
        x = self.net2(x, edge_index)  # Output raw logits suitable for loss computation

        return x