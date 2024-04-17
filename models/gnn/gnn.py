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
        
        self.mlp_channels = mlp_channels

        self.encoder = SemanticModule(in_channels, hidden_channels, mlp_channels)
        self.decoder = nn.Sequential(
            nn.Linear(mlp_channels[-1], hidden_channels),  
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1)  
        )

    def forward(self, x_dict, edge_index_dict):
        node_features = self.encoder(x_dict, edge_index_dict)
        num_nodes = node_features.size(0)
        
        row_indices, col_indices = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing='ij')
        
        node_features_expanded = node_features.unsqueeze(1).expand(-1, num_nodes, -1)
        
        node_pair_features = node_features_expanded[row_indices, col_indices]
        
        edge_features = node_pair_features.view(-1, self.mlp_channels[-1])
        out = self.decoder(edge_features).view(num_nodes, num_nodes)

        return torch.sigmoid(out)
