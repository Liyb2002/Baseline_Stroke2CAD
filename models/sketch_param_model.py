import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn

import models.LineEmbedding

class SketchPredictor(nn.Module):
    def __init__(self, embedding_size = 128, 
                 num_layers = 8 , num_gnn_layers=3, num_heads = 8 , 
                 hidden_size = 128, gnn_hidden_size = 128, 
                 max_seq_length = 1000, dropout=0.1):
        super(SketchPredictor, self).__init__()
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.dropout = dropout

        self.embedding = models.LineEmbedding.LineEmbeddingNetwork_nonPos()

        self.gnn_layers = nn.ModuleList([
            geom_nn.GCNConv(gnn_hidden_size, gnn_hidden_size) 
            for _ in range(num_gnn_layers)
        ])
        self.gnn_first_layer = geom_nn.GCNConv(embedding_size, gnn_hidden_size)

        self.transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout),
            num_layers
        )

        self.fc_out = nn.Linear(hidden_size + embedding_size, 1)

    def forward(self, batch_embedding, batch_connectivity_matrix):

        batch_stroke_probabilities = []

        for stroke_embedding, adjacency_matrix in zip (batch_embedding, batch_connectivity_matrix):

            transformer_output = self.transformer_layers(stroke_embedding)

            gnn_output = self.gnn_first_layer(stroke_embedding, adjacency_matrix)
            for layer in self.gnn_layers:
                gnn_output = layer(gnn_output, adjacency_matrix)

            combined_output = torch.cat([transformer_output, gnn_output], dim=-1)
            
            stroke_probabilities = torch.sigmoid(self.fc_out(combined_output))
            
            batch_stroke_probabilities.append(stroke_probabilities)

        return batch_stroke_probabilities
