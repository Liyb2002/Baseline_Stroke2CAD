import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn

import models.stroke_cloud_model

class SketchPredictor(nn.Module):
    def __init__(self, embedding_size = 128, 
                 num_layers = 6 , num_heads = 8 , 
                 hidden_size = 512, gnn_hidden_size = 128, 
                 max_seq_length = 1000, dropout=0.1):
        super(SketchPredictor, self).__init__()
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.dropout = dropout

        self.embedding = models.stroke_cloud_model.LineEmbeddingNetwork()
        self.gnn = geom_nn.GCNConv(embedding_size, gnn_hidden_size)


        self.transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout),
            num_layers
        )

        self.fc_out = nn.Linear(hidden_size + gnn_hidden_size, 1)


    def forward(self, straight_strokes, curved_strokes, adjacency_matrix):
        stroke_embedding = self.embedding(straight_strokes, curved_strokes)

        transformer_output = self.transformer_layers(stroke_embedding)

        gnn_output = self.gnn(stroke_embedding, adjacency_matrix)

        combined_output = torch.cat([transformer_output, gnn_output], dim=-1)
        
        stroke_probabilities = torch.sigmoid(self.fc_out(combined_output))
        return stroke_probabilities

