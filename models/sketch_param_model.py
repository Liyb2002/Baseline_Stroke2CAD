import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn

import models.stroke_cloud_model

class CrossAttention(nn.Module):
    def __init__(self, size, num_heads):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=size, num_heads=num_heads)

    def forward(self, stroke_embeddings, graph_embeddings):
        attended_output, _ = self.attention(stroke_embeddings, graph_embeddings, graph_embeddings)
        return attended_output


class SketchPredictor(nn.Module):
    def __init__(self, embedding_size = 128, 
                 num_layers = 6 , num_heads = 8 , 
                 hidden_size = 128, gnn_hidden_size = 128, 
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


        self.cross_attention = CrossAttention(embedding_size, num_heads)

        self.transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout),
            num_layers
        )

        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, strokes, adjacency_matrix):
        stroke_embedding = self.embedding(strokes)

        gnn_output = self.gnn(stroke_embedding, adjacency_matrix)

        cross_attended_output = self.cross_attention(stroke_embedding, gnn_output)

        transformer_output = self.transformer_layers(cross_attended_output)

        stroke_probabilities = torch.sigmoid(self.fc_out(transformer_output))
        
        return stroke_probabilities

