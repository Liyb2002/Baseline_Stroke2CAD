import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import random

import numpy as np

class StraightLineEmbedding(nn.Module):
    def __init__(self, embedding_size = 128):
        super(StraightLineEmbedding, self).__init__()
        self.fc1 = nn.Linear(6, embedding_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        return x


class CurvedLineEmbedding(nn.Module):
    def __init__(self, num_target_points = 10, embedding_size = 128):
        super(CurvedLineEmbedding, self).__init__()
        self.num_target_points = num_target_points
        self.embedding_size = embedding_size
        self.fc = nn.Linear(num_target_points * 3, embedding_size)

    def forward(self, x):
        processed_curves = torch.stack([self.process_curve(curve) for curve in x])

        flattened_curves = processed_curves.view(processed_curves.size(0), -1)
        embedded_curves = self.fc(flattened_curves)
        embedded_curves = F.relu(embedded_curves)

        return embedded_curves

    def process_curve(self, curve):
        num_points_in_curve = curve.size(0) // 3

        if num_points_in_curve > self.num_target_points:
            sampled_indices = np.random.choice(num_points_in_curve, self.num_target_points, replace=False)
            sampled_indices = np.sort(sampled_indices) * 3
            sampled_indices = np.hstack([sampled_indices + i for i in range(3)])
            curve = curve[sampled_indices]
        elif num_points_in_curve < self.num_target_points:
            padding_size = self.num_target_points * 3 - curve.size(0)
            padding = torch.zeros(padding_size)
            curve = torch.cat((curve, padding))

        return curve


class LineEmbeddingNetwork(nn.Module):
    def __init__(self):
        super(LineEmbeddingNetwork, self).__init__()
        self.straight_line_embedding = StraightLineEmbedding()
        self.curved_line_embedding = CurvedLineEmbedding()

    def forward(self, straight_strokes, curved_strokes):        
        straight_features = [torch.tensor([line.point0, line.point1]).flatten() for line in straight_strokes]
        curved_features = [torch.tensor(line.points).flatten() for line in curved_strokes]

        # Pad the sequences if necessary
        straight_features_padded = pad_sequence(straight_features, batch_first=True)
        curved_features_padded = pad_sequence(curved_features, batch_first=True)

        # print("len straight_features", len(straight_features_padded))
        # print("len curved_features", len(curved_features_padded))

        straight_embedded = self.straight_line_embedding(straight_features_padded)
        curved_embedded = self.curved_line_embedding(curved_features_padded)

        combined = torch.cat((straight_embedded, curved_embedded), dim=0)
        return combined
    
    def create_padding_mask(self, padded_sequences):
        mask = padded_sequences != 0
        return mask



class StrokeTransformer(nn.Module):
    def __init__(self, num_features, num_classes, num_layers=3, num_heads=4):
        super(StrokeTransformer, self).__init__()
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=num_features, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(num_features, num_classes)


    def forward(self, x):
        x = x.permute(1, 0, 2)
        transformed = self.transformer_encoder(x)
        x = transformed[0, :, :]
        x = self.fc(x)

        return x


class StrokeToCADModel(nn.Module):
    def __init__(self, num_classes):
        super(StrokeToCADModel, self).__init__()
        self.line_embedding_network = LineEmbeddingNetwork()
        self.stroke_transformer = StrokeTransformer(256, num_classes)  

    def forward(self, straight_strokes, curved_strokes):
        stroke_embeddings = self.line_embedding_network(straight_strokes, curved_strokes)
        print("done embedding")
        transformer_output = self.stroke_transformer(stroke_embeddings)
        return transformer_output
