import torch
import torch.nn as nn
import torch.nn.functional as F


class StraightLineEmbedding(nn.Module):
    def __init__(self):
        super(StraightLineEmbedding, self).__init__()
        self.fc1 = nn.Linear(6, 128)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        return x


class CurvedLineEmbedding(nn.Module):
    def __init__(self):
        super(CurvedLineEmbedding, self).__init__()
        self.fc1 = nn.Linear(10, 128)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        return x


class LineEmbeddingNetwork(nn.Module):
    def __init__(self):
        super(LineEmbeddingNetwork, self).__init__()
        self.straight_line_embedding = StraightLineEmbedding()
        self.curved_line_embedding = CurvedLineEmbedding()

    def forward(self, straight_lines, curved_lines):
        straight_embedded = self.straight_line_embedding(straight_lines)
        curved_embedded = self.curved_line_embedding(curved_lines)

        combined = torch.cat((straight_embedded, curved_embedded), dim=1)
        return combined


class StrokeTransformer(nn.Module):
    def __init__(self):
        super(StrokeTransformer, self).__init__()

    def forward(self, x):
        return x


class StrokeToCADModel(nn.Module):
    def __init__(self):
        super(StrokeToCADModel, self).__init__()
        self.line_embedding_network = LineEmbeddingNetwork()
        self.stroke_transformer = StrokeTransformer()  

    def forward(self, straight_strokes, curved_strokes):
        # Embed strokes
        stroke_embeddings = self.line_embedding_network(straight_strokes, curved_strokes)

        transformer_output = self.stroke_transformer(stroke_embeddings)
        return transformer_output
