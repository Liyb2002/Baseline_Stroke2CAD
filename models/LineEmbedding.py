import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import random

import numpy as np



class StraightLineEmbedding_nonPos(nn.Module):
    def __init__(self, embedding_size = 128, max_len=5000):
        super(StraightLineEmbedding_nonPos, self).__init__()
        self.fc1 = nn.Linear(6, embedding_size)


    def forward(self, x):
        x = torch.stack(x)
        x = self.fc1(x)
        x = F.relu(x)
        return x



class CurvedLineEmbedding_nonPos(nn.Module):
    def __init__(self, num_target_points = 10, embedding_size = 128, max_len=5000):
        super(CurvedLineEmbedding_nonPos, self).__init__()
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

class LineEmbeddingNetwork_nonPos(nn.Module):
    def __init__(self):
        super(LineEmbeddingNetwork_nonPos, self).__init__()
        self.straight_line_embedding = StraightLineEmbedding_nonPos()
        self.curved_line_embedding = CurvedLineEmbedding_nonPos()

    def forward(self, strokes):        

        embedded_strokes = []

        for stroke in strokes:
            if stroke.type == 'straight_stroke':
                straight_feature = torch.cat([stroke.point0, stroke.point1])
                embedded_stroke = self.straight_line_embedding([straight_feature])
            else:
                curved_feature = stroke.points.flatten()
                embedded_stroke = self.curved_line_embedding([curved_feature])

            embedded_strokes.append(embedded_stroke)

        combined = torch.cat(embedded_strokes, dim=0)

        return combined


