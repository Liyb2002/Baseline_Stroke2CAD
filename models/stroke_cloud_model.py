import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import random

import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, reset_positions):
        seq_len, _ = x.shape
        pe = self.pe[:seq_len, :]   

        reset_positions = reset_positions.to(pe.device).unsqueeze(-1) 
        pe = pe.masked_fill(reset_positions, 0)
        pe_cumulative = torch.cumsum(pe, dim=0)

        x = x + pe_cumulative
        return x



class StraightLineEmbedding(nn.Module):
    def __init__(self, embedding_size = 128, max_len=5000):
        super(StraightLineEmbedding, self).__init__()
        self.fc1 = nn.Linear(6, embedding_size)
        self.pos_encoder = PositionalEncoding(embedding_size, max_len)


    def forward(self, x):

        #get rid of this stack if positional encoding
        x = torch.stack(x)
        
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.pos_encoder(x, reset_positions)
        return x



class CurvedLineEmbedding(nn.Module):
    def __init__(self, num_target_points = 10, embedding_size = 128, max_len=5000):
        super(CurvedLineEmbedding, self).__init__()
        self.num_target_points = num_target_points
        self.embedding_size = embedding_size
        self.fc = nn.Linear(num_target_points * 3, embedding_size)
        self.pos_encoder = PositionalEncoding(embedding_size, max_len)


    def forward(self, x):
        processed_curves = torch.stack([self.process_curve(curve) for curve in x])

        flattened_curves = processed_curves.view(processed_curves.size(0), -1)
        embedded_curves = self.fc(flattened_curves)
        embedded_curves = F.relu(embedded_curves)
        # embedded_curves = self.pos_encoder(embedded_curves, reset_positions)

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

    def forward(self, strokes):        

        straight_strokes = []
        curved_strokes = []
        for stroke in strokes:
            if stroke.type == 'straight_stroke':
                straight_strokes.append(stroke)
            else:
                curved_strokes.append(stroke)

        straight_features = [torch.cat([line.point0, line.point1]) for line in straight_strokes]
        # straight_features_padded = pad_sequence(straight_features, batch_first=True)
        # straight_padded_length = straight_features_padded.size(0)
        # reset_positions_straight = self.create_reset_positions(straight_strokes, straight_padded_length)
        # straight_embedded = self.straight_line_embedding(straight_features_padded, reset_positions_straight)
        straight_embedded = self.straight_line_embedding(straight_features)

        if len(curved_strokes) == 0:
            curved_embedded = torch.tensor([])
        else:
            curved_features = [line.points.flatten() for line in curved_strokes]
            # curved_features_padded = pad_sequence(curved_features, batch_first=True)
            # curved_padded_length = curved_features_padded.size(0)
            # reset_positions_curved = self.create_reset_positions(curved_strokes, curved_padded_length)
            # curved_embedded = self.curved_line_embedding(curved_features_padded, reset_positions_curved )
            curved_embedded = self.curved_line_embedding(curved_features)

        combined = torch.cat((straight_embedded, curved_embedded), dim=0)
        return combined
    

    def create_padding_mask(self, padded_sequences):
        mask = padded_sequences != 0
        return mask


    def create_reset_positions(self, strokes, padded_length):
        reset_positions = [False] * padded_length
        cumulative_length = 0

        for stroke in strokes:
            length = 2 if hasattr(stroke, 'point1') else len(stroke.points)
            
            if cumulative_length < padded_length:
                reset_positions[cumulative_length] = True

            cumulative_length += length

        return torch.tensor(reset_positions, dtype=torch.bool)




class StrokeToCADModel(nn.Module):
    def __init__(self, embedding_size = 128, 
                 num_layers = 6 , num_heads = 8 , hidden_size = 512, 
                 vocab_size = 3, max_seq_length = 1000, dropout=0.1):
        super(StrokeToCADModel, self).__init__()
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.dropout = dropout

        self.embedding = LineEmbeddingNetwork()
        self.transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout),
            num_layers
        )
        self.fc = nn.Linear(embedding_size, vocab_size)

    def forward(self, strokes, target_operation=None):
        stroke_embedding = self.embedding(strokes)

        encoder_output = self.transformer_layers(stroke_embedding)

        output = self.fc(encoder_output[-1])

        if target_operation is not None:
            loss = nn.CrossEntropyLoss()(output.unsqueeze(0), target_operation.unsqueeze(0))
            return loss
        else:
            return output

    def predict_next_operation(self, straight_strokes, curved_strokes):
        stroke_embedding = self.embedding(straight_strokes, curved_strokes)

        encoder_output = self.transformer_layers(stroke_embedding)

        output = self.fc(encoder_output[0])

        predicted_operation = output.argmax(dim=-1)

        return predicted_operation