import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import preprocessing.preprocess
import models.stroke_cloud_transformer
import data_structure.stroke_class

def operation_transformer(dataset, model, num_epochs=1, batch_size=2, learning_rate=1e-3):

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=preprocessing.io_utils.stroke_cloud_collate)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        for batch in data_loader:
            CAD_Programs, final_edges= batch

            straight_strokes, curve_strokes = separate_strokes(final_edges) 
            outputs = model(straight_strokes, curve_strokes)

            operation_types = [stroke.operation_type for stroke in straight_strokes + curve_strokes]
            operation_types = torch.tensor(operation_types, dtype=torch.long)

            loss = criterion(outputs, operation_types)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()  
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in data_loader:
                CAD_Programs, final_edges = batch
                straight_strokes, curve_strokes = separate_strokes(final_edges)

                outputs = model(straight_strokes, curve_strokes)
                operation_types = [stroke.operation_type for stroke in straight_strokes + curve_strokes]
                operation_types = torch.tensor(operation_types, dtype=torch.long)

                loss = criterion(outputs, operation_types)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += operation_types.size(0)
                correct += (predicted == operation_types).sum().item()

        avg_loss = total_loss / len(data_loader)
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy}%')



def separate_strokes(final_edges):

    straight_strokes = []
    curve_strokes = []

    for combined in final_edges:

        for key in combined:
            data_block = combined[key]
            
            per_stroke_keys = []
            for per_stroke_key in data_block:
                per_stroke_keys.append(per_stroke_key)
            
            if len(data_block[per_stroke_keys[0]]) == 2:            
                straight_strokes.append(data_structure.stroke_class.StraightLine3D(data_block))
            else:
                curve_strokes.append(data_structure.stroke_class.CurveLine3D(data_block))

    return straight_strokes, curve_strokes




stroke_cloud_dataset = preprocessing.preprocess.get_stroke_cloud()
model = models.stroke_cloud_transformer.StrokeToCADModel(7)
operation_transformer(stroke_cloud_dataset, model)
