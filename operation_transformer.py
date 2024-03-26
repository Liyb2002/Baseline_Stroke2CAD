import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm 

import preprocessing.preprocess
import models.stroke_cloud_transformer
import data_structure.stroke_class
import onshape.parse_CAD

def operation_transformer(dataset, model, num_epochs=3, batch_size=1, learning_rate=1e-3):

    total_size = len(dataset)
    train_size = int(0.8 * total_size) 
    validation_size = total_size - train_size  

    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=preprocessing.io_utils.stroke_cloud_collate)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=preprocessing.io_utils.stroke_cloud_collate)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader):
            CAD_Program_path, final_edges= batch

            straight_strokes, curve_strokes = separate_strokes(final_edges) 
            parsed_CAD_program = onshape.parse_CAD.parseCAD(CAD_Program_path)
            operation = parsed_CAD_program[0]['sequence'][0]['type']
            operation_id = onshape.parse_CAD.operation_to_id(operation)
            print("operation", operation)


            outputs = model(straight_strokes, curve_strokes)
            print("outputs", outputs)

            operation_types = [stroke.operation_type for stroke in straight_strokes + curve_strokes]
            operation_types = torch.tensor(operation_types, dtype=torch.long)

            loss = criterion(outputs, operation_types)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()  
        total_val_loss = 0
        total, correct = 0, 0

        with torch.no_grad():
            for batch in tqdm(validation_loader):
                CAD_Programs, final_edges = batch
                straight_strokes, curve_strokes = separate_strokes(final_edges)

                outputs = model(straight_strokes, curve_strokes)
                operation_types = [stroke.operation_type for stroke in straight_strokes + curve_strokes]
                operation_types = torch.tensor(operation_types, dtype=torch.long)

                loss = criterion(outputs, operation_types)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += operation_types.size(0)
                correct += (predicted == operation_types).sum().item()

        avg_val_loss = total_val_loss / len(validation_loader)
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation accuracy: {accuracy:.4f}')



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
model = models.stroke_cloud_transformer.StrokeToCADModel()
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# model.to(device)

operation_transformer(stroke_cloud_dataset, model)
