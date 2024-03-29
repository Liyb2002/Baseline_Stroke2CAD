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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader):
            CAD_Program_path, final_edges= batch

            straight_strokes, curve_strokes = separate_strokes(final_edges) 
            parsed_CAD_program = onshape.parse_CAD.parseCAD(CAD_Program_path)

            operations = [program['sequence'][0]['type'] for program in parsed_CAD_program]
            operation_ids = [onshape.parse_CAD.operation_to_id(operation) for operation in operations] 
            target_operation = torch.tensor(operation_ids[0])

            outputs = model(straight_strokes, curve_strokes)

            loss = criterion(outputs.unsqueeze(0), target_operation.unsqueeze(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

        model.eval()
        total_validation_loss = 0

        with torch.no_grad():
            for batch in tqdm(validation_loader):
                CAD_Program_path, final_edges = batch
                straight_strokes, curve_strokes = separate_strokes(final_edges)
                parsed_CAD_program = onshape.parse_CAD.parseCAD(CAD_Program_path)
                operations = [program['sequence'][0]['type'] for program in parsed_CAD_program]
                operation_ids = [onshape.parse_CAD.operation_to_id(operation) for operation in operations]

                straight_strokes = straight_strokes
                curve_strokes = curve_strokes
                target_operation = torch.tensor(operation_ids[0])

                outputs = model(straight_strokes, curve_strokes)
                # predicted_operation = torch.argmax(outputs)
                # print(f"Predicted Operation: {predicted_operation}")

                loss = criterion(outputs.unsqueeze(0), target_operation.unsqueeze(0))

                total_validation_loss += loss.item()

        avg_validation_loss = total_validation_loss / len(validation_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_validation_loss:.4f}")



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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

operation_transformer(stroke_cloud_dataset, model)
