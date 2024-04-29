import os
import json
from torch.utils.data.dataloader import default_collate
from pathlib import Path
import torch


home_dir = Path(__file__).parent.parent

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    



#---------------------------------------------------------------------#
def save_model(model, model_name="model"):
    save_dir = os.path.join(home_dir, "output", model_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the full path for the model file
    model_file_name = model_name + ".ckpt"
    model_path = os.path.join(save_dir, model_file_name)

    # Save the model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def load_model(model, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        print(f"Loaded model from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))
        return model
    else:
        return None


#---------------------------------------------------------------------#

def transform_point_list(point_list):
    transformed_list = []

    for i in range(len(point_list) - 1):
        transformed_list.append(point_list[i])
        transformed_list.append(point_list[i+1])

    transformed_list.append(point_list[-1])
    transformed_list.append(point_list[0])

    return transformed_list


def ensure_sequential_vertex_order(vertex_pairs):
    #ensure we have the format [[a,b], [b,c]...[x,a]]
    if len(vertex_pairs) <= 1:
        return vertex_pairs

    ordered_pairs = [vertex_pairs[0]]

    for i in range(1, len(vertex_pairs)):
        current_pair = vertex_pairs[i]
        previous_end = ordered_pairs[-1][1] 

        if current_pair[0] != previous_end:
            if current_pair[1] == previous_end:
                current_pair = [current_pair[1], current_pair[0]]
            else:
                print(f"Warning: Pair {current_pair} does not connect sequentially with the previous pair.")
        ordered_pairs.append(current_pair)

    return ordered_pairs
