import os
import json
from torch.utils.data.dataloader import default_collate
from pathlib import Path
import torch


home_dir = Path(__file__).parent.parent

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def stroke_collate(batch):
    CAD_Programs = [item['CAD_Program'] for item in batch]
    stroke_images = [item['stroke_image'] for item in batch]

    batched_images = default_collate(stroke_images)
    return CAD_Programs, batched_images


def stroke_cloud_collate(batch):
    CAD_Programs = [item[0] for item in batch]
    final_edges_list = [item[1] for item in batch]

    return CAD_Programs, final_edges_list



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
