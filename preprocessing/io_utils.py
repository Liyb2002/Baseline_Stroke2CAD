import os
import json
from torch.utils.data.dataloader import default_collate

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def stroke_collate(batch):
    # Separate CAD_Program and stroke_image
    CAD_Programs = [item['CAD_Program'] for item in batch]
    stroke_images = [item['stroke_image'] for item in batch]

    # Use default_collate for images, handle CAD_Programs separately
    batched_images = default_collate(stroke_images)
    return CAD_Programs, batched_images
