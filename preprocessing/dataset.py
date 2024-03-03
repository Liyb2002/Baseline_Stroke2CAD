import torch
import os
from tqdm import tqdm
import json
from glob import glob

from torch.utils.data import Dataset
from io_utils import read_json_file


class StrokeDataset(Dataset):
    def __init__(self, data_path, batch_size=1):
        self.data_path = data_path
        self.batch_size = batch_size
        self.CAD_stroke_pairs = self.get_files(data_path, 0, batch_size)

    def __len__(self):
        return len(self.CAD_stroke_pairs)

    def get_files(self, data_path, start_index=0, batch_size=1):
        CAD_stroke_pairs = []

        if os.path.exists(data_path):
            sub_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
            sub_folders = sub_folders[start_index:start_index + batch_size]

            for sub_folder in tqdm(sub_folders):
                sub_folder_path = os.path.join(data_path, sub_folder)
                CAD_path = os.path.join(sub_folder_path, 'parsed_features.json')
                if not os.path.exists(CAD_path):
                    continue

                CAD_Program = read_json_file(CAD_path)

                stroke_folders = [d for d in os.listdir(sub_folder_path) if os.path.isdir(os.path.join(sub_folder_path, d))]
                for stroke_folder in stroke_folders:
                    training_data_path = os.path.join(sub_folder_path, stroke_folder, 'training_data')

                    npr_images = glob(os.path.join(training_data_path, 'npr*.png'))
                    for npr_image in npr_images:
                        CAD_stroke_pairs.append({'CAD_Program': CAD_Program, 'npr_image': npr_image})

        return CAD_stroke_pairs
                
        
