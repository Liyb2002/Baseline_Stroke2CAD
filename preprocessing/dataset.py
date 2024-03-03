import torch
import os
from tqdm import tqdm
import json

from torch.utils.data import Dataset
from io_utils import read_json_file


class StrokeDataset(Dataset):
    def __init__(self, data_path):
        self.get_files(data_path)
        self.sample_pair()        

    def dataset_size(self):
        dataset_size = 0
        for _, _ in self.CAD_stroke_pair.items():
            dataset_size += 1

    def sample_pair(self):
        sample_folder = './sampled_data/'
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)

        if not self.CAD_stroke_pair:
            print("No data available.")
            return

        first_key = next(iter(self.CAD_stroke_pair))
        first_cad_entry = self.CAD_stroke_pair[first_key]

        if not first_cad_entry['strokes_data']:
            print("No strokes data available for the first CAD.")
            return
        first_strokes_data = first_cad_entry['strokes_data'][0]
        sample_data = {
            'CAD': first_cad_entry['CAD'],
            'strokes_data': first_strokes_data
        }

        sample_file_path = os.path.join(sample_folder, f'sampled_pair.json')

        try:
            with open(sample_file_path, 'w') as file:
                json.dump(sample_data, file, indent=4)
            print(f"First CAD-stroke pair saved to {sample_file_path}")
        except Exception as e:
            print(f"Error saving first CAD-stroke pair: {e}")
 

    def get_files(self, data_path):
        self.CAD_stroke_pair = {}  
        if os.path.exists(data_path):
            sub_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        
        for sub_folder in tqdm(sub_folders):
            sub_folder_path = os.path.join(data_path, sub_folder)
            CAD_path = os.path.join(data_path, sub_folder, 'parsed_features.json')

            if os.path.exists(CAD_path):
                self.CAD_stroke_pair[sub_folder] = {
                    'CAD': read_json_file(CAD_path),
                    'strokes_data': []
                }
                        
            stroke_folders = [d for d in os.listdir(sub_folder_path) if os.path.isdir(os.path.join(sub_folder_path, d))]
            for stroke_folder in stroke_folders:
                stroke_folder_path = os.path.join(sub_folder_path, stroke_folder)
                strokes_dict_path = os.path.join(stroke_folder_path, 'strokes_dict.json')
                final_edges_path = os.path.join(stroke_folder_path, 'final_edges.json')

                stroke_data = {}
                if os.path.exists(strokes_dict_path):
                    stroke_data['strokes_dict'] = read_json_file(strokes_dict_path)
                if os.path.exists(final_edges_path):
                    stroke_data['final_edges'] = read_json_file(final_edges_path)

                if stroke_data:
                    self.CAD_stroke_pair[sub_folder]['strokes_data'].append(stroke_data)


