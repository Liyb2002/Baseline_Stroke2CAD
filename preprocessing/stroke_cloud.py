from torch.utils.data import Dataset
import os
from tqdm import tqdm
from io_utils import read_json_file

class Stroke_Cloud_Dataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.CAD_stroke_pairs = self.get_files(data_path)

    def __len__(self):
        return len(self.straight_stroke) + len(self.curve_stroke)

    def __getitem__(self, idx):
        item = self.CAD_stroke_pairs[idx]
        CAD_Program = item['CAD_Program']
        final_edges = item['final_edges']

        return {'CAD_Program': CAD_Program, 'final_edges': final_edges}

    def get_files(self, data_path):
        CAD_stroke_pairs = []

        sub_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        for sub_folder in tqdm(sub_folders):
            sub_folder_path = os.path.join(data_path, sub_folder)
            CAD_path = os.path.join(sub_folder_path, 'parsed_features.json')
            if not os.path.exists(CAD_path):
                    continue

            CAD_Program = read_json_file(CAD_path)

            stroke_folders = [d for d in os.listdir(sub_folder_path) if os.path.isdir(os.path.join(sub_folder_path, d))]
            for stroke_folder in stroke_folders:
                final_edges_path = os.path.join(sub_folder_path, stroke_folder, 'final_edges.json')
                if os.path.exists(final_edges_path):
                    final_edges = read_json_file(final_edges_path)
                    CAD_stroke_pairs.append({'CAD_Program': CAD_Program, 'final_edges': final_edges})

        return CAD_stroke_pairs


    def get_staight_stroke(self):
        print("get_staight_stroke")