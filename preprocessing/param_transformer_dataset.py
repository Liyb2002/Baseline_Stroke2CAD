
from torch.utils.data import Dataset
import os
from tqdm import tqdm
from preprocessing.io_utils import read_json_file
from torch.utils.data import DataLoader, random_split

class Stroke_Cloud_Dataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.CAD_stroke_pairs = self.get_files(data_path)
        print("loaded CAD_stroke_pairs: ", len(self.CAD_stroke_pairs))

    def __len__(self):
        return len(self.CAD_stroke_pairs)

    def __getitem__(self, idx):
        item = self.CAD_stroke_pairs[idx]
        CAD_Program = item['CAD_Program']
        final_edges = item['final_edges']
        strokes_dict_path = item['strokes_dict']


    def process(self, dataset, batch_size = 16):
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=preprocessing.io_utils.stroke_cloud_collate)
