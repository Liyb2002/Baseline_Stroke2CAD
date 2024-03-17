import torch
import os
from tqdm import tqdm
import json
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


from preprocessing.io_utils import read_json_file
import matplotlib.pyplot as plt


class opensketch_dataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.image_paths = self.get_files(data_path)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), 
        transforms.Resize((256, 256)),              
        transforms.ToTensor(),                     
        transforms.Normalize((0.5,), (0.5,))    
    ])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):

        item = self.stroke_image_pairs[idx]
        CAD_Program = item['CAD_Program']
        stroke_image_path = item['npr_image']

        stroke_image = Image.open(stroke_image_path).convert('L') 
        if self.transform:
            stroke_image = self.transform(stroke_image)
        
        return {'CAD_Program': CAD_Program, 'stroke_image': stroke_image}

    def get_files(self, data_path):
        if not os.path.exists(data_path):
            print("cannot find data_path", data_path)

        found_paths = []
        sub_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

        for sub_folder in tqdm(sub_folders):
            sub_folder_path = os.path.join(data_path, sub_folder)

            for root, dirs, files in os.walk(sub_folder_path):
                if 'view1_presentation_opaque.png' in files:
                    file_path = os.path.join(root, 'view1_presentation_opaque.png')
                    found_paths.append(file_path)

        return found_paths
