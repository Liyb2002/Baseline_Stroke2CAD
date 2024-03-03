import json
import os

from dataset import StrokeDataset

data_path = '../dataset'
save_folder = '../output/sampled_data'

stroke_DS = StrokeDataset(data_path)
stroke_DS.save_sample(save_folder)

