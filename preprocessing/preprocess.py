import json
import os

from preprocessing.io_utils import home_dir
from preprocessing.dataset import StrokeDataset


def get_stroke_dataset():
    data_filepath = home_dir / "dataset"
    sample_chaeck_folder = home_dir /"output"/"sampled_check_data"

    stroke_dataset = StrokeDataset(data_filepath)
    stroke_dataset.save_sample(sample_chaeck_folder)
    
    return stroke_dataset

