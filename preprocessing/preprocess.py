import json
import os
from pathlib import Path

from preprocessing.dataset import StrokeDataset


def get_stroke_dataset():
    data_filepath = Path(__file__).parent.parent / "dataset"
    sample_chaeck_folder = Path(__file__).parent.parent /"output"/"sampled_check_data"

    stroke_dataset = StrokeDataset(data_filepath)
    stroke_dataset.save_sample(sample_chaeck_folder)
    
    return stroke_dataset

