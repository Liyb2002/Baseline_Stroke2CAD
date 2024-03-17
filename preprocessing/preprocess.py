import json
import os

from preprocessing.io_utils import home_dir
from preprocessing.dataset import StrokeDataset
from preprocessing.stroke_cloud import Stroke_Cloud_Dataset
from preprocessing.opensketch_dataset import opensketch_dataset

def get_stroke_dataset():
    data_filepath = home_dir / "dataset" / "CAD2Sketch"
    sample_chaeck_folder = home_dir /"output"/"sampled_check_data"
    openSketch_data_filepath = home_dir / "dataset" / "sketches_json_first_viewpoint"

    stroke_dataset = StrokeDataset(data_filepath)
    stroke_dataset.save_sample(sample_chaeck_folder)

    # os_dataset = opensketch_dataset(openSketch_data_filepath)
    # os_dataset.save_sample(sample_chaeck_folder)

    print("number of elements", len(stroke_dataset))

    return stroke_dataset


def get_stroke_cloud():
    data_filepath = home_dir / "dataset" / "CAD2Sketch"
    stroke_cloud_dataset = Stroke_Cloud_Dataset(data_filepath)

    return stroke_cloud_dataset

