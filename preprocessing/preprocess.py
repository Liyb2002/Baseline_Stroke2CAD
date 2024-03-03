import json
import os

from dataset import StrokeDataset

data_path = '../dataset'

stroke_DS = StrokeDataset(data_path)
print("len",len(stroke_DS))