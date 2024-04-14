from torch.utils.data import Dataset
import os
from tqdm import tqdm
from preprocessing.io_utils import read_json_file
import preprocessing.gnn_graph

class Stroke_Cloud_Dataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.CAD_stroke_pairs = self.get_files(data_path)
        print("loaded CAD_stroke_pairs: ", len(self.CAD_stroke_pairs))
        self.getGraph = False

    def __len__(self):
        return len(self.CAD_stroke_pairs)

    def returnGraph(self, getGraph = True):
        self.getGraph = True

    def __getitem__(self, idx, getGraph = False):
        item = self.CAD_stroke_pairs[idx]
        CAD_Program_path = item['CAD_Program']
        final_edges_path = item['final_edges']
        strokes_dict_path = item['strokes_dict']

        if self.getGraph:
            graph = preprocessing.gnn_graph.create_graph_from_json(final_edges_path, CAD_Program_path, strokes_dict_path)
            return graph


        return CAD_Program_path, final_edges_path, strokes_dict_path
    


    def get_files(self, data_path):
        CAD_stroke_pairs = []

        sub_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        sub_folders = sub_folders[:1]
        for sub_folder in tqdm(sub_folders):
            sub_folder_path = os.path.join(data_path, sub_folder)
            CAD_path = os.path.join(sub_folder_path, 'parsed_features.json')
            if not os.path.exists(CAD_path):
                    continue

            stroke_folders = [d for d in os.listdir(sub_folder_path) if os.path.isdir(os.path.join(sub_folder_path, d))]
            for stroke_folder in stroke_folders:
                final_edges_path = os.path.join(sub_folder_path, stroke_folder, 'final_edges.json')
                strokes_dict_path = os.path.join(sub_folder_path, stroke_folder, 'strokes_dict.json')
                if os.path.exists(final_edges_path) and os.path.exists(strokes_dict_path):
                    # final_edges = read_json_file(final_edges_path)
                    CAD_stroke_pairs.append({'CAD_Program': CAD_path, 'final_edges': final_edges_path, 'strokes_dict': strokes_dict_path})
                
        return CAD_stroke_pairs
