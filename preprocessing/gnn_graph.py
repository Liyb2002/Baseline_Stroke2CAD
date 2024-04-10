import os
import torch
from io_utils import read_json_file

import os
import sys
import torch
import torch_geometric
from torch_geometric.data import Data, HeteroData
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from networkx.algorithms import community
import seaborn as sns


def create_graph_from_json(final_edges_path, parsed_features_path, stroke_dict_path):

    final_edges_json = read_json_file(final_edges_path)
    parsed_features_json = read_json_file(parsed_features_path)
    stroke_dict_json = read_json_file(stroke_dict_path)

    #parse parsed_features
    parsed_features_sequence = {}
    if parsed_features_json is not None:
        for dict_i in parsed_features_json['sequence']:
            parsed_features_sequence[dict_i['index']] = {'type': dict_i['type'],
                                        'node_indices': dict_i['index']} 

    #parse final_edges
    sketch_graph = None

    num_strokes = len(final_edges_json.keys())
    node_features = []
    node_labels = []
    connectivity_matrix = torch.zeros((num_strokes, num_strokes))

    #go through all the strokes
    ordered_stroke_ids = []
    for key in final_edges_json.keys():
        stroke = final_edges_json[key]
        geometry = stroke["geometry"]
        id = stroke["id"]
        ordered_stroke_ids.append(id)
        node_features.append(geometry[0] + geometry[len(geometry)-1])

        node_label = stroke["feature_id"]
        node_labels.append(node_label)

    #each node has feature as its start and ending point
    #each node has label as the feature_id

    #node_features has shape n_strokes x 6
    node_features = torch.tensor(node_features, dtype=torch.float32)

    #node_labels has shape n_strokes x _
    node_labels = torch.tensor(node_labels, dtype=torch.float32)


    #build connectivity matrix
    #connectivity_matrix has shape n_strokes x n_strokes
    stroke_id_set = set(ordered_stroke_ids)
    id_to_index = {stroke_id: index for index, stroke_id in enumerate(ordered_stroke_ids)}
    for stroke_dict_item in stroke_dict_json:
        if stroke_dict_item['id'] in stroke_id_set:
            stroke_index = id_to_index[stroke_dict_item['id']]
            for sublist in stroke_dict_item['intersections']:
                for intersected_id in sublist:
                    if intersected_id in stroke_id_set:
                        intersected_index = id_to_index[intersected_id]
                        connectivity_matrix[stroke_index, intersected_index] = 1
                        connectivity_matrix[intersected_index, stroke_index] = 1

    print("connectivity_matrix", connectivity_matrix)






stroke_dict_path = '/Users/yuanboli/Documents/GitHub/Baseline_Stroke2CAD/dataset/CAD2Sketch/1069/63.86_149.75_1.4/strokes_dict.json'
parsed_features_path = '/Users/yuanboli/Documents/GitHub/Baseline_Stroke2CAD/dataset/CAD2Sketch/1069/parsed_features.json'
final_edges_path = '/Users/yuanboli/Documents/GitHub/Baseline_Stroke2CAD/dataset/CAD2Sketch/1069/63.86_149.75_1.4/final_edges.json'

create_graph_from_json(final_edges_path, parsed_features_path, stroke_dict_path)