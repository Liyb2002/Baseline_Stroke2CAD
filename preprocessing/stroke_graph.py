import json
import numpy as np
import torch

def build_connectivity_matrix(strokes_dict_path, stroke_objects):

    with open(strokes_dict_path[0], 'r') as file:
        data = json.load(file)
    stroke_ids = set()
    
    ordered_stroke_ids = [stroke_obj.line_id for stroke_obj in stroke_objects]
    stroke_id_set = set(ordered_stroke_ids)

    id_to_index = {stroke_id: index for index, stroke_id in enumerate(ordered_stroke_ids)}


    n = len(ordered_stroke_ids)
    connectivity_matrix = np.zeros((n, n), dtype=int)
    for stroke in data:
        if stroke['id'] in stroke_id_set:
            stroke_index = id_to_index[stroke['id']]
            for sublist in stroke['intersections']:
                for intersected_id in sublist:
                    if intersected_id in stroke_id_set:
                        intersected_index = id_to_index[intersected_id]
                        connectivity_matrix[stroke_index, intersected_index] = 1
                        connectivity_matrix[intersected_index, stroke_index] = 1

    connectivity_matrix = adjacency_matrix_to_edge_index(connectivity_matrix)
    return connectivity_matrix


def adjacency_matrix_to_edge_index(connectivity_matrix):
    edge_index = []
    for i in range(connectivity_matrix.shape[0]):
        for j in range(connectivity_matrix.shape[1]):
            if connectivity_matrix[i, j] != 0:  # Assuming non-zero value indicates an edge
                edge_index.append([i, j])
    return torch.tensor(edge_index).t().contiguous()


