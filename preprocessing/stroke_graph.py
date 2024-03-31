import json
import numpy as np


def build_connectivity_matrix(strokes_dict_path):

    with open(strokes_dict_path[0], 'r') as file:
        data = json.load(file)
    stroke_ids = set()

    for stroke in data:
        stroke_ids.add(stroke['id'])
        flattened_stroke_intersections = [item for sublist in stroke['intersections'] for item in sublist]
        stroke_ids.update(flattened_stroke_intersections)

    n = len(stroke_ids)
    connectivity_matrix = np.zeros((n, n), dtype=int)
    id_to_index = {stroke_id: index for index, stroke_id in enumerate(sorted(stroke_ids))}

    for stroke in data:
        stroke_id = stroke['id']
        stroke_index = id_to_index[stroke_id]
        flattened_stroke_intersections = [item for sublist in stroke['intersections'] for item in sublist]
        for intersected_id in flattened_stroke_intersections:
            intersected_index = id_to_index[intersected_id]
            connectivity_matrix[stroke_index, intersected_index] = 1
            connectivity_matrix[intersected_index, stroke_index] = 1

    print("good connectivity_matrix", connectivity_matrix.shape)
    return connectivity_matrix



