import json
import numpy as np
import torch
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


def build_gt_label(entity_info, stroke_objects):
    labels = torch.zeros((len(stroke_objects), 1), dtype=torch.float32)

    for edge in entity_info['edges']:
        edge_id = edge['edge_id']
        edge_type = edge['edge_type']
        edge_direction = edge['edge_direction']
        edge_origin = edge['edge_origin']

        if edge_type != 'Line':
            print("edge_type", edge_type)
            continue

        gt_line = [edge_origin, edge_direction]

        for i, stroke in enumerate(stroke_objects):
            if stroke.type == 'straight_stroke':

                stroke_line = [stroke.point0, stroke.point1]

                if same_line(gt_line, stroke_line):
                    labels[i, 0] = 1
                
    return labels
            

def distance(point1, point2):
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
    return dist


def same_line(gt_line, target_line, threshold = 0.001):

    #check if we have a correct point
    correct_point = False
    if distance(gt_line[0], target_line[0]) < threshold or distance(gt_line[0], target_line[1]) < threshold:
        correct_point = True
    
    #check if we have a correct direction
    correct_dir = False
    dir_target_line = np.array([x - y for x,y in zip(target_line[0], target_line[1])])
    normalized_dir_target_line = np.round(dir_target_line / np.linalg.norm(dir_target_line), 3)
    invert_normalized_dir_target_line = [-x for x in normalized_dir_target_line]

    if np.allclose(gt_line[1], normalized_dir_target_line) or np.allclose(gt_line[1], invert_normalized_dir_target_line):
        correct_dir = True

    return correct_point and correct_dir


def plot_3D(lines):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for line in lines:
        x_coords = [point[0] for point in line]
        y_coords = [point[1] for point in line]
        z_coords = [point[2] for point in line]

        ax.plot(x_coords, y_coords, z_coords, marker='o')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()

