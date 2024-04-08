import json
import numpy as np
import torch
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def build_connectivity_matrix(batch_strokes_dict_path, batch_stroke_objects):
    connectivity_matrices = []
    raw_connectivity_matrices = []
    plane_dicts = []


    for strokes_dict_path, stroke_objects in zip(batch_strokes_dict_path, batch_stroke_objects):
        with open(strokes_dict_path, 'r') as file:
            data = json.load(file)
        
        ordered_stroke_ids = [stroke_obj.line_id for stroke_obj in stroke_objects]
        stroke_id_set = set(ordered_stroke_ids)

        id_to_index = {stroke_id: index for index, stroke_id in enumerate(ordered_stroke_ids)}

        # for line_id, index in id_to_index.items():
        #     print(f"{line_id}: {index}")


        n = len(ordered_stroke_ids)
        raw_connectivity_matrix = np.zeros((n, n), dtype=int)
        for stroke in data:
            if stroke['id'] in stroke_id_set:
                stroke_index = id_to_index[stroke['id']]
                for sublist in stroke['intersections']:
                    for intersected_id in sublist:
                        if intersected_id in stroke_id_set:
                            intersected_index = id_to_index[intersected_id]
                            raw_connectivity_matrix[stroke_index, intersected_index] = 1
                            raw_connectivity_matrix[intersected_index, stroke_index] = 1

        connectivity_matrix = adjacency_matrix_to_edge_index(raw_connectivity_matrix)

        plane_dict = build_plane_dict(stroke_id_set, id_to_index, data)

        # plot_plane_dict(plane_dict, stroke_objects)
        connectivity_matrices.append(connectivity_matrix)
        raw_connectivity_matrices.append(raw_connectivity_matrix)
        plane_dicts.append(plane_dict)

    return connectivity_matrices, raw_connectivity_matrices, plane_dicts


def adjacency_matrix_to_edge_index(connectivity_matrix):
    #just a format transforming function, nothing important
    edge_index = []
    for i in range(connectivity_matrix.shape[0]):
        for j in range(connectivity_matrix.shape[1]):
            if connectivity_matrix[i, j] != 0:
                edge_index.append([i, j])
    return torch.tensor(edge_index).t().contiguous()


def build_plane_dict(stroke_id_set, id_to_index, strokes_dict):
    plane_dict = {}
    for stroke in strokes_dict:
        plane_ids = stroke['planes']
        stroke_id = stroke['id']

        if stroke_id in stroke_id_set: 
            for plane_id in plane_ids:
                if plane_id not in plane_dict:
                    plane_dict[plane_id] = []
                plane_dict[plane_id].append(stroke_id)

    
    for plane_id in plane_dict:
        plane_dict[plane_id] = [id_to_index[stroke_id] for stroke_id in plane_dict[plane_id] if stroke_id in id_to_index]

    return plane_dict
        

def plot_plane_dict(plane_dict, stroke_objects):
    for plane_id in plane_dict:
        lines = []
        strokes_indices = plane_dict[plane_id]
        for index in strokes_indices:
            if stroke_objects[index].type == 'straight_stroke':
                line = [stroke_objects[index].point0, stroke_objects[index].point1]
                lines.append(line)
        plot_3D(lines)


def build_gt_label(entity_info, stroke_objects):
    labels = torch.zeros((len(stroke_objects), 1), dtype=torch.float32)

    for edge in entity_info['edges']:
        if edge['edge_type'] != 'Line':
            print("edge_type", edge['edge_type'])
            continue

        gt_line = [edge['edge_origin'], edge['edge_direction']]

        for i, stroke in enumerate(stroke_objects):
            if stroke.type == 'straight_stroke':
                stroke_line = [stroke.point0, stroke.point1]
                if same_line(gt_line, stroke_line):
                    labels[i, 0] = 1

    target_id = torch.nonzero(labels == 1)
    # print("target_id", target_id)

    return labels


def build_gt_label_from_ID(ID, batch_stroke_objects):

    batch_labels = []
    for stroke_objects in batch_stroke_objects:
        labels = torch.zeros((len(stroke_objects), 1), dtype=torch.float32)

        for i, stroke in enumerate(stroke_objects):
            if stroke.feature_id == ID:
                labels[i, 0] = 1
        
        batch_labels.append(labels)

    return batch_labels

def distance(point1, point2):
    point1, point2 = np.array(point1), np.array(point2)
    return np.sqrt(np.sum((point1 - point2) ** 2))


def same_line(gt_line, target_line, threshold=0.001):
    gt_line = [point.cpu().numpy() if isinstance(point, torch.Tensor) else np.array(point) for point in gt_line]
    target_line = [point.cpu().numpy() if isinstance(point, torch.Tensor) else np.array(point) for point in target_line]

    # Check if there is a correct point
    if distance(gt_line[0], target_line[0]) < threshold or distance(gt_line[0], target_line[1]) < threshold:

        # Check if there is a correct direction

        dir_gt_line = np.round((gt_line[1]) / np.linalg.norm(gt_line[1]), 3)
        dir_target_line = np.round((target_line[1] - target_line[0]) / np.linalg.norm(target_line[1] - target_line[0]), 3)
        if np.allclose(dir_gt_line, dir_target_line) or np.allclose(dir_gt_line, -dir_target_line):
            return True
    return False


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

