import numpy as np
import preprocessing.stroke_graph

def cross_product(vec1, vec2):
    return np.cross(vec1, vec2)

def is_coplanar(vec1, vec2, vec3):
    return np.dot(vec1, cross_product(vec2, vec3)) == 0

def get_vector(point1, point2):
    return np.array(point2) - np.array(point1)

def check_line_in_plane(plane, line):
    vec1, vec2 = plane[:2]
    vec3 = get_vector(*line)
    return is_coplanar(get_vector(*vec1), get_vector(*vec2), vec3)

def find_unique_points(points):

    unique_tuples = set(tuple(point) for point in points)
    unique_points = [list(point) for point in unique_tuples]

    return unique_points


def tensor_to_tuple(tensor):
    return tuple(tensor.tolist())

def find_planes(stroke_indices, stroke_objects, raw_connectivity_matrix):

    planes = []
    plane_stroke_ids = []

    for i in range(len(stroke_indices)):

        if stroke_objects[i].type == 'curve_stroke':
            continue

        for j in range(i + 1, len(stroke_indices)):
            if stroke_objects[j].type == 'curve_stroke':
                continue

            line1 = [stroke_objects[i].point0, stroke_objects[i].point1]
            line2 = [stroke_objects[j].point0, stroke_objects[j].point1]

            # Check if lines intersect
            if raw_connectivity_matrix[i,j] == 1:
                vec1 = get_vector(*line1)
                vec2 = get_vector(*line2)

                # Find a third line to check for coplanarity
                for k in range(j+1, len(stroke_indices)):
                    if stroke_objects[k].type != 'curve_stroke':
                        line3 = [stroke_objects[k].point0, stroke_objects[k].point1]
                        if raw_connectivity_matrix[i,k] == 1 or raw_connectivity_matrix[j,k] == 1:
                            vec3 = get_vector(*line3)
                            if is_coplanar(vec1, vec2, vec3):

                                points = [
                                stroke_objects[i].point0, stroke_objects[i].point1,
                                stroke_objects[j].point0, stroke_objects[j].point1,
                                stroke_objects[k].point0, stroke_objects[k].point1]

                                unique_points = find_unique_points(points)
                                if len(unique_points) == 0:
                                    planes.append(points)
                                    plane_stroke_ids.append([stroke_indices[i],stroke_indices[j],stroke_indices[k]])
                                    break

                                # find the 4th stroke
                                for l in range(k+1, len(stroke_indices)):
                                    if stroke_objects[l].type != 'curve_stroke':
                                        line4 = [tensor_to_tuple(stroke_objects[l].point0), tensor_to_tuple(stroke_objects[l].point1)]
                                        unique_tuples = [tensor_to_tuple(p) for p in unique_points]

                                        if set(line4) == set(unique_tuples):
                                            planes.append((
                                                [stroke_objects[i].point0, stroke_objects[i].point1],
                                                [stroke_objects[j].point0, stroke_objects[j].point1],
                                                [stroke_objects[k].point0, stroke_objects[k].point1],
                                                line4
                                            ))
                                            plane_stroke_ids.append([stroke_indices[i],stroke_indices[j],stroke_indices[k],stroke_indices[l]])
                                            break

    return planes, plane_stroke_ids

def find_planes_gnn(stroke_indices, graph):

    planes = []
    plane_stroke_ids = []
    node_features = graph['stroke'].x
    raw_connectivity_matrix = graph.connectivity_matrix

    for i in range(len(stroke_indices)):

        for j in range(i + 1, len(stroke_indices)):

            x1_i, y1_i, z1_i, x2_i, y2_i, z2_i = node_features[i]
            x1_j, y1_j, z1_j, x2_j, y2_j, z2_j = node_features[j]


            line1 = [[x1_i, y1_i, z1_i], [x2_i, y2_i, z2_i]]
            line2 = [[x1_j, y1_j, z1_j], [x2_j, y2_j, z2_j]]
            line1 = [[element.item() for element in sublist] for sublist in line1]
            line2 = [[element.item() for element in sublist] for sublist in line2]

            # Check if lines intersect
            if raw_connectivity_matrix[i,j] == 1:
                vec1 = get_vector(*line1)
                vec2 = get_vector(*line2)

                # Find a third line to check for coplanarity
                for k in range(j+1, len(stroke_indices)):
                    x1_k, y1_k, z1_k, x2_k, y2_k, z2_k = node_features[k]
                    line3 = [[x1_k, y1_k, z1_k], [x2_k, y2_k, z2_k]]
                    line3 = [[element.item() for element in sublist] for sublist in line3]

                    if raw_connectivity_matrix[i,k] == 1 or raw_connectivity_matrix[j,k] == 1:
                        vec3 = get_vector(*line3)
                        if is_coplanar(vec1, vec2, vec3):

                            points = [
                            line1[0], line1[1],
                            line2[0], line2[1],
                            line3[0], line3[1]]

                            unique_points = find_unique_points(points)
                            if len(unique_points) == 0:
                                planes.append(points)
                                plane_stroke_ids.append([stroke_indices[i],stroke_indices[j],stroke_indices[k]])
                                break
                            
                            unique_points_set = set(tuple(point) for point in unique_points)
                            # find the 4th stroke
                            for l in range(k+1, len(stroke_indices)):
                                x1_l, y1_l, z1_l, x2_l, y2_l, z2_l = node_features[l]
                                line4 = [[x1_l, y1_l, z1_l], [x2_l, y2_l, z2_l]]
                                line4 = [[element.item() for element in sublist] for sublist in line4]
                                line4_set = set(tuple(point) for point in line4)


                                if line4_set.issubset(unique_points_set):
                                    planes.append((
                                        line1[0], line1[1],
                                        line2[0], line2[1],
                                        line3[0], line3[1],
                                        line4[0], line4[1]
                                    ))
                                    plane_stroke_ids.append([stroke_indices[i],stroke_indices[j],stroke_indices[k],stroke_indices[l]])
                                    break

    return planes, plane_stroke_ids
