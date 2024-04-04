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
    unique = []
    for p in points:
        if sum((p == point).all() for point in points) == 1:
            unique.append(p)
    return unique

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

