import numpy as np

def cross_product(vec1, vec2):
    return np.cross(vec1, vec2)

def is_coplanar(vec1, vec2, vec3):
    return np.dot(vec1, cross_product(vec2, vec3)) == 0

def get_vector(point1, point2):
    return np.array(point2) - np.array(point1)


def find_planes(stroke_indices, stroke_objects, raw_connectivity_matrix):
    planes = []
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
                for k in range(len(stroke_indices)):
                    if k != i and k != j:
                        line3 = [stroke_objects[k].point0, stroke_objects[k].point1]
                        if raw_connectivity_matrix[i,k] == 1 or raw_connectivity_matrix[j,k] == 1:
                            vec3 = get_vector(*line3)
                            if is_coplanar(vec1, vec2, vec3):
                                planes.append((line1, line2, line3))
    return planes

