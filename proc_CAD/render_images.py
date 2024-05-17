import brep_read

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def find_bounding_box(edges_features):
    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
    max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')

    for edge_info in edges_features:
        points = edge_info['vertices']
        if edge_info['is_curve'] and edge_info['sampled_points']:
            points += edge_info['sampled_points']

        for x, y, z in points:
            if x < min_x: min_x = x
            if y < min_y: min_y = y
            if z < min_z: min_z = z
            if x > max_x: max_x = x
            if y > max_y: max_y = y
            if z > max_z: max_z = z

    bounding_box_vertices = [
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z]
    ]

    return bounding_box_vertices

def plot(edges_features, bounding_box_vertices):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for edge_info in edges_features:
        if not edge_info['is_curve']:
            xs, ys, zs = zip(*edge_info['vertices'])
            ax.plot(xs, ys, zs, marker='o', color='r', label='Vertices')  

        if edge_info['is_curve'] and 'sampled_points' in edge_info:
            xp, yp, zp = zip(*edge_info['sampled_points'])
            ax.plot(xp, yp, zp, linestyle='--', color='b', label='Sampled Points')


    box_edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face edges
            (4, 5), (5, 7), (7, 6), (6, 4),  # Top face edges
            (0, 4), (1, 5), (2, 6), (3, 7)   # Side edges connecting bottom and top faces
        ]

    for start, end in box_edges:
        xs, ys, zs = zip(bounding_box_vertices[start], bounding_box_vertices[end])
        ax.plot(xs, ys, zs, 'g--')

    # Setting labels
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    # Show plot
    plt.show()


edges_features = brep_read.create_graph_from_step_file('./canvas/step_5.stp')
bounding_box_vertices = find_bounding_box(edges_features)
plot(edges_features, bounding_box_vertices)