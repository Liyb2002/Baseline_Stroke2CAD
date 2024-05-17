import brep_read
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import helper

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

    bounding_box_edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face edges
        (4, 5), (5, 7), (7, 6), (6, 4),  # Top face edges
        (0, 4), (1, 5), (2, 6), (3, 7)   # Side edges connecting bottom and top faces
        ]

    for edge in bounding_box_edges:
        edge_feature = {
            'vertices': [bounding_box_vertices[edge[0]], bounding_box_vertices[edge[1]]],
            'type': 'scaffold',
            'is_curve': False,
            'sampled_points': [],
            'projected_edge': [],
            'sigma' : 0.0,
            'mu': 0.0
        }
        edges_features.append(edge_feature)


    object_center = [(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2]

    return edges_features, object_center

def plot(edges_features):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for edge_info in edges_features:
        if not edge_info['is_curve']:
            xs, ys, zs = zip(*edge_info['vertices'])
            if edge_info['type'] == 'scaffold':
                ax.plot(xs, ys, zs, marker='o', color='green', label='Construction Line')
            else:
                ax.plot(xs, ys, zs, marker='o', color='red', label='Vertices')  

        # Plot curved edges with sampled points if they exist
        if edge_info['is_curve'] and edge_info['sampled_points']:
            xp, yp, zp = zip(*edge_info['sampled_points'])
            ax.plot(xp, yp, zp, linestyle='--', color='blue', label='Sampled Points')


    # Setting labels
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    # Show plot
    plt.show()

def optimize_opacities(edges_features, stylesheet):
    for edge_info in edges_features:
        edge_type = edge_info['type']

        if edge_type == 'scaffold':
            edge_info['mu'] = stylesheet["opacities_per_type"]["scaffold"]["mu"]
            edge_info['sigma'] = stylesheet["opacities_per_type"]["scaffold"]["sigma"]
            
        if edge_type == 'feature_line':
            edge_info['mu'] = stylesheet["opacities_per_type"]["vis_edges"]["mu"]
            edge_info['sigma'] = stylesheet["opacities_per_type"]["vis_edges"]["sigma"]

    
    return edges_features

def project_points(edges_features, obj_center):
    helper.project_points(edges_features, obj_center)



# Load styles
stroke_dataset_designer_name = 'Professional1'

opacity_profiles_name = os.path.join("styles/opacity_profiles", stroke_dataset_designer_name+".json")
if os.path.exists(opacity_profiles_name):
    with open(opacity_profiles_name, "r") as fp:
        opacity_profiles = json.load(fp)

style_sheet_file_name = os.path.join("styles/stylesheets/"+stroke_dataset_designer_name+".json")
if os.path.exists(style_sheet_file_name):
    with open(style_sheet_file_name, "r") as fp:
        stylesheet = json.load(fp)




edges_features = brep_read.create_graph_from_step_file('./canvas/step_5.stp')
edges_features, obj_center= find_bounding_box(edges_features)
edges_features = optimize_opacities(edges_features, stylesheet)
project_points(edges_features, obj_center)
# plot(edges_features)