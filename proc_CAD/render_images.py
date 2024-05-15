import brep_read

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot(edges_features):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for edge_info in edges_features:
        if not edge_info['is_curve']:
            xs, ys, zs = zip(*edge_info['vertices'])
            ax.plot(xs, ys, zs, marker='o', color='r', label='Vertices')  

        if edge_info['is_curve'] and 'sampled_points' in edge_info:
            xp, yp, zp = zip(*edge_info['sampled_points'])
            ax.plot(xp, yp, zp, linestyle='--', color='b', label='Sampled Points')
            
    # Setting labels
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    # Show plot
    plt.show()


edges_features = brep_read.create_graph_from_step_file('./canvas/step_5.stp')
plot(edges_features)