import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_graph_strokes(graph):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    node_features = graph['stroke'].x


    node_features = node_features.numpy()

    for stroke in node_features:
        x1, y1, z1, x2, y2, z2 = stroke
        ax.plot([x1, x2], [y1, y2], [z1, z2], marker='o')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    plt.title('3D Graph Strokes Plot')
    plt.show()
