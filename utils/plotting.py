import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np

def plot_3d_graph_strokes(graph):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Access the node features and labels from the graph
    node_features = graph['stroke'].x
    node_labels = graph['stroke'].y

    if node_features.is_cuda:
        node_features = node_features.cpu()
    if node_labels.is_cuda:
        node_labels = node_labels.cpu()

    node_features = node_features.numpy()
    node_labels = node_labels.numpy()

    # Unique labels and color map
    labels = np.unique(node_labels)
    colors = plt.cm.get_cmap('tab20', len(labels))

    # Map each label to a color
    label_color_map = {label: colors(i) for i, label in enumerate(labels)}

    # Loop through the node features and labels
    for stroke, label in zip(node_features, node_labels):
        x1, y1, z1, x2, y2, z2 = stroke
        color = label_color_map[label]
        ax.plot([x1, x2], [y1, y2], [z1, z2], marker='o', color=color)

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    plt.title('3D Graph Strokes Plot by Label')
    plt.show()



def plot_3d_graph_strokes(graph, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Access the stroke data from the graph
    stroke_data = graph['stroke'].x

    # Check if the data is on a GPU and move it to CPU if necessary
    if stroke_data.is_cuda:
        stroke_data = stroke_data.cpu()
    
    # Convert stroke data to numpy array
    stroke_data = stroke_data.numpy()

    # Check if labels are on a GPU and move to CPU if necessary, and convert to numpy array
    if labels.is_cuda:
        labels = labels.cpu()
    labels = labels.numpy()
    
    # Define colors for the strokes based on label values
    colors = ['red' if label > 0.8 else 'black' for label in labels]

    # Loop through the stroke data and plot each stroke with its corresponding color
    for stroke, color in zip(stroke_data, colors):
        x1, y1, z1, x2, y2, z2 = stroke
        ax.plot([x1, x2], [y1, y2], [z1, z2], marker='o', color=color)

    for stroke, color in zip(stroke_data, colors):
        if color == 'red':
            x1, y1, z1, x2, y2, z2 = stroke
            ax.plot([x1, x2], [y1, y2], [z1, z2], marker='o', color=color)


    # Setting labels for the axes
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    plt.title('3D Graph of Strokes')
    plt.show()
