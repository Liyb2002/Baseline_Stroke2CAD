import brep_read

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

edges_features = brep_read.create_graph_from_step_file('./canvas/step_5.stp')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for edge in edges_features:

    x_coords = [edge[0], edge[3]]
    y_coords = [edge[1], edge[4]]
    z_coords = [edge[2], edge[5]]

    # Plot each edge
    # print("x_coords", x_coords, "y_coords", y_coords, "z_coords", z_coords)
    ax.plot(x_coords, y_coords, z_coords, 'b-')  # 'b-' specifies a blue line

# Setting labels
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')

# Show plot
plt.show()
