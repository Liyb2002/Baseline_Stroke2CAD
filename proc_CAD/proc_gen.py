import sketch
import numpy as np

# Example usage:
axis = np.random.choice(['x', 'y', 'z'])
rectangle_points = sketch.generate_random_rectangle(axis)
sketch.save_points_to_json(rectangle_points, index=0)


