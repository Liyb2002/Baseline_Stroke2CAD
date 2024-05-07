import sketch
import numpy as np
import brep_class

canvas_class = brep_class.Brep()

# Example usage:
axis = np.random.choice(['x', 'y', 'z'])
rectangle_points, normal = sketch.generate_random_rectangle(axis)
canvas_class.add_sketch_op(rectangle_points, normal)
canvas_class.write_to_json()

