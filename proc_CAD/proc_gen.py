import random_gen
import numpy as np
import brep_class

canvas_class = brep_class.Brep()

# Example usage:
axis = np.random.choice(['x', 'y', 'z'])
rectangle_points, normal = random_gen.generate_random_rectangle(axis)

extrude_amount = random_gen.generate_random_extrude_add()
canvas_class.add_sketch_op(rectangle_points, normal)
canvas_class.add_extrude_add_op(extrude_amount)
canvas_class.write_to_json()

