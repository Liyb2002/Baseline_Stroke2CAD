from pathlib import Path
from build123d import *
import os
import numpy as np

import build123.helper

home_dir = Path(__file__).parent.parent

def example_process():
    brep_dir = os.path.join(home_dir,  "canvas", "brep")
    stl_dir = os.path.join(home_dir,  "canvas", "vis.stl")

    with BuildPart() as example:
        with BuildSketch():
            with BuildLine():
                l1 = Line((0, 0, 0), (0,0,1))
                l2 = Line((0, 0, 1), (1,0,1))
                l3 = Line((1, 0, 1), (1,0,0))
                l4 = Line((1, 0, 0), (0,0,0
                                    ))
            perimeter = make_face()
        extrude(amount = 2)

    example.part.export_stl(stl_dir)

    # perimeter.export_brep(brep_dir)

    # perimeter.export_stl(stl_dir)


# def rotate_point(transform):
#     origin = np.array([transform['origin']['x'], transform['origin']['y'], transform['origin']['z']])
#     x_axis = np.array([transform['x_axis']['x'], transform['x_axis']['y'], transform['x_axis']['z']])
#     y_axis = np.array([transform['y_axis']['x'], transform['y_axis']['y'], transform['y_axis']['z']])
#     z_axis = np.array([transform['z_axis']['x'], transform['z_axis']['y'], transform['z_axis']['z']])
#     print("origin", origin)
#     print("x_axis", x_axis)


def build_sketch(count, canvas, Points_list, origin, whole_sketch_rotation, per_face_rotation):
    brep_dir = os.path.join(home_dir, "canvas", f"brep_{count}.json")
    stl_dir = os.path.join(home_dir, "canvas", f"vis_{count}.stl")

    with BuildSketch():
        with BuildLine():
            lines = []
            for i in range(0, len(Points_list), 2):
                start_point_sublist = Points_list[i]
                end_point_sublist = Points_list[i+1]
                start_point = (start_point_sublist[0] + origin[0], 
                               start_point_sublist[1] + origin[1], 
                               start_point_sublist[2] + origin[2])
                
                
                end_point = (end_point_sublist[0] + origin[0], 
                            end_point_sublist[1] + origin[1], 
                            end_point_sublist[2] + origin[2])


                start_point = build123.helper.rotate_point_singleX(start_point, per_face_rotation)
                end_point = build123.helper.rotate_point_singleX(end_point, per_face_rotation)

                # start_point = build123.helper.rotate_point(start_point, whole_sketch_rotation)
                # end_point = build123.helper.rotate_point(end_point, whole_sketch_rotation)
      
                line = Line(start_point, end_point)
                lines.append(line)

        perimeter = make_face()

    
    if canvas != None:
        updated_canvas = Compound(label="Assembly", children=(canvas, perimeter))

    else:
        updated_canvas = Compound(label="Assembly", children=[perimeter])


    updated_canvas.export_brep(brep_dir)

    updated_canvas.export_stl(stl_dir)

    return perimeter


def build_extrude(count, canvas, target_face, extrude_amount):
    brep_dir = os.path.join(home_dir, "canvas", f"brep_{count}.json")
    stl_dir = os.path.join(home_dir, "canvas", f"vis_{count}.stl")

    new_element = extrude( target_face, amount=extrude_amount)
    new_element.label = "Extruded Part"


    if canvas != None:
        updated_canvas = Compound(label="Assembly", children=(canvas, new_element))

    else:
        updated_canvas = Compound(label="Assembly", children=[new_element])


    updated_canvas.export_stl(stl_dir)
    updated_canvas.export_brep(brep_dir)

    return updated_canvas
