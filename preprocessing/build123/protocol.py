from pathlib import Path
from build123d import *
import os
import numpy as np

import build123.helper

home_dir = Path(__file__).parent.parent


def build_sketch(count, canvas, Points_list, 
                 face_translation, whole_sketch_translation, 
                 whole_sketch_rotation, per_face_rotation):
    brep_dir = os.path.join(home_dir, "canvas", f"brep_{count}")
    stl_dir = os.path.join(home_dir, "canvas", f"vis_{count}.stl")

    with BuildSketch():
        with BuildLine():
            lines = []
            for i in range(0, len(Points_list), 2):
                start_point_sublist = Points_list[i]
                end_point_sublist = Points_list[i+1]
                start_point = (start_point_sublist[0],
                               start_point_sublist[1], 
                               start_point_sublist[2])
                
                
                end_point = (end_point_sublist[0],
                            end_point_sublist[1], 
                            end_point_sublist[2])


                start_point = build123.helper.rotate_point_singleX(start_point, per_face_rotation)
                end_point = build123.helper.rotate_point_singleX(end_point, per_face_rotation)

                # start_point = build123.helper.translate_local(start_point, face_translation)
                # end_point = build123.helper.translate_local(end_point, face_translation)

                start_point = build123.helper.rotate_point(start_point, whole_sketch_rotation)
                end_point = build123.helper.rotate_point(end_point, whole_sketch_rotation)

                start_point = build123.helper.translate_global(start_point, whole_sketch_translation)
                end_point = build123.helper.translate_global(end_point, whole_sketch_translation)

                line = Line(start_point, end_point)
                lines.append(line)

        perimeter = make_face()

    return perimeter


def build_extrude(count, canvas, target_face, extrude_amount, is_Add):
    brep_dir = os.path.join(home_dir, "canvas", f"brep_{count}")
    stl_dir = os.path.join(home_dir, "canvas", f"vis_{count}.stl")
    step_dir = os.path.join(home_dir, "canvas", f"step_{count}.stp")

    
    if canvas != None:
        with canvas: 
            if is_Add >= 0:
                extrude( target_face, amount=extrude_amount)
            else:
                extrude( target_face, amount=extrude_amount)

                # extrude( target_face, amount=-extrude_amount, mode=Mode.SUBTRACT)

    else:
        with BuildPart() as canvas:
            if is_Add >= 0:
                extrude( target_face, amount=extrude_amount)
            else:
                extrude( target_face, amount=extrude_amount)
                # extrude( target_face, amount=-extrude_amount, mode=Mode.SUBTRACT)


    canvas.part.export_stl(stl_dir)
    canvas.part.export_brep(brep_dir)
    canvas.part.export_step(step_dir)

    return canvas
