from pathlib import Path
from build123d import *
import os
import numpy as np

import build123.helper

home_dir = Path(__file__).parent.parent


def build_sketch(count, Points_list):
    brep_dir = os.path.join(home_dir, "canvas", f"brep_{count}.stp")
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


                line = Line(start_point, end_point)
                lines.append(line)

        perimeter = make_face()

    perimeter.export_stl(stl_dir)
    perimeter.export_brep(brep_dir)

    return perimeter


def build_extrude(count, canvas, target_face, extrude_amount, is_Add):
    stl_dir = os.path.join(home_dir, "canvas", f"vis_{count}.stl")
    step_dir = os.path.join(home_dir, "canvas", f"step_{count}.stp")

    
    if canvas != None:
        with canvas: 
            if is_Add >= 0:
                extrude( target_face, amount=extrude_amount)
            else:
                # extrude( target_face, amount=extrude_amount)

                extrude( target_face, amount=-extrude_amount, mode=Mode.SUBTRACT)

    else:
        with BuildPart() as canvas:
            if is_Add >= 0:
                extrude( target_face, amount=extrude_amount)
            else:
                # extrude( target_face, amount=extrude_amount)
                extrude( target_face, amount=-extrude_amount, mode=Mode.SUBTRACT)


    canvas.part.export_stl(stl_dir)
    canvas.part.export_step(step_dir)

    return canvas