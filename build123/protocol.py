from pathlib import Path
from build123d import *
import os


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


def build_sketch(Points_list):
    brep_dir = os.path.join(home_dir,  "canvas", "brep")
    stl_dir = os.path.join(home_dir,  "canvas", "vis.stl")

    with BuildSketch():
        with BuildLine():
            lines = []
            for i in range(0, len(Points_list), 2):
                start_point_sublist = Points_list[i]
                end_point_sublist = Points_list[i+1]
                start_point = (start_point_sublist[0], start_point_sublist[2], start_point_sublist[1])
                end_point = (end_point_sublist[0], end_point_sublist[2], end_point_sublist[1])

                line = Line(start_point, end_point)
                lines.append(line)

        perimeter = make_face()

    perimeter.export_brep(brep_dir)

    perimeter.export_stl(stl_dir)