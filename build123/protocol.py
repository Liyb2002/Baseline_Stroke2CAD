from pathlib import Path
from build123d import *
import os


home_dir = Path(__file__).parent.parent

def build_sketch():
    brep_dir = os.path.join(home_dir,  "canvas", "brep")
    stl_dir = os.path.join(home_dir,  "canvas", "vis.stl")

    # with BuildPart() as example:
    with BuildSketch() as plan:
        perimeter = Rectangle(10, 20)

    perimeter.export_brep(brep_dir)

    perimeter.export_stl(stl_dir)