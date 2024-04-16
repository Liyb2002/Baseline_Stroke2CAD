
from build123d import *


with BuildPart() as example:
    Cylinder(radius=10, height=3)

example.part.export_brep("brep")

example.part.export_stl("vis.stl")

