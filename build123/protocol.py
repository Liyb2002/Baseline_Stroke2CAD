
from build123d import *


with BuildPart() as example:
    Cylinder(radius=10, height=3)

example.part.export_brep("to_export")


example.part.export_stl("stl")

