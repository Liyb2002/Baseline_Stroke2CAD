from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopoDS import TopoDS_Shape, topods
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRepTools import breptools
from OCC.Core.Geom import Geom_Surface
from OCC.Core.BRep import BRep_Tool

def read_step_file(filename):
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(filename)
    
    if status == 1:
        step_reader.TransferRoot()
        shape = step_reader.Shape()  
        return shape
    else:
        raise Exception("Error reading STEP file.")

def print_faces(shape):
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_id = 0
    while explorer.More():
        face = topods.Face(explorer.Current())
        face_id += 1
        surface = BRep_Tool.Surface(face)
        u_min, u_max, v_min, v_max = breptools.UVBounds(face)
        print(f"Face {face_id}:")
        print(f"  Surface Type: {surface.DynamicType().Name()}")
        print(f"  UV Bounds: U Min = {u_min}, U Max = {u_max}, V Min = {v_min}, V Max = {v_max}")
        print("---------------------------")
        explorer.Next()

def main():
    # Path to your STEP file
    step_path = './canvas/step_4.step'

    # Load the STEP file
    shape = read_step_file(step_path)

    # Print out details of all faces
    print_faces(shape)

if __name__ == "__main__":
    main()
