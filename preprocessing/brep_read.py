from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopoDS import TopoDS_Shape, topods
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRepTools import breptools
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop

def read_step_file(filename):
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(filename)
    
    if status == 1:  # Check if the read was successful
        step_reader.TransferRoot()  # Transfers the whole STEP file
        shape = step_reader.Shape()  # Retrieves the translated shape
        return shape
    else:
        raise Exception("Error reading STEP file.")

def print_face_details(face):
    u_min, u_max, v_min, v_max = breptools.UVBounds(face)
    print(f"UV Bounds: U Min = {u_min}, U Max = {u_max}, V Min = {v_min}, V Max = {v_max}")

def print_edge_details(face):
    edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
    edge_id = 0
    while edge_explorer.More():
        edge = topods.Edge(edge_explorer.Current())
        edge_id += 1
        # Calculate the length of the edge using the new static method
        properties = GProp_GProps()
        brepgprop.LinearProperties(edge, properties)
        length = properties.Mass()
        print(f"  Edge {edge_id}: Length = {length:.5f}")
        print_vertex_details(edge)
        edge_explorer.Next()

def print_vertex_details(edge):
    vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
    vertices = []
    vertex_id = 0
    while vertex_explorer.More():
        vertex = topods.Vertex(vertex_explorer.Current())
        vertex_id += 1
        pt = BRep_Tool.Pnt(vertex)
        vertices.append((pt.X(), pt.Y(), pt.Z()))
        print(f"Vertex {vertex_id}: Coordinates = ({pt.X():.5f}, {pt.Y():.5f}, {pt.Z():.5f})")
        vertex_explorer.Next()
    return vertices

def main():
    # Path to your STEP file
    step_path = './canvas/step_4.step'

    # Load the STEP file
    shape = read_step_file(step_path)

    # Explore and print out details of all faces and their edges
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_id = 0
    while face_explorer.More():
        face = topods.Face(face_explorer.Current())
        face_id += 1
        print(f"Face {face_id}:")
        print_face_details(face)
        print_edge_details(face)
        print("-----------------------")
        face_explorer.Next()

if __name__ == "__main__":
    main()
