from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopoDS import TopoDS_Shape, topods
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRepTools import breptools
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop

import SBGCN_graph

def read_step_file(filename):
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(filename)
    
    if status == 1:  # Check if the read was successful
        step_reader.TransferRoot()  # Transfers the whole STEP file
        shape = step_reader.Shape()  # Retrieves the translated shape
        return shape
    else:
        raise Exception("Error reading STEP file.")

def create_face_node(face):
    u_min, u_max, v_min, v_max = breptools.UVBounds(face)
    print(f"UV Bounds: U Min = {u_min}, U Max = {u_max}, V Min = {v_min}, V Max = {v_max}")

def create_edge_node(face):
    edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
    edge_id = len(graph.nodes)
    while edge_explorer.More():
        edge = topods.Edge(edge_explorer.Current())
        edge_id += 1
        # Calculate the length of the edge using the new static method
        properties = GProp_GProps()
        brepgprop.LinearProperties(edge, properties)
        length = properties.Mass()
        print(f"  Edge {edge_id}: Length = {length:.5f}")
        create_vertex_node(edge)
        edge_explorer.Next()

def create_vertex_node(edge):
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

def create_graph_from_step_file(step_path):
    shape = read_step_file(step_path)

    graph = SBGCN_graph.HeteroGraph()

    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = topods.Face(face_explorer.Current())
        face_id = len(graph.nodes)  # Generate a unique ID for the face node
        graph.add_node(face_id, "face")
        
        # Explore edges of the face
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while edge_explorer.More():
            edge = topods.Edge(edge_explorer.Current())
            edge_id = len(graph.nodes)  # Generate a unique ID for the edge node
            graph.add_node(edge_id, "edge")
            graph.add_edge(face_id, edge_id, "has_edge")
            
            # Explore vertices of the edge
            vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
            while vertex_explorer.More():
                vertex = topods.Vertex(vertex_explorer.Current())
                vertex_id = len(graph.nodes)  # Generate a unique ID for the vertex node
                graph.add_node(vertex_id, "vertex")
                graph.add_edge(edge_id, vertex_id, "has_vertex")
                vertex_explorer.Next()
            
            edge_explorer.Next()
        
        face_explorer.Next()
    
    return graph


def run():
    step_path = '../preprocessing/canvas/step_4.step'

    graph = create_graph_from_step_file(step_path)

    node_counts = graph.count_nodes_by_type()
    for node_type, count in node_counts.items():
        print(f"Number of {node_type} nodes: {count}")

run()