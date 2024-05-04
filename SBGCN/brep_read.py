from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopoDS import TopoDS_Shape, topods
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRepTools import breptools
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop

from torch.utils.data import Dataset

import os
from tqdm import tqdm
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
    return {"uv_bounds": (u_min, u_max, v_min, v_max)}

def create_edge_node(edge):
    properties = GProp_GProps()
    brepgprop.LinearProperties(edge, properties)
    length = properties.Mass()
    
    edge_start = BRep_Tool.Pnt(topods.Vertex(TopExp_Explorer(edge, TopAbs_VERTEX).Current()))
    edge_end = BRep_Tool.Pnt(topods.Vertex(TopExp_Explorer(edge, TopAbs_VERTEX, True).Current()))
    
    return {"length": length, "start_point": (edge_start.X(), edge_start.Y(), edge_start.Z()), "end_point": (edge_end.X(), edge_end.Y(), edge_end.Z())}

def create_vertex_node(vertex):
    pt = BRep_Tool.Pnt(vertex)
    return {"coordinates": (pt.X(), pt.Y(), pt.Z())}

def create_graph_from_step_file(step_path):
    shape = read_step_file(step_path)

    graph = SBGCN_graph.HeteroGraph()

    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = topods.Face(face_explorer.Current())
        face_features=create_face_node(face)
        face_node_id = graph.avoid_duplicate("face", face_features)

        if face_node_id is None:
            face_id = len(graph.nodes)
            graph.add_node(face_id, "face", features=face_features)
        

        # Explore edges of the face
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while edge_explorer.More():
            edge = topods.Edge(edge_explorer.Current())
            edge_features=create_edge_node(edge)
            edge_node_id = graph.avoid_duplicate("edge", edge_features)

            if edge_node_id is None:
                edge_id = len(graph.nodes) 
                graph.add_node(edge_id, "edge", features=create_edge_node(edge))
                graph.add_edge(face_id, edge_id, "has_edge")
            
            # Explore vertices of the edge
                vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
                while vertex_explorer.More():
                    vertex = topods.Vertex(vertex_explorer.Current())
                    vertex_features=create_edge_node(edge)
                    vertex_node_id = graph.avoid_duplicate("vertex", vertex_features)

                    if vertex_node_id is None:
                        vertex_id = len(graph.nodes)
                        graph.add_node(vertex_id, "vertex", features=create_vertex_node(vertex))
                        graph.add_edge(edge_id, vertex_id, "has_vertex")
                    
                    vertex_explorer.Next()
            
            edge_explorer.Next()
        
        face_explorer.Next()
    
    return graph



class BRep_Dataset(Dataset):
    def __init__(self, data_path, num_graphs = 32):
        self.data_path = data_path
        self.graphs = []

        graph = create_graph_from_step_file(self.data_path)

        node_counts = graph.count_nodes_by_type()
        for node_type, count in node_counts.items():
            print(f"Number of {node_type} nodes: {count}")
            graph.print_features_for_node_type(node_type)

        
        for i in range(num_graphs):
            self.graphs.append(graph)


    def __len__(self):
        return len(self.CAD_stroke_pairs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]

        return graph
    
