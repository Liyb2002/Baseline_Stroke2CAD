from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopoDS import TopoDS_Shape, topods
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRepTools import breptools
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop

from torch.utils.data import Dataset

import torch
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
    
    return { "start_point": (edge_start.X(), edge_start.Y(), edge_start.Z()), "end_point": (edge_end.X(), edge_end.Y(), edge_end.Z())}

def create_vertex_node(vertex):
    pt = BRep_Tool.Pnt(vertex)
    return {"coordinates": (pt.X(), pt.Y(), pt.Z())}


def check_duplicate(new_feature, feature_list, face = 0):
    for idx, existing_feature in feature_list:
        if existing_feature == new_feature:
            return idx
    
    return -1

def create_graph_from_step_file(step_path):
    shape = read_step_file(step_path)

    face_features_list = []
    edge_features_list = []
    vertex_features_list = []
    
    edge_index_face_edge_list = []
    edge_index_edge_vertex_list = []

    index_counter = 0
    index_to_type = {}

    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = topods.Face(face_explorer.Current())
        face_features = create_face_node(face)

        if check_duplicate(face_features, face_features_list, 1) != -1:
            face_explorer.Next()
            continue

        face_features_list.append((index_counter, face_features))
        current_face_counter = index_counter
        index_to_type[current_face_counter] = 'face'
        index_counter += 1


        # Explore edges of the face
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while edge_explorer.More():
            edge = topods.Edge(edge_explorer.Current())
            edge_features = create_edge_node(edge)

            edge_duplicate_id = check_duplicate(edge_features, edge_features_list)
            if edge_duplicate_id != -1:
                edge_index_face_edge_list.append([current_face_counter, edge_duplicate_id])
                edge_explorer.Next()
                continue
            
            edge_features_list.append((index_counter, edge_features))
            current_edge_counter = index_counter
            edge_index_face_edge_list.append([current_face_counter, current_edge_counter])
            index_to_type[current_edge_counter] = 'edge'
            index_counter += 1


            # Explore vertices of the edge
            vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
            while vertex_explorer.More():
                vertex = topods.Vertex(vertex_explorer.Current())
                vertex_features = create_vertex_node(vertex)


                vertex_duplicate_id = check_duplicate(vertex_features, vertex_features_list)
                if vertex_duplicate_id != -1:
                    edge_index_edge_vertex_list.append([current_edge_counter, vertex_duplicate_id])
                    vertex_explorer.Next()
                    continue
                
                vertex_features_list.append((index_counter, vertex_features))
                edge_index_edge_vertex_list.append([current_edge_counter, index_counter])
                index_to_type[index_counter] = 'vertex'
                index_counter += 1
                
                vertex_explorer.Next()
            
            edge_explorer.Next()
        
        
        face_explorer.Next()

    graph_data = SBGCN_graph.GraphHeteroData(face_features_list, edge_features_list, vertex_features_list,
                                  edge_index_face_edge_list, edge_index_edge_vertex_list)
    
    return graph_data


create_graph_from_step_file( '../preprocessing/canvas/step_4.step')

class BRep_Dataset(Dataset):
    def __init__(self, data_path, num_graphs = 32):
        self.data_path = data_path
        self.graphs = []

        graph = create_graph_from_step_file(self.data_path)

        graph.count_nodes()

        for i in range(num_graphs):
            self.graphs.append(graph)


    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]

        return graph
    
