from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopoDS import TopoDS_Shape, topods
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRepTools import breptools
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop

from torch.utils.data import Dataset
from itertools import combinations

import torch
import os

def read_step_file(filename):
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(filename)
    
    if status == 1:  # Check if the read was successful
        step_reader.TransferRoot()  # Transfers the whole STEP file
        shape = step_reader.Shape()  # Retrieves the translated shape
        return shape
    else:
        raise Exception("Error reading STEP file.")


def create_edge_node(edge):
    properties = GProp_GProps()
    brepgprop.LinearProperties(edge, properties)

    vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)

    verts = []
    while vertex_explorer.More():
        vertex = topods.Vertex(vertex_explorer.Current())
        pt = BRep_Tool.Pnt(vertex)
        verts.append([pt.X(), pt.Y(), pt.Z()])
        vertex_explorer.Next()

    edge_features = [verts[0][0], verts[0][1], verts[0][2], verts[1][0], verts[1][1], verts[1][2]]
    return edge_features


def check_duplicate(new_feature, feature_list, face = 0):
    for existing_feature in feature_list:
        if existing_feature == new_feature:
            return 0
    
    return -1


def create_graph_from_step_file(step_path):
    shape = read_step_file(step_path)

    edge_features_list = []
    
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = topods.Face(face_explorer.Current())

        # Explore edges of the face
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while edge_explorer.More():
            edge = topods.Edge(edge_explorer.Current())
            edge_features = create_edge_node(edge)

            edge_duplicate_id = check_duplicate(edge_features, edge_features_list)
            if edge_duplicate_id != -1:
                edge_explorer.Next()
                continue
            
            edge_features_list.append(edge_features)
            
            edge_explorer.Next()
        
        
        face_explorer.Next()
    
    return edge_features_list

