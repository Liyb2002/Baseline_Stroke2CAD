import torch
from torch_geometric.data import Data, HeteroData


class GraphHeteroData(HeteroData):
    def __init__(self, face_features, edge_features, vertex_features, edge_index_face_edge, edge_index_edge_vertex):
        super(GraphHeteroData, self).__init__()

        self['face'].x = face_features
        self['edge'].x = edge_features
        self['vertex'].x = vertex_features

        self['face'].num_nodes = len(face_features)
        self['edge'].num_nodes = len(edge_features)
        self['vertex'].num_nodes = len(vertex_features)

        self['face', 'connected', 'edge'].edge_index = edge_index_face_edge
        self['edge', 'connects', 'vertex'].edge_index = edge_index_edge_vertex

    def to_device(self, device):
        for key, value in self.items():
            if torch.is_tensor(value):
                self[key] = value.to(device)

    def count_nodes(self):
        num_faces = len(self['face'].x)
        num_edges = len(self['edge'].x)
        num_vertices = len(self['vertex'].x)
        
        print("Number of faces:", num_faces)
        print("Number of edges:", num_edges)
        print("Number of vertices:", num_vertices)

