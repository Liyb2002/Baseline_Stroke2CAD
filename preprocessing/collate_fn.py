

import preprocessing.gnn_graph

#---------------------------------------------------------------------#

# def stroke_cloud_collate(batch):
#     CAD_Programs = [item[0] for item in batch]
#     final_edges_list = [item[1] for item in batch]

#     max_length = max(len(edges) for edges in final_edges_list)

#     padded_final_edges_list = []
#     for edges in final_edges_list:
#         start_token = [-1] * 3  
#         padded_edges = start_token + edges + [0] * (max_length - len(edges))
#         padded_final_edges_list.append(padded_edges)

#     padded_final_edges_tensor = torch.tensor(padded_final_edges_list, dtype=torch.float)

#     return CAD_Programs, padded_final_edges_tensor




#---------------------------------------------------------------------#

def stroke_cloud_collate(batch):
    if isinstance(batch[0], preprocessing.gnn_graph.SketchHeteroData):
        return batch
    
    CAD_Programs = [item[0] for item in batch]
    final_edges_list = [item[1] for item in batch]
    strokes_dict_path = [item[2] for item in batch]

    return CAD_Programs, final_edges_list, strokes_dict_path
