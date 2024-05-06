
import brep_read
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader 

import SBGCN_network
import decoder
from tqdm import tqdm


def train_graph_embedding(dataset, num_epochs=10, batch_size=1, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    model = SBGCN_network.FaceEdgeVertexGCN()
    decoder_model = decoder.SBGCN_Decoder()
    optimizer = optim.Adam(list(model.parameters()) + list(decoder_model.parameters()), lr=learning_rate)

    # Create DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                                 shuffle=False)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0

        
        # Iterate over batches
        for batch in tqdm(dataloader):

            step_path = batch[0]
            graph = brep_read.create_graph_from_step_file(step_path)
            graph.count_nodes()
                        # Forward pass
            x_f, x_e, x_v = model(graph)
            reconstruct_matrix = decoder_model(x_f, x_e, x_v)
            gt_matrix = graph['face'].z
            loss = criterion(reconstruct_matrix, gt_matrix)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(batch)


    
    return model








def run():


    step_path =  ['../preprocessing/canvas/step_4.step']
    for i in range(1):
        step_path.append('../preprocessing/canvas/step_4.step')

    dataset = brep_read.BRep_Dataset(step_path)


    model = train_graph_embedding(dataset)

    print("done")


run()