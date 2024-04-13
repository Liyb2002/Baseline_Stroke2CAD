import torch.nn as nn
from torch_geometric.nn import EdgeConv, HeteroConv, GCNConv
from torch_geometric.nn.models import MLP
from .basic import MLPLinear
import numpy as np
from torch_geometric.nn import aggr
import torch_scatter
from torch_geometric.utils import scatter


from torch_geometric.nn.conv import MessagePassing

def act_layer(act_type, inplace=False, neg_slope=0.2, n_prelu=1):
    """
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return layer


def norm_layer(norm_type, nc):
    """
    """
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm1d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm1d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return layer

class MLPLinear(nn.Sequential):
    def __init__(self, channels, act_type='relu', norm_type='batch', bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))
            if norm_type and norm_type != 'None':
                m.append(norm_layer(norm_type, channels[i]))
            if act_type:
                m.append(act_layer(act_type))
        super(MLPLinear, self).__init__(*m)
        
class MultiSeq(nn.Sequential):
    def __init__(self, *args):
        super(MultiSeq, self).__init__(*args)

    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
    


class GeneralHeteroConv(torch.nn.Module):
    def __init__(self, gcn_types, in_channels, out_channels, instance_net_type = None):
        super(GeneralHeteroConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gcn_types = gcn_types
        self.instance_net_type = instance_net_type
        self.gconv = HeteroConv(self.create_HeteroConv_dict(), aggr='sum')
            
    def find_aggr_fun(self):
        aggr_fns = []
        for gcn_type in self.gcn_types:
            gcn_type_split = gcn_type.split('_')
            aggr_fns.append(gcn_type_split[-1])
        return aggr_fns
    
    def create_HeteroConv_dict(self):
        heteroConv_dict = {}
        edges_types = [('stroke', 'temp_previous', 'stroke'),
                       ('stroke', 'intersects', 'stroke')]
        if self.instance_net_type =='HeteroConv':
            edges_types.append(('stroke', 'semantic', 'stroke'))
        aggr_fns = self.find_aggr_fun()
        for i in range(len(edges_types)):
            if self.instance_net_type =='HeteroConv' and i == len(edges_types) - 1 :
                heteroConv_dict[edges_types[i]] = EdgeConv(
                                                    nn=MLPLinear(
                                                        channels=[self.in_channels*2, self.out_channels],
                                                        act_type='relu', 
                                                        norm_type=None
                                                    ),
                                                    aggr=aggr_fns[i]
                                                )
            else:
                heteroConv_dict[edges_types[i]] = EdgeConv(
                                                    nn=MLPLinear(
                                                        channels=[self.in_channels*2, self.out_channels],
                                                        act_type='relu', 
                                                        norm_type=None
                                                    ),
                                                    aggr=aggr_fns[i]
                                                )
            
        return heteroConv_dict
            
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict = None, data=None):
        """
        x: (BxN) x F
        """
        if edge_attr_dict is None:
            res = self.gconv(x_dict, edge_index_dict)
        else:
            res = self.gconv(x_dict, edge_index_dict, edge_attr_dict)
        return res   
    
class ResidualGeneralHeteroConvBlock(torch.nn.Module):
    def __init__(self, gcn_types, in_channels, out_channels, is_instance_net = False):
        super(ResidualGeneralHeteroConvBlock, self).__init__()
        self.mlp_edge_conv = GeneralHeteroConv(gcn_types, in_channels, out_channels, is_instance_net)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict = None, data=None):
        """
        x: (BxN) x F
        """
        out = self.mlp_edge_conv(x_dict, edge_index_dict, edge_attr_dict, data)
        out['stroke'] += x_dict['stroke']
        return out  
    
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, 2 * out_channels)
        self.conv3 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.conv3(x, edge_index)