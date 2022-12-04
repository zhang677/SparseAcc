# Implement a 2-layer GCN using numpy?
# Current norm=='both'
# Load agg1_adj and agg1_index 
# Create COO matrix
# Load input_feature
# AGG
# Load fc1_weight and fc1_bias
# UPDATE
# Relu
# Load agg2_adj and agg2_index 
# Create COO matrix
# AGG
# Load fc1_weight and fc1_bias
# UPDATE
# Load output_feature
# Assert equal
import dgl.function as fn
from dgl.utils import expand_as_pair
import torch as th

def dgl_GraphConv(in_feats , out_feats, graph, feat, weight, bias=None, norm='both', edge_weight=None):
    norm_name = norm
    with graph.local_scope():
        aggregate_fn = fn.copy_src('h', 'm')
        if edge_weight is not None:
            assert edge_weight.shape[0] == graph.number_of_edges()
            graph.edata['_edge_weight'] = edge_weight
            aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')
        
        feat_src, feat_dst = expand_as_pair(feat, graph)
        if norm_name in ['left', 'both']:
            degs = graph.out_degrees().float().clamp(min=1)
            if norm_name == 'both':
                norm = th.pow(degs, -0.5)
            else:
                norm = 1.0 / degs
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = th.reshape(norm, shp)
            feat_src = feat_src * norm
        
        if in_feats > out_feats:
            feat_src = th.matmul(feat_src, weight)
            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
        else:
            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
            rst = th.matmul(graph.dstdata['h'], weight)

        if norm_name in ['right', 'both']:
            degs = graph.in_degrees().float().clamp(min=1)
            if norm_name == 'both':
                norm = th.pow(degs, -0.5)
            else:
                norm = 1.0 / degs
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = th.reshape(norm, shp)
            rst = rst * norm
        
        if bias is not None:
            rst = rst + bias
        
        return rst



        



