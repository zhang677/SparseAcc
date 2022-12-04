import numpy as np
import scipy.sparse as sp


agg1_index = np.load("../trace/agg1_index.npy")
agg1_adj = np.load("../trace/agg1_adj.npy")
input_feature = np.load("../trace/feat1.npy")
coo_row = agg1_index[0]
coo_col = agg1_index[1]
num_nodes = input_feature.shape[0]

"""
num_nodes = input_feature.shape[0]
agg1_coo_ones = sp.coo_matrix((np.ones(agg1_adj.shape), (coo_row, coo_col)), shape=(num_nodes, num_nodes))
agg1_coo_ones_T = sp.coo_matrix((np.ones(agg1_adj.shape), (coo_col, coo_row)), shape=(num_nodes, num_nodes))
temp_ones = np.ones((num_nodes, num_nodes))
right = agg1_coo_ones.multiply(agg1_coo_ones.dot(temp_ones))
right.data = 1 / right.data
left = (agg1_coo_ones_T.multiply(agg1_coo_ones_T.dot(temp_ones))).transpose()
left.data = 1 / left.data
both = left.multiply(right)
both.data = np.sqrt(both.data)
"""

agg1_coo = sp.coo_matrix((agg1_adj, (coo_row, coo_col)), shape=(num_nodes, num_nodes))
fc1_weight = np.load("../trace/fc1_weight.npy")
feat1 = agg1_coo.dot(input_feature.dot(fc1_weight))
feat1 = feat1 * (feat1 > 0)

agg2_index = np.load("../trace/agg2_index.npy")
agg2_adj = np.load("../trace/agg2_adj.npy")
coo_row = agg2_index[0]
coo_col = agg2_index[1]
num_nodes = feat1.shape[0]
agg2_coo = sp.coo_matrix((agg2_adj, (coo_row, coo_col)), shape=(num_nodes, num_nodes))
fc2_weight = np.load("../trace/fc2_weight.npy")
feat2 = agg2_coo.dot(feat1.dot(fc2_weight))


from dgl.data import PubmedGraphDataset
from dgl import AddSelfLoop
from inference import dgl_GraphConv
import torch.nn.functional as F
import torch
raw_dir = "../data/dgl"
transform = AddSelfLoop() 
data = PubmedGraphDataset(raw_dir=raw_dir, transform=transform)
g = data[0].int()
input_feature_t = torch.from_numpy(input_feature)
fc1_weight_t = torch.from_numpy(fc1_weight)
feat1_dgl = dgl_GraphConv(fc1_weight.shape[0], fc1_weight.shape[1], g, input_feature_t, fc1_weight_t, norm='both')
feat1_dgl = F.relu(feat1_dgl)

fc2_weight_t = torch.from_numpy(fc2_weight)
feat2_dgl = dgl_GraphConv(fc2_weight.shape[0], fc2_weight.shape[1], g, feat1_dgl, fc2_weight_t, norm='both')
feat2_dgl_np = np.array(feat2_dgl)

print(f"feat2_dgl_np vs. feat2: {np.all(np.isclose(feat2_dgl_np, feat2, rtol=1e-5, atol=1e-6), axis=0)}")
print((feat2_dgl_np-feat2)[np.nonzero(np.isclose(feat2_dgl_np, feat2, rtol=1e-5, atol=1e-7) == False)])


import yaml
from os import path
root = "../trace/"
f = open(path.join(root,"ir_generated.yaml"), "r")
totinfo = yaml.safe_load(f)
bias = None
feat = 0
for info in totinfo:
    if info['op_type'] == 'mm':
        input_feat = np.load(path.join(root,info['op_input_data']['read_data_path']))
        weight = np.load(path.join(root,info['op_weight']['read_data_path']))
        feat_shape = info['op_weight']['data_shape']
        feat = input_feat.dot(weight)        
        
    elif info['op_type'] == 'agg':
        if info['reduce_type'] == 'sum':
            index = np.load(path.join(root,info['op_adj']['read_index_path']))
            adj = np.load(path.join(root,info['op_adj']['read_data_path']))
            num_nodes = info['op_adj']['data_shape'][0]
            agg_coo = sp.coo_matrix((adj, (index[0], index[1])), shape=(num_nodes, num_nodes))
            input_feat = np.load(path.join(root,info['op_input_data']['read_data_path']))
            feat = agg_coo.dot(input_feat)

    if info['bias'] == True:
        bias = np.load(path.join(root,info['op_bias']['read_data_path']))
        feat = feat + bias
    if info['relu'] == True:
        feat = feat * (feat > 0)
    np.save(path.join(root,info['op_output_data']['write_data_path']), feat)

ir_feat = np.load(path.join(root,"feat5.npy"))
print(f"ir_feat vs. feat2: {np.all(np.isclose(ir_feat, feat2, rtol=1e-5, atol=1e-6), axis=0)}")
print((ir_feat-feat2)[np.nonzero(np.isclose(ir_feat, feat2, rtol=1e-5, atol=1e-7) == False)])

from utils import enlarge_and_save

enlarge_and_save(torch.from_numpy(np.load(path.join(root,"true_output.npy"))), 1, "enlarge_true_output")
true_output = np.load(path.join(root,"enlarge_true_output.npy"))
print(f"ir_feat vs. true_feat: {np.all(np.isclose(ir_feat, true_output, rtol=1e-2, atol=0), axis=0)}")
print((np.abs(ir_feat-true_output) / true_output)[np.nonzero(np.isclose(ir_feat, true_output, rtol=1e-2, atol=0) == False)])


