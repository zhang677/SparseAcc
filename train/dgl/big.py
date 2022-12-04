import numpy as np
import scipy.sparse as sp

agg1_index = np.load("../trace/agg1_index.npy")
agg1_adj = np.load("../trace/agg1_adj.npy")
coo_row = agg1_index[0]
coo_col = agg1_index[1]
input_feature = np.load("../trace/input_feature.npy")

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

from dgl.data import PubmedGraphDataset
from dgl import AddSelfLoop
from inference import dgl_GraphConv
import torch.nn.functional as F
import torch
raw_dir = "../data/dgl"
transform = AddSelfLoop() 
data = PubmedGraphDataset(raw_dir=raw_dir, transform=transform)
g = data[0].int()
fc1_weight = np.load("../trace/fc1_weight.npy")
fc1_bias = np.load("../trace/fc1_bias.npy")
input_feature_t = torch.from_numpy(input_feature)
fc1_weight_t = torch.from_numpy(fc1_weight)
fc1_bias_t = torch.from_numpy(fc1_bias)


# Check both
feat1_dgl = dgl_GraphConv(500, 16, g, input_feature_t, fc1_weight_t, fc1_bias_t, norm='both')
feat1_dgl = F.relu(feat1_dgl)
feat1_dgl_np = np.array(feat1_dgl)

feat1 = both.dot(input_feature.dot(fc1_weight)) + fc1_bias
feat1 = feat1 * (feat1 > 0)

agg1_coo = sp.coo_matrix((agg1_adj, (coo_row, coo_col)), shape=(num_nodes, num_nodes))
feat1_ori = agg1_coo.dot(input_feature.dot(fc1_weight)) + fc1_bias
feat1_ori = feat1_ori * (feat1_ori > 0)

print("Both")
print(f"feat1_dgl_np vs. feat1: {np.all(np.isclose(feat1_dgl_np, feat1, rtol=1e-5, atol=1e-7), axis=0)}")
print(f"feat1_ori vs. feat1: {np.all(np.isclose(feat1_ori, feat1, rtol=1e-5, atol=1e-7), axis=0)}")

# Check no-norm
feat1_dgl = dgl_GraphConv(500, 16, g, input_feature_t, fc1_weight_t, fc1_bias_t, norm='None')
feat1_dgl = F.relu(feat1_dgl)
feat1_dgl_np = np.array(feat1_dgl)

feat1 = agg1_coo_ones.dot(input_feature.dot(fc1_weight)) + fc1_bias
feat1 = feat1 * (feat1 > 0)

print("None")
print(f"feat1_dgl_np vs. feat1: {np.all(np.isclose(feat1_dgl_np, feat1, rtol=1e-5, atol=1e-7), axis=0)}")

# Check left
feat1_dgl = dgl_GraphConv(500, 16, g, input_feature_t, fc1_weight_t, fc1_bias_t, norm='left')
feat1_dgl = F.relu(feat1_dgl)
feat1_dgl_np = np.array(feat1_dgl)

feat1 = left.dot(input_feature.dot(fc1_weight)) + fc1_bias
feat1 = feat1 * (feat1 > 0)

print("Left")
print(f"feat1_dgl_np vs. feat1: {np.all(np.isclose(feat1_dgl_np, feat1, rtol=1e-5, atol=1e-7), axis=0)}")

# Check right
feat1_dgl = dgl_GraphConv(500, 16, g, input_feature_t, fc1_weight_t, fc1_bias_t, norm='right')
feat1_dgl = F.relu(feat1_dgl)
feat1_dgl_np = np.array(feat1_dgl)

feat1 = right.dot(input_feature.dot(fc1_weight)) + fc1_bias
feat1 = feat1 * (feat1 > 0)

print("Right")
print(f"feat1_dgl_np vs. feat1: {np.all(np.isclose(feat1_dgl_np, feat1), axis=0)}")