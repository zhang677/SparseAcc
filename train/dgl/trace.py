from gcn import GCN
from math import log2, pow
import torch
import torch.nn.functional as F
import argparse
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl import AddSelfLoop
import dgl
import numpy as np
import yaml


class Counter:
    def __init__(self, names) -> None:
        self.states = {}
        for name in names:
            self.states[name] = 0
    
    def add(self, name):
        if name in self.states.keys():
            self.states[name] += 1
        else:
            self.states[name] = 1
        return self.states[name]
    
    def query(self, name):
        assert name in self.states.keys()

        return self.states[name] 

def get_upper_power_two(x: int):
    return int(pow(2, int(log2(x)) + 1))

def enlarge_and_save(t: torch.Tensor, dims ,name: str):
    print(f"{name}: {t.shape}")
    if len(t.shape) == 1:
        old_col = t.shape[0]
        f = open(f"../trace/{name}.npy", "wb")
        new_col = get_upper_power_two(old_col)
        F.pad(t, (0, int(new_col - old_col)), "constant", 0)
        np.save(f, t.cpu().numpy())
        f.close()
        return torch.Size([new_col])
    if len(t.shape) == 2:
        (old_row, old_col) = t.shape
        f = open(f"../trace/{name}.npy", "wb")
        if dims == (0,1):
            new_row = get_upper_power_two(old_row)
            new_col = get_upper_power_two(old_col)
            F.pad(t, (0, int(new_col - old_col), 0, int(new_row - old_row)), "constant", 0)
        elif dims == 0:
            new_row = get_upper_power_two(old_row)
            new_col = old_col
            F.pad(t, (0, 0, 0, int(new_row - old_row)), "constant", 0)
        elif dims == 1:
            new_col = get_upper_power_two(old_col)
            new_row = old_row
            F.pad(t, (0, int(new_col - old_col)), "constant", 0)
        else:
            raise NotImplementedError
        np.save(f, t.cpu().numpy())
        f.close()
        return torch.Size((new_row, new_col))
    raise NotImplementedError

def generate_ir(model, g):
    class Counter:
        def __init__(self, names) -> None:
            self.states = {}
            for name in names:
                self.states[name] = 0
        
        def add(self, name):
            if name in self.states.keys():
                self.states[name] += 1
            else:
                self.states[name] = 1
            return self.states[name]
        
        def query(self, name):
            assert name in self.states.keys()

            return self.states[name] 

    model.eval()
    final = []
    counter = Counter(["fc", "agg"])
    for (i, layer) in enumerate(model.layers):
        current = {}
        layer_name = layer.__class__.__name__
        if layer_name == "GraphConv":
            reduce_type = "sum"
            relu = False
            if layer.__dict__['_activation'] is not None:
                if layer.__dict__['_activation'].__name__ == "relu":
                    relu = True
            in_feat = get_upper_power_two(layer.__dict__['_in_feats'])
            out_feat = get_upper_power_two(layer.__dict__['_out_feats'])
            num_nodes = g.num_nodes()

            # Add mm
            current['op_type'] = 'mm'
            fc_num = int(counter.add("fc"))
            current['op_name'] = f"fc{fc_num}"
            if i == 0:
                current['op_input_data'] = {"data_name": "input_feature", "data_shape": [num_nodes, in_feat], "read_data_path": "input_feature.npy"}
            else:
                agg_num = int(counter.query("agg"))
                current['op_input_data'] = {"data_name": f"agg{agg_num}_out_feature", "data_shape": [num_nodes, in_feat]} # Currently assumes the model is a stack of GraphConv
            
            current['op_acc_data'] = None
            current['op_output_data'] = {"data_name": f"fc{fc_num}_out_feature", "data_shape": [num_nodes, out_feat]}
            current['op_weight'] = {"data_name": f"fc{fc_num}_weight", "data_shape": [in_feat, out_feat], "read_data_path": f"fc{fc_num}_weight.npy"}
            current['op_bias'] = {"data_name": f"fc{fc_num}_bias", "data_shape": [1, out_feat], "read_data_path": f"fc{fc_num}_bias.npy"}
            current['accumulation'] = False
            current['bias'] = True
            current['relu'] = relu
            final.append(current)

            # Add agg
            current = {}
            current['op_type'] = 'agg'
            agg_num = int(counter.add("agg"))
            current['op_name'] = f"agg{agg_num}"
            fc_num = int(counter.query("fc"))
            current['op_input_data'] = {"data_name": f"fc{fc_num}_out_feature", "data_shape": [num_nodes, out_feat]}
            current['op_output_data'] = {"data_name": f"agg{agg_num}_out_feature", "data_shape": [num_nodes, out_feat]}
            current['op_adj'] = {"data_name": f"agg{agg_num}_adj", "data_shape": [num_nodes, num_nodes], "non_zeros": g.num_edges(), "read_data_path": f"agg{agg_num}_adj.npy", "read_index_path": f"agg{agg_num}_index.npy"}
            current['apply'] = True
            current['reduce_type'] = reduce_type
            current['relu'] = relu
            final.append(current)

    f = open("../trace/ir_generated.yaml", "w")
    yaml.dump(final, f)

def save_adj(g, norm, name):

    tg = dgl.reorder_graph(g, edge_permute_algo='src')
    tg = dgl.reorder_graph(tg, edge_permute_algo='dst') # Sort for the [dst->src] output

    in_degs = tg.in_degrees().float().clamp(min=1)
    out_degs = tg.out_degrees().float().clamp(min=1)

    tg.ndata['r_norm'] = 1.0 / in_degs
    tg.ndata['l_norm'] = 1.0 / out_degs
    if norm == "left":
        def copy(edges):
            return {'norm': edges.src['l_norm']} # After sorting, it should get from src.
    elif norm == "right":
        def copy(edges):
            return {'norm': edges.dst['r_norm']}
    elif norm == "both":
        tg.ndata['l_norm'] = torch.pow(in_degs, -0.5)
        tg.ndata['r_norm'] = torch.pow(out_degs, -0.5)
        def copy(edges):
            return {'norm': edges.dst['l_norm'] * edges.src['r_norm']}

    tg.apply_edges(copy)

    agg_adj = tg.edata['norm'].cpu().numpy().transpose()
    f = open(f"../trace/{name}_adj.npy", "wb")
    np.save(f, agg_adj)
    f.close()

    coo = tg.adjacency_matrix(transpose = True ,scipy_fmt = 'coo')
    row_ids = np.expand_dims(coo.row, axis=0)
    col_ids = np.expand_dims(coo.col, axis=0)
    agg_index = np.concatenate((row_ids, col_ids), axis=0)
    f = open(f"../trace/{name}_index.npy", "wb")
    np.save(f, agg_index)
    f.close()

def save_all(model, g):
    counter = Counter(["fc", "agg"])
    model.eval()
    for (i, layer) in enumerate(model.layers):
        if i == 0:
            enlarge_and_save(g.ndata['feat'], 1, "input_feature")
        layer_name = layer.__class__.__name__
        if layer_name == "GraphConv":
            fc_num = int(counter.add("fc"))
            enlarge_and_save(layer.state_dict()['weight'], (0,1), f"fc{fc_num}_weight")
            enlarge_and_save(layer.state_dict()['bias'], (0,1), f"fc{fc_num}_bias")
            agg_num = int(counter.add("agg"))
            save_adj(g, layer._norm, f"agg{agg_num}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gcn", 
                        help="Type of the pre-trained model")
    parser.add_argument("--path", type=str, default="../trace/model.pt", 
                        help="Path to the pre-trained model")
    parser.add_argument("--dataset", type=str, default="cora",
                        help="Dataset name ('cora', 'citeseer', 'pubmed').")
    args = parser.parse_args()

    raw_dir = "../data/dgl"
    # load and preprocess dataset
    transform = AddSelfLoop()  # by default, it will first remove self-loops to prevent duplication
    if args.dataset == 'cora':
        data = CoraGraphDataset(raw_dir=raw_dir, transform=transform)
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset(raw_dir=raw_dir, transform=transform)
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset(raw_dir=raw_dir, transform=transform)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    g = data[0]

    model = torch.load(args.path)
    generate_ir(model, g)
    save_all(model, g)



