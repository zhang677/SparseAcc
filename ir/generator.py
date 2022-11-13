import pandas as pd
import json
import os

import torch
import torch.nn as nn
import dgl.nn as dglnn

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(dglnn.GraphConv(in_size, hid_size, activation=F.relu))
        self.layers.append(dglnn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h

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

def export_csv(path, name):
    filename = os.path.join(path, name)
    print(filename)
    df = pd.read_json(filename)
    print(df)
    trace = df['traceEvents'].apply(pd.Series)
    trace.to_csv(filename.replace('.json','.csv'))

def read_json(path, name):
    filename = os.path.join(path, name)
    with open(filename) as f:
        data = json.load(f)['traceEvents']
        trace = []
        for (k, v) in enumerate(data):
            if "cat" in v.keys():
                if v["cat"] != "Kernel":
                    trace.append({"categary": v["cat"], "name": v["name"], "Occupancy %": 0})
                else:
                    print(v)
                    trace.append({"categary": v["cat"], "name": v["name"], "Occupancy %": v["args"]["est. achieved occupancy %"]})
            else:
                trace.append({"categary": "None", "name": v["name"], "Occupancy %": 0})
        df = pd.DataFrame(trace)
        df.to_csv(filename.replace('.json','.csv'), index_label="Kernel_id")

def generate_yaml(model: nn.Module, g):
    import yaml 

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
            in_feat = layer.__dict__['_in_feats']
            out_feat = layer.__dict__['_out_feats']
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

    f = open("ir_generated.yaml", "w")
    yaml.dump(final, f)


if __name__ == "__main__":
    # read_json("../train/trace","trace.json")
    model = torch.load("../train/trace/model.pt")
    from dgl.data import PubmedGraphDataset
    from dgl import AddSelfLoop
    raw_dir = "../train/data/dgl"
    # load and preprocess dataset
    transform = AddSelfLoop()
    data = PubmedGraphDataset(raw_dir=raw_dir, transform=transform)
    g = data[0]
    generate_yaml(model, g)