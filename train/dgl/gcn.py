import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl import AddSelfLoop
import argparse

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
    
def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g, features, labels, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # training loop
    for epoch in range(10):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(g, features, labels, val_mask, model)
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(epoch, loss.item(), acc))

def trace_fx(model, g, features):
    from torch.fx import symbolic_trace

    model.eval()
    print(model)
    symbolic_traced = symbolic_trace(model, concrete_args={"g": g, "features": features})
    print(symbolic_traced.graph)

def trace_jit(model, g, features):
    with torch.no_grad(): 
        jit_model = torch.jit.trace(model, (g, features), '../trace/jit_model.pth')
        print(jit_model) 

def trace_profile(model, g, features):
    from torch.profiler import profile, ProfilerActivity

    model.eval()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        model(g, features)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(model.layers[0])
    prof.export_chrome_trace("../trace/trace.json")

def print_model(model, g, feature):
    model.eval()
    print(model)
    print(type(feature))
    print(type(g))
    for layer in model.layers:
        print(layer.__class__)
        print(layer.__dict__['_norm'])
        print(layer.__dict__['_in_feats'])
        print(layer.__dict__['_out_feats'])
        if layer.__dict__['_activation'] is not None:
            print(layer.__dict__['_activation'].__name__)
        print(layer.state_dict()['weight'].shape)
        print(layer.state_dict()['bias'].shape)

def enlarge_feature(g):
    from math import log2, pow

    shape = g.ndata['feat'].shape
    feat_num = pow(2, int(log2(shape[1])) + 1)
    g.ndata['feat'] = F.pad(g.ndata['feat'], (0,int(feat_num-shape[1])), "constant", 0)

def store_adj(g, norm, num):
    import numpy as np

    tg = dgl.reorder_graph(g, edge_permute_algo='src')
    tg = dgl.reorder_graph(tg, edge_permute_algo='dst') # Sort for the [dst->src] output

    in_degs = tg.in_degrees().float().clamp(min=1)
    r_norm = 1 / in_degs
    tg.ndata['r_norm'] = r_norm.unsqueeze(1)
    out_degs = tg.out_degrees().float().clamp(min=1)
    l_norm = 1 / out_degs
    tg.ndata['l_norm'] = l_norm.unsqueeze(1)
    if norm == "left":
        def copy(edges):
            return {'norm': edges.src['l_norm']} # After sorting, it should get from src.
    elif norm == "right":
        def copy(edges):
            return {'norm': edges.dst['r_norm']}
    tg.apply_edges(copy)

    agg_adj = tg.edata['norm'].cpu().numpy().transpose()
    f = open(f"../trace/agg{int(num)}_adj.npy", "wb")
    np.save(f, agg_adj)
    f.close()

    coo = tg.adjacency_matrix(transpose = True ,scipy_fmt = 'coo')
    row_ids = np.expand_dims(coo.row, axis=0)
    col_ids = np.expand_dims(coo.col, axis=0)
    agg_index = np.concatenate((row_ids, col_ids), axis=0)
    f = open(f"../trace/agg{int(num)}_index.npy", "wb")
    np.save(f, agg_index)
    f.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora",
                        help="Dataset name ('cora', 'citeseer', 'pubmed').")
    parser.add_argument("--trace", type=str, default="fx",
                        help="Trace tool ('fx', 'jit', 'prof', 'print','none')")
    parser.add_argument("--enlarge", action="store_true", help="Enlarge input feature to power of 2")
    args = parser.parse_args()
    print(f'Training with DGL built-in GraphConv module.')

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.enlarge:
        enlarge_feature(g)

    g = g.int().to(device)
    features = g.ndata['feat']
    labels = g.ndata['label']
    masks = g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']
        
    # normalization
    # [GENGHAN]: No need to calcuate here.
    #degs = g.in_degrees().float()
    #norm = torch.pow(degs, -0.5).to(device)
    #norm[torch.isinf(norm)] = 0
    #g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model    
    in_size = features.shape[1]
    out_size = data.num_classes
    model = GCN(in_size, 16, out_size).to(device)

    # model training
    print('Training...')
    train(g, features, labels, masks, model)

    if args.trace == 'fx':
        trace_fx(model, g, features)
    elif args.trace == 'jit':
        trace_jit(model, g, features)    
    elif args.trace == 'prof':
        trace_profile(model, g, features)
    elif args.trace == 'print':
        print_model(model, g, features)
        
    # test the model
    print('Testing...')
    acc = evaluate(g, features, labels, masks[2], model)
    print("Test accuracy {:.4f}".format(acc))