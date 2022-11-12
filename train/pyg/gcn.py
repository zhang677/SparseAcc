import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import log
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, use_gdc):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True,
                             normalize=not use_gdc)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True,
                             normalize=not use_gdc)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

@torch.no_grad()
def test(model, data):
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_weight).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

def train(model, data, args):
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0)
    ], lr=args.lr)  # Only perform weight-decay on first convolution.
    best_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_weight)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        train_acc, val_acc, tmp_test_acc = test(model, data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)

def trace_fx(model, x, edge_index, edge_weight):
    from torch.fx import symbolic_trace
    model.eval()
    symbolic_traced = symbolic_trace(model, concrete_args={"x": x, "edge_index": edge_index, "edge_weight": edge_weight})
    print(symbolic_traced.graph)

def trace_jit(dataset, args, device):
    class JitGCN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, use_gdc):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels, cached=True,
                                normalize=not use_gdc).jittable()
            self.conv2 = GCNConv(hidden_channels, out_channels, cached=True,
                                normalize=not use_gdc).jittable()

        def forward(self, x, edge_index, edge_weight=None):
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv1(x, edge_index, edge_weight).relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
            return x
    model = JitGCN(dataset.num_features, args.hidden_channels, dataset.num_classes, args.use_gdc)
    data = dataset[0]
    model, data = model.to(device), data.to(device)
    jit_model = torch.jit.script(model)
    print(jit_model(data.x, data.edge_index, data.edge_weight))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--hidden_channels', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
    parser.add_argument('--trace', type=str, default="fx", help="Trace tool ('fx', 'jit')")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = "../data/pyg"
    dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0]

    if args.use_gdc:
        transform = T.GDC(
            self_loop_weight=1,
            normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=dict(method='ppr', alpha=0.05),
            sparsification_kwargs=dict(method='topk', k=128, dim=0),
            exact=True,
        )
        data = transform(data)


    model = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes, args.use_gdc)
    model, data = model.to(device), data.to(device)
    
    if args.trace == "jit":
        trace_jit(dataset, args, device)

    # model training
    print("Training...")
    train(model, data, args)

    # model testing
    print("Testing...")
    accs = test(model, data)
    print(f"Test accuracy {round(accs[0],4)}, {round(accs[1],4)}, {round(accs[2],4)}")
