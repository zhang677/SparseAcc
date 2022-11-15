from gcn import GCN
import torch
import torch.nn as nn
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl import AddSelfLoop
import numpy as np
import argparse

def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def inference(g, features, model, path):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        f = open(path, "wb")
        print(logits.shape)
        np.save(f, logits.cpu().numpy())
        f.close()

def train(g, features, labels, masks, model, epochs):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # training loop
    for epoch in range(epochs):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(g, features, labels, val_mask, model)
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(epoch, loss.item(), acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora",
                        help="Dataset name ('cora', 'citeseer', 'pubmed').")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--train", action="store_true", help="Do training")
    parser.add_argument("--path", type=str, default="../trace/model.pt", help="Model for inference")

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

    g = g.int().to(device)
    features = g.ndata['feat']
    labels = g.ndata['label']
    masks = g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']

    in_size = features.shape[1]
    out_size = data.num_classes
    hidden_size = 16
    

    # model training
    if args.train == True:
        print('Training...')
        model = GCN(in_size, hidden_size, out_size).to(device)
        train(g, features, labels, masks, model, args.epochs)
        print(f"Save model to {args.path}")
        torch.save(model, args.path)
    else:
        print("Inference...")
        model = torch.load(args.path).to(device)
        model_path = "../trace/output_feature.npy"
        inference(g, features, model, model_path)
    