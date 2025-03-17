import pickle
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Mish, Embedding
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ZINC
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, GraphNorm, BatchNorm


def train(model, data, optimizer, criterion, args):
    model.train()
    # data = next(iter(train_loader)) # Iterate in batches over the training dataset.
    X = data[0].type(torch.float32).to(args.device)
    Y = data[1].type(torch.float32).to(args.device)

    optimizer.zero_grad()  # Clear gradients.
    
    batch_loss = 0    
    pred = model(X)
    loss = criterion(pred, Y)
    loss.backward()  # Derive gradients.
    batch_loss += loss.detach().cpu().numpy()
    optimizer.step()  # Update parameters based on gradients.
        
    return batch_loss

# Test loop
def test(model, test_loader, criterion, args):
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:  # Iterate in batches over the test dataset.
            X = data[0].type(torch.float32).to(args.device)
            Y = data[1].type(torch.float32).to(args.device)

            pred = model(X)
            loss = criterion(pred, Y)
            # print(f'Test MSE: {loss:.4f}')
            total_loss += loss.detach().cpu().numpy()
    return total_loss/len(test_loader.dataset)