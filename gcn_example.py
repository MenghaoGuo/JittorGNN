import os.path as osp
import argparse

import jittor as jt
from jittor import nn
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from jittor_geometric.nn import GCNConv, ChebConv, SGConv, GCN2Conv

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
                             normalize=not args.use_gdc)

    def execute(self):
        x, edge_index, edge_weight = jt.array(data.x.numpy()), jt.array(
            data.edge_index.numpy()).int32(), data.edge_attr
        x = nn.relu(self.conv1(x, edge_index, edge_weight))
        x = nn.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        return nn.log_softmax(x, dim=1)


model, data = Net(), data
optimizer = nn.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.01)  # Only perform weight-decay on first convolution.


def train():
    model.train()
    pred = model()[jt.array(data.train_mask.numpy())]
    label = jt.array(data.y[data.train_mask].numpy())
    loss = nn.nll_loss(pred, label)
    optimizer.step(loss)


def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        y_ = jt.array(data.y[mask].numpy())
        mask = jt.array(mask.numpy())
        tmp = []
        for i in range(mask.shape[0]):
            if mask[i] == True:
                tmp.append(logits[i])
        logits_ = jt.stack(tmp)
        pred, _ = jt.argmax(logits_, dim=1)
        acc = pred.equal(y_).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


# train()
best_val_acc = test_acc = 0
for epoch in range(1, 201):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))
