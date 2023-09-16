import torch as th
import torch.nn as nn
import torch.functional as F
import numpy as np
import dgl
import dgl.nn as dglnn
import sklearn.linear_model as lm
import sklearn.metrics as skm
import tqdm
from conv import myGATConv

class HGAT(nn.Module):
    def __init__(self, edge_feats, num_edges, in_feats, n_hidden, n_classes, n_layers, activation, dropout, feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha):
        super().__init__()
        self.init(edge_feats, num_edges, in_feats, n_hidden, n_classes, n_layers, activation, dropout, feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha)

    def init(self, edge_feats, num_edges, in_feats, n_hidden, n_classes, n_layers, activation, dropout, feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.activation = activation
        self.num_heads = 3
        if n_layers > 1:
            self.layers.append(myGATConv(edge_feats, num_edges, in_feats, n_hidden, self.num_heads, feat_drop, attn_drop, negative_slope, residual=False, activation=self.activation,  allow_zero_in_degree=True, alpha=alpha))
            for i in range(1, n_layers - 1):
                self.layers.append(myGATConv(edge_feats, num_edges, n_hidden*self.num_heads, n_hidden, self.num_heads, feat_drop, attn_drop, negative_slope, residual=True, activation=self.activation, allow_zero_in_degree=True, alpha=alpha))
            self.layers.append(myGATConv(edge_feats, num_edges, n_hidden*self.num_heads, n_classes, self.num_heads, feat_drop, attn_drop, negative_slope, residual=True, activation=self.activation, allow_zero_in_degree=True, alpha=alpha))
        else:
            self.layers.append(myGATConv(edge_feats, num_edges, in_feats, n_classes, num_heads=self.num_heads, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.epsilon = th.FloatTensor([1e-12]).cuda()

    def forward(self, blocks, x, efeats):
        h = x
        for l, (layer, block, efeat) in enumerate(zip(self.layers, blocks, efeats)):
            h = layer(block, h, efeat)
            if l != len(self.layers) - 1:
                h = h.flatten(1)
#             if l != len(self.layers) - 1:
#                 h = self.activation(h)
#                 h = self.dropout(h)
        
#         print(h.shape)
        logits = h.mean(1)
        logits = logits / (th.max(th.norm(logits, dim=1, keepdim=True), self.epsilon))
#         print(logits.shape)
        return logits

    def inference(self, g, x, device, batch_size, num_workers):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        
        print("inference")
        
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.num_nodes(), self.n_hidden*self.num_heads if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            
            train_seeds = th.arange(g.num_nodes()).to(g.device)
            
            dataloader_index = 0
            dataloader_size = 1000000
            
            while dataloader_index * dataloader_size < len(train_seeds):
                print("inference_dataloader", dataloader_index)
                batch_seeds = train_seeds[dataloader_size*dataloader_index: dataloader_size*(dataloader_index+1)]
                dataloader_index += 1
                dataloader = dgl.dataloading.DataLoader(
                    g,
                    batch_seeds,
                    sampler,
                    device=device if num_workers == 0 else None,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=num_workers)

                for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                    block = blocks[0]

                    block = block.int().to(device)
                    h = x[input_nodes].to(device)

#                     e_feat = []
                    
                    e_feat = block.int().edata['etype'].flatten().long().to(device)
#                     for u, v in zip(*block.int().edges()):
#                         u = input_nodes[u].cpu().item()
#                         v = input_nodes[v].cpu().item()
#                         if (u, v) in edge2type:
#                         e_feat.append(edge2type[(u, v)])
#                         else:
#                             e_feat.append(18)
#                     e_feat = th.tensor(e_feat, dtype=th.long).to(device)

                    h = layer(block, h, e_feat)
    #                 if l != len(self.layers) - 1:
    #                     h = self.activation(h)
    #                     h = self.dropout(h)
                    if l != len(self.layers) - 1:
                        h = h.flatten(1)
                    else:
                        h = h.mean(1)
                        h = h / (th.max(th.norm(h, dim=1, keepdim=True), self.epsilon))

                    y[output_nodes] = h.cpu()
            x = y
        
        np.save("emb.npy", y.detach().numpy())
        return y

def compute_acc_unsupervised(emb, labels, train_nids, val_nids, test_nids):
    """
    Compute the accuracy of prediction given the labels.
    """
    emb = emb.cpu().numpy()
    labels = labels.cpu().numpy()
    train_nids = train_nids.cpu().numpy()
    train_labels = labels[train_nids]
    val_nids = val_nids.cpu().numpy()
    val_labels = labels[val_nids]
    test_nids = test_nids.cpu().numpy()
    test_labels = labels[test_nids]

    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)

    lr = lm.LogisticRegression(multi_class='multinomial', max_iter=10000)
    lr.fit(emb[train_nids], train_labels)

    pred = lr.predict(emb)
    f1_micro_eval = skm.f1_score(val_labels, pred[val_nids], average='micro')
    f1_micro_test = skm.f1_score(test_labels, pred[test_nids], average='micro')
    return f1_micro_eval, f1_micro_test