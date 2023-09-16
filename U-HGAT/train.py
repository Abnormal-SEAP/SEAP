import os
import dgl
import numpy as np
import random
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import time
import argparse
from torch.nn.parallel import DistributedDataParallel

from model import HGAT, compute_acc_unsupervised as compute_acc
from negative_sampler import NegativeSampler
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class CrossEntropyLoss(nn.Module):
    def forward(self, block_outputs, pos_graph, neg_graph):
        with pos_graph.local_scope():
            pos_graph.ndata['h'] = block_outputs
            pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            pos_score = pos_graph.edata['score']
        with neg_graph.local_scope():
            neg_graph.ndata['h'] = block_outputs
            neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            neg_score = neg_graph.edata['score']

        score = th.cat([pos_score, neg_score])
        label = th.cat([th.ones_like(pos_score), th.zeros_like(neg_score)]).long()
        loss = F.binary_cross_entropy_with_logits(score, label.float())
        return loss

def evaluate(model, g, nfeat, labels, train_nids, val_nids, test_nids, device, args):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        # single gpu
        if isinstance(model, HGAT):
            pred = model.inference(g, nfeat, device, args.batch_size, args.num_workers)
        # multi gpu
        else:
            pred = model.module.inference(g, nfeat, device, args.batch_size, args.num_workers)
    model.train()
    return compute_acc(pred, labels, train_nids, val_nids, test_nids)

#### Entry point
def run(proc_id, n_gpus, args, devices, data):
    # Unpack data
    device = th.device(devices[proc_id])
    if n_gpus > 0:
        th.cuda.set_device(device)
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=proc_id)
    train_nid, val_nid, test_nid, n_classes, g, nfeat, labels = data

    if args.data_device == 'gpu':
        nfeat = nfeat.to(device)
    elif args.data_device == 'uva':
        nfeat = dgl.contrib.UnifiedTensor(nfeat, device=device)
    in_feats = nfeat.shape[1]
    
    print("dataloader")

    # Create PyTorch DataLoader for constructing blocks
    n_edges = g.num_edges()
    train_seeds = th.arange(n_edges)

    if args.graph_device == 'gpu':
        train_seeds = train_seeds.to(device)
        g = g.to(device)
        args.num_workers = 0
    elif args.graph_device == 'uva':
        train_seeds = train_seeds.to(device)
        g.pin_memory_()
        args.num_workers = 0

    
    # Create sampler
    sampler = dgl.dataloading.NeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    
    print(len(train_seeds))
    
    sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler, exclude='reverse_id',
        # For each edge with ID e in Reddit dataset, the reverse edge is e ± |E|/2.
        reverse_eids=th.cat([
            th.arange(n_edges // 2, n_edges),
            th.arange(0, n_edges // 2)]).to(train_seeds),
        negative_sampler=NegativeSampler(g, args.num_negs, args.neg_share,
                                         device if args.graph_device == 'uva' else None))
    
#     print(train_seeds.numpy())

#     # train_seeds = th.tensor(random.sample(list(train_seeds.numpy()), 100))
#     dataloader = dgl.dataloading.DataLoader(
#         g, train_seeds, sampler,
#         device=device,
# #         use_ddp=n_gpus > 1,
#         batch_size=args.batch_size,
#         shuffle=True,
#         drop_last=False,
#         num_workers=0)
# #         use_uva=args.graph_device == 'uva')
    
    print("sampler done")
    
    
    

    # Define model and optimizer
    model = HGAT(args.edge_feats, args.num_edges, in_feats, args.num_hidden, n_classes, args.num_layers, F.elu, args.dropout, args.dropout, args.dropout, args.slope, True, 0.05)
    model = model.to(device)
    if n_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[device], output_device=device)
    loss_fcn = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    
    if args.inference:
        model.load_state_dict(th.load("model.pt"))
        model.eval()
        with th.no_grad():
            model.inference(g, nfeat, device, args.batch_size, args.num_workers)

    else:
        # Training loop
        avg = 0
        iter_pos = []
        iter_neg = []
        iter_d = []
        iter_t = []
        best_eval_acc = 0
        best_test_acc = 0
        for epoch in range(args.num_epochs):
            tic = time.time()

            dataloader_index = 0
            dataloader_size = int(len(train_seeds) / 5) + 1
            random_seeds = train_seeds.numpy()
            np.random.shuffle(random_seeds)
            train_seeds = th.tensor(random_seeds)
#             print(train_seeds)
            while dataloader_index * dataloader_size < len(train_seeds):
                print("train_dataloader", dataloader_index)
                batch_seeds = train_seeds[dataloader_size*dataloader_index: dataloader_size*(dataloader_index+1)]
                dataloader_index += 1
                dataloader = dgl.dataloading.DataLoader(
                    g, batch_seeds, sampler,
                    device=device,
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=False,
                    num_workers=0)

                # Loop over the dataloader to sample the computation dependency graph as a list of
                # blocks.
                tic_step = time.time()
                for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(dataloader):
#                     print(step)
        #             print(len(input_nodes))
#                     print("pos_graph", pos_graph, pos_graph.edges())
#                     print("neg_graph", neg_graph, neg_graph.edges())

        #             isolated_nodes = ((pos_graph.in_degrees() == 0) & (pos_graph.out_degrees() == 0)).nonzero().squeeze(1)
        #             print(len(isolated_nodes))
                    input_nodes = input_nodes.to(nfeat.device)
                    batch_inputs = nfeat[input_nodes].to(device)
                    blocks = [block.int() for block in blocks]
                    efeats = []
                    for block in blocks:
                        # print("isolate", block.dstdata[dgl.NID][block.in_degrees()==0])
#                         print("block", block)
                        e_feat = []
#                         print("begin edge", step)
                        efeats.append(block.int().edata['etype'].flatten().long().to(device))
#                         for u, v in zip(*block.int().edges()):
#                             u = input_nodes[u].cpu().item()
#                             v = input_nodes[v].cpu().item()
#                             if (u, v) in edge2type:
# #                                 print(1)
#                                 e_feat.append(edge2type[(u, v)])
#                             else:
#                                 print(2)
#                                 e_feat.append(18)
                        # efeats.append(th.tensor(e_feat, dtype=th.long).to(device))
#                         print("end edge", step)
                    d_step = time.time()

                    # Compute loss and prediction
                    batch_pred = model(blocks, batch_inputs, efeats)
                    loss = loss_fcn(batch_pred, pos_graph, neg_graph)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    th.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                        }, "checkpoint.pt")

                    t = time.time()
                    pos_edges = pos_graph.num_edges()
                    neg_edges = neg_graph.num_edges()
                    iter_pos.append(pos_edges / (t - tic_step))
                    iter_neg.append(neg_edges / (t - tic_step))
                    iter_d.append(d_step - tic_step)
                    iter_t.append(t - d_step)
                    if step % args.log_every == 0 and proc_id == 0:
                        gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                        print('[{}]Epoch {:05d} | Step {:05d} | Loss {:.4f} | Speed (samples/sec) {:.4f}|{:.4f} | Load {:.4f}| train {:.4f} | GPU {:.1f} MB'.format(
                            proc_id, epoch, step, loss.item(), np.mean(iter_pos[3:]), np.mean(iter_neg[3:]), np.mean(iter_d[3:]), np.mean(iter_t[3:]), gpu_mem_alloc))
                    tic_step = time.time()

                toc = time.time()
                if proc_id == 0:
                    print('Epoch Time(s): {:.4f}'.format(toc - tic))
                    if epoch >= 5:
                        avg += toc - tic
        #             if (epoch + 1) % args.eval_every == 0:
        #                 eval_acc, test_acc = evaluate(model, g, nfeat, labels, train_nid, val_nid, test_nid, device, args)
        #                 print('Eval Acc {:.4f} Test Acc {:.4f}'.format(eval_acc, test_acc))
        #                 if eval_acc > best_eval_acc:
        #                     best_eval_acc = eval_acc
        #                     best_test_acc = test_acc
        #                 print('Best Eval Acc {:.4f} Test Acc {:.4f}'.format(best_eval_acc, best_test_acc))

                if n_gpus > 1:
                    th.distributed.barrier()

        th.save(model.state_dict(), "model.pt")
        model.eval()
        with th.no_grad():
            model.inference(g, nfeat, device, args.batch_size, args.num_workers)
        if proc_id == 0:
            print('Avg epoch time: {}'.format(avg / (epoch - 4)))

def main(args):
    devices = list(map(int, args.gpu.split(',')))
    n_gpus = len(devices)

#     num_nodes, num_edges, feat_dim = 100, 300, 30

#     edge_index = th.randint(0, num_nodes, (2, num_edges))
#     x = th.randn(num_nodes, feat_dim)
#     y = th.randint(0, 2, (num_nodes,))
    
#     g = dgl.graph((edge_index[0],edge_index[1]))
#     n_classes = 2
    
#     # set train/val/test mask in node_classification task
#     train_mask = th.zeros(num_nodes).bool()
#     train_mask[0 : int(0.3 * num_nodes)] = True
#     val_mask = th.zeros(num_nodes).bool()
#     val_mask[int(0.3 * num_nodes) : int(0.7 * num_nodes)] = True
#     test_mask = th.zeros(num_nodes).bool()
#     test_mask[int(0.7 * num_nodes) :] = True


    # load dataset
#     if args.dataset == 'reddit':
#         g, n_classes = load_reddit(self_loop=False)
#     elif args.dataset == 'ogbn-products':
#         g, n_classes = load_ogb('ogbn-products')
#     elif args.dataset == 'paypal':
#         glist, label_dict = dgl.load_graphs("/home/binwu5/summer-intern-2022-graph/newbid.bin")
#         g = glist[0]
#         n_classes = 2
#     else:
#         raise Exception('unknown dataset')

#     pt = th.load('/home/binwu5/cogdl-master/examples/simple_hgn/paypal.pt')
    
    glist, label_dict = dgl.load_graphs("/home/binwu5/summer-intern-2022-graph/new_graph_etype.bin")
    g = glist[0]
    # g.edata['h'] = th.zeros(g.num_edges(), 1)
    n_classes = 128 # for embedding size 
    
    num_nodes = g.num_nodes()
    train_mask = th.zeros(num_nodes).bool()
    train_mask[0 : int(0.3 * num_nodes)] = True
    val_mask = th.zeros(num_nodes).bool()
    val_mask[int(0.3 * num_nodes) : int(0.7 * num_nodes)] = True
    test_mask = th.zeros(num_nodes).bool()
    test_mask[int(0.7 * num_nodes) :] = True
    
    cust_feature = np.load("cust_feature_pca.npy")
    x = th.cat((th.tensor(cust_feature).float(), th.zeros(num_nodes - len(cust_feature), 128)),0)
    print("feat size", x.shape)
    y = th.randint(0, n_classes, (num_nodes,))
        
    # edge2type = np.load("edge2type.npy", allow_pickle=True).item()
    print("data loaded")
    train_nid = train_mask
    val_nid = val_mask
    test_nid = test_mask

    
    nfeat = x
    labels = y

    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves memory and CPU.
    g.create_formats_()

    # this to avoid competition overhead on machines with many cores.
    # Change it to a proper number on your machine, especially for multi-GPU training.
    os.environ['OMP_NUM_THREADS'] = str(mp.cpu_count() // 2 // n_gpus)

    # Pack data
    data = train_nid, val_nid, test_nid, n_classes, g, nfeat, labels
    
    if devices[0] == -1:
        assert args.graph_device == 'cpu', \
               f"Must have GPUs to enable {args.graph_device} sampling."
        assert args.data_device == 'cpu', \
               f"Must have GPUs to enable {args.data_device} feature storage."
        run(0, 0, args, ['cpu'], data)
    elif n_gpus == 1:
        run(0, n_gpus, args, devices, data)
    else:
        mp.spawn(run, args=(n_gpus, args, devices, data), nprocs=n_gpus)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument("--gpu", type=str, default='0',
                           help="GPU, can be a list of gpus for multi-gpu training,"
                                " e.g., 0,1,2,3; -1 for CPU")
    argparser.add_argument('--dataset', type=str, default='paypal',
                           choices=('paypal', 'reddit', 'ogbn-products'))
    argparser.add_argument('--num-epochs', type=int, default=5)
    argparser.add_argument('--num-hidden', type=int, default=64)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--num-negs', type=int, default=1)
    argparser.add_argument('--neg-share', default=False, action='store_true',
                           help="sharing neg nodes for positive nodes")
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=100000)
    argparser.add_argument('--log-every', type=int, default=1)
    argparser.add_argument('--eval-every', type=int, default=1)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--edge-feats', type=int, default=64)
    argparser.add_argument('--num-edges', type=int, default=18)
    argparser.add_argument('--slope', type=float, default=0.05)
    argparser.add_argument('--inference', type=bool, default=True)
    argparser.add_argument('--num-workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--graph-device', choices=('cpu', 'gpu', 'uva'), default='cpu',
                           help="Device to perform the sampling. "
                                "Must have 0 workers for 'gpu' and 'uva'")
    argparser.add_argument('--data-device', choices=('cpu', 'gpu', 'uva'), default='cpu',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "Use 'cpu' to keep the features on host memory and "
                                "'uva' to enable UnifiedTensor (GPU zero-copy access on "
                                "pinned host memory).")
    args = argparser.parse_args()

    main(args)