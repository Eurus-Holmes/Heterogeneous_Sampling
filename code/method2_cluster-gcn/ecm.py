"""Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Reference Code: https://github.com/tkipf/relational-gcn
"""
import argparse
import itertools
import numpy as np
import time
import torch as th
import torch.nn.functional as F

import dgl
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from model import EntityClassify, RelGraphEmbed

def extract_embed(node_embed, input_nodes):
    emb = {}
    for ntype, nid in input_nodes.items():
        nid = input_nodes[ntype]
        emb[ntype] = node_embed[ntype][nid]
    return emb

def evaluate(model, loader, node_embed, labels, category, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    count = 0
    for input_nodes, seeds, blocks in loader:
        blocks = [blk.to(device) for blk in blocks]
        seeds = seeds[category]
        emb = extract_embed(node_embed, input_nodes)
        emb = {k: e.to(device) for k, e in emb.items()}
        lbl = labels[seeds].to(device)
        logits = model(emb, blocks)[category]
        loss = F.cross_entropy(logits, lbl)
        acc = th.sum(logits.argmax(dim=1) == lbl).item()
        total_loss += loss.item() * len(seeds)
        total_acc += acc
        count += len(seeds)
    return total_loss / count, total_acc / count

def main(args):
    # check cuda
    device = 'cpu'
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(args.gpu)
        device = 'cuda:%d' % args.gpu

    # load graph data
    if args.dataset == 'aifb':
        dataset = AIFBDataset()
    elif args.dataset == 'mutag':
        dataset = MUTAGDataset()
    elif args.dataset == 'bgs':
        dataset = BGSDataset()
    elif args.dataset == 'am':
        dataset = AMDataset()
    else:
        raise ValueError()

    g = dataset[0]
    category = dataset.predict_category
    num_classes = dataset.num_classes
    train_mask = g.nodes[category].data['train_mask']#.pop('train_mask')
    test_mask = g.nodes[category].data['test_mask']#.pop('test_mask')
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
    labels = g.nodes[category].data['labels']#.pop('labels')

    #ids = g.nodes[category].data[dgl.NID]
    #print(ids)
    #a = 35 / 0
    # split dataset into train, validate, test
    if args.validation:
        val_idx = train_idx[:len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
        print(val_idx)
    else:
        val_idx = train_idx
    #print(train_idx)
    #print(test_idx)

    #print(train_mask)
    
    #g.ndata['train_mask'] = th.ones(g.num_nodes(), 1)
    for ntype in g.ntypes:
      if ntype == category:
        continue
      g.nodes[ntype].data['train_mask'] = th.ones(g.num_nodes(ntype), dtype=th.uint8)

    #g.nodes[category].data['train_mask'] = train_mask
    #train_mask2 = g.nodes[category].data['train_mask']

    #print(train_mask2)
    #a = 35 / 0

    for ntype in g.ntypes:
      if ntype == category:
        continue
      g.nodes[ntype].data['test_mask'] = th.ones(g.num_nodes(ntype), dtype=th.uint8)

    #print(g.nodes[category].data['labels'])
    #a = 5 / 0
    for ntype in g.ntypes:
      if ntype == category:
        continue
      g.nodes[ntype].data['labels'] = th.ones(g.num_nodes(ntype), dtype=th.int64)

    #train_dictionary = dict(zip(train_idx, g.ndata[dgl.NID][category]))
    #print(train_dictionary[th.tensor(5)]) #Expecting 9

    # create embeddings
    embed_layer = RelGraphEmbed(g, args.n_hidden)

    if not args.data_cpu:
        labels = labels.to(device)
        embed_layer = embed_layer.to(device)

    node_embed = embed_layer()
    # create model
    model = EntityClassify(g,
                           args.n_hidden,
                           num_classes,
                           num_bases=args.n_bases,
                           num_hidden_layers=args.n_layers - 2,
                           dropout=args.dropout,
                           use_self_loop=args.use_self_loop)

    if use_cuda:
        model.cuda()

    # Note: each 'id' or 'eid' corresponds to the specific node typpe
    g.edata['eid'] = g.edata[dgl.EID]
    g.ndata['id'] = g.ndata[dgl.NID]
    # print(g.ndata[dgl.NID])
    # print("----------------------------")
    # sg = dgl.node_subgraph(g,{category: range(0, 237)})
    # print(sg.ndata['id'])
    # print(g.ndata['id'])
    # a = 35/0


    hg = dgl.to_homogeneous(g, ndata=['id', 'train_mask', 'test_mask', 'labels'], edata=['eid'])
    # print(hg.ndata['id'])
    # print(len(hg.ndata['id']))
    # print(len(set(hg.ndata['id'])))
    # print(hg.ndata[dgl.NID])
    # print("----------------------------")
    # print(hg.edata['eid'])
    # print(hg.edata[dgl.EID])
    # a = 433252/ 0

    num_parts = args.num_parts
    cache_path_str = str(num_parts) + '_' + args.dataset + '_' + str(args.cluster_size) + '.pkl'
    cluster_gcn_sampler = dgl.dataloading.ClusterGCNSampler(
          hg, num_parts, cache_path=cache_path_str)
    cluster_gcn_loader = dgl.dataloading.DataLoader(
          hg, th.arange(num_parts), cluster_gcn_sampler,
          batch_size=args.cluster_size, shuffle=True, drop_last=False, num_workers=4)
    #print(hg)
    count = 0
    #print(g)

    for hsg in cluster_gcn_loader:
      #print(hsg.ndata['id'])
      #sg = dgl.to_heterogeneous(hsg, g.ntypes, g.etypes)
      #print(sg)
      # print(sg.ndata[dgl.NID])
      # print(sg.ndata['id'])
      #sg.ndata[dgl.NID] = sg.ndata['id']
      #sg.edata[dgl.EID] = sg.edata['eid']
      # print(sg.ndata[dgl.NID])
      count += 1
      #print(dgl.node_subgraph(g,{category: train_idx}))
      #a = 11663 / 0
    print("Count of subgraphs:", count)
    #a = 35 / 0
 
    

    # bs = args.batch_size #TODO: change this
    # # train sampler
    # sampler = dgl.dataloading.MultiLayerNeighborSampler([args.fanout] * args.n_layers)
    # loader = dgl.dataloading.DataLoader(
    #     g, {category: train_idx}, sampler,
    #     batch_size=bs, shuffle=False, num_workers=0)

    # # validation sampler
    # # we do not use full neighbor to save computation resources
    # val_sampler = dgl.dataloading.MultiLayerNeighborSampler([args.fanout] * args.n_layers)
    # val_loader = dgl.dataloading.DataLoader(
    #     g, {category: val_idx}, val_sampler,
    #     batch_size=27, shuffle=False, num_workers=0)

    # optimizer
    all_params = itertools.chain(model.parameters(), embed_layer.parameters())
    optimizer = th.optim.Adam(all_params, lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    t0 = time.time()
    bt = []
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        optimizer.zero_grad()
        if epoch > 3:
            t0 = time.time()


        #print(g.ntypes)
        #a = 5 / 0
        for hsg in cluster_gcn_loader:
            #print(hsg.ndata['id'])
            sg = dgl.to_heterogeneous(hsg, g.ntypes, g.etypes)
            #print(sg)
            # print(sg.ndata[dgl.NID])
            # print(sg.ndata['id'])
            sg.ndata[dgl.NID] = sg.ndata['id']
            sg.edata[dgl.EID] = sg.edata['eid']
            # print(sg.ndata[dgl.NID])
            count += 1
            #print(dgl.node_subgraph(g,{category: train_idx}))
            #a = 11663 / 0

            sg_train_mask = sg.nodes[category].data['train_mask']#.pop('train_mask')
            sg_test_mask = sg.nodes[category].data['test_mask']#.pop('test_mask')
            sg_train_idx = th.nonzero(sg_train_mask, as_tuple=False).squeeze()
            sg_test_idx = th.nonzero(sg_test_mask, as_tuple=False).squeeze()
            sg_labels = sg.nodes[category].data.pop('labels')#['labels']

            bs =  sg.num_nodes(category)#TODO: change this?
            # train sampler
            sampler = dgl.dataloading.MultiLayerNeighborSampler([args.fanout] * args.n_layers)
            loader = dgl.dataloading.DataLoader(
                sg, {category: sg_train_idx}, sampler,
                batch_size=bs, shuffle=False, num_workers=0)

            # validation sampler
            # we do not use full neighbor to save computation resources
            val_sampler = dgl.dataloading.MultiLayerNeighborSampler([args.fanout] * args.n_layers)
            val_loader = dgl.dataloading.DataLoader(
                g, {category: val_idx}, val_sampler,
                batch_size=args.batch_size , shuffle=False, num_workers=0)

            for i, (input_nodes, seeds, blocks) in enumerate(loader):
                blocks = [blk.to(device) for blk in blocks]
                seeds = seeds[category]     # we only predict the nodes with type "category"
                batch_tic = time.time()
                emb = extract_embed(node_embed, input_nodes)
                lbl = sg_labels[seeds]
                if use_cuda:
                    emb = {k : e.cuda() for k, e in emb.items()}
                    lbl = lbl.cuda()
                logits = model(emb, blocks)[category]
                loss = F.cross_entropy(logits, lbl)
                loss.backward()
                optimizer.step()

                train_acc = th.sum(logits.argmax(dim=1) == lbl).item() / len(seeds)
                bt.append(time.time() - batch_tic)
                print("Epoch {:05d} | Batch {:03d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Time: {:.4f}".
                    format(epoch, i, train_acc, loss.item(), time.time() - batch_tic))

        if epoch > 3:
            dur.append(time.time() - t0)

        val_loss, val_acc = evaluate(model, val_loader, node_embed, labels, category, device)
        print("Epoch {:05d} | Valid Acc: {:.4f} | Valid loss: {:.4f} | Time: {:.4f}".
              format(epoch, val_acc, val_loss, np.average(dur)))
    print()
    if args.model_path is not None:
        th.save(model.state_dict(), args.model_path)

    t5 = time.time()
    print("Training time: : {:.4f}s | Batch time: : {:.4f}s".format((t5-t0), np.mean(bt)))

    output = model.inference(
        g, bs, 'cuda' if use_cuda else 'cpu', 0, node_embed)
    test_pred = output[category][test_idx]
    test_labels = labels[test_idx].to(test_pred.device)
    test_acc = (test_pred.argmax(1) == test_labels).float().mean()
    print("Test Acc: {:.4f}".format(test_acc))
    print()
    t1 = time.time()
    print("Total time: : {:.4f}s | Batch time: : {:.4f}s".format((t1-t0), np.mean(bt)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=20,
            help="number of training epochs")
    parser.add_argument("-x", "--num-parts", type=int, default=6,
            help="number of parts of partitions for Cluster-GCN")
    parser.add_argument("-y", "--cluster-size", type=int, default=3,
            help="number of partitions in each cluster in Cluster-GCN")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--model_path", type=str, default=None,
            help='path for save the model')
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    parser.add_argument("--batch-size", type=int, default=100,
            help="Mini-batch size. If -1, use full graph training.")
    parser.add_argument("--fanout", type=int, default=4,
            help="Fan-out of neighbor sampling.")
    parser.add_argument('--data-cpu', action='store_true',
            help="By default the script puts all node features and labels "
                 "on GPU when using it to save time for data copy. This may "
                 "be undesired if they cannot fit in GPU memory at once. "
                 "This flag disables that.")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    main(args)
