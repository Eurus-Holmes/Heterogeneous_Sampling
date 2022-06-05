"""Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Reference Code: https://github.com/tkipf/relational-gcn
"""
import argparse
import itertools
from operator import index
import numpy as np
import time
import torch as th
import torch.nn.functional as F
import torchmetrics.functional as MF

import dgl
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from model import EntityClassify, RelGraphEmbed

def extract_embed(node_embed, input_nodes):
    emb = {}
    for ntype, nid in input_nodes.items():
        print(ntype, nid)
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


def sample_blocks(self, g, seed_nodes, exclude_eids=None):
    output_nodes = seed_nodes
    blocks = []
    for fanout in reversed(self.fanouts):
        frontier = g.sample_neighbors(
            seed_nodes, fanout, edge_dir=self.edge_dir, prob=self.prob,
            replace=self.replace, output_device=self.output_device,
            exclude_edges=exclude_eids)
        eid = frontier.edata[EID]
        block = to_block(frontier, seed_nodes)
        block.edata[EID] = eid
        seed_nodes = block.srcdata[NID]
        blocks.insert(0, block)

    return seed_nodes, output_nodes, blocks

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
    train_mask = g.nodes[category].data.pop('train_mask')
    test_mask = g.nodes[category].data.pop('test_mask')
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
    labels = g.nodes[category].data.pop('labels')

    # split dataset into train, validate, test
    if args.validation:
        val_idx = train_idx[:len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
    else:
        val_idx = train_idx

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

    num_parts = 10


    # print(g.ntypes)
    # print(g.etypes)
    # print(g.canonical_etypes)
    print(g)

    for relation in g.canonical_etypes:
        # e.g., ('Forschungsgebiete', 'ontology#dealtWithIn', 'Projekte')
        one_relation_subgraph = dgl.edge_type_subgraph(g, [relation]) # 1. 这里处理要只留下relation里有target的东西
        # 2. 注意这个矩阵的dimension，没有被transpose过
        # 3. 这里要检查每个relation的index
        print(one_relation_subgraph)
        print(labels)



        # print(sg.nodes[relation[0]].data) # feature is also copied
        # print(sg)
        # print(relation[2])
        #   hsg = dgl.to_homogeneous(one_relation_subgraph)
        hsg = dgl.to_homogeneous(one_relation_subgraph) # error, because not all the nodes have the label


        #   这里的顺序不对，这里应该保存所有的relation_sub_graph， 但是所有的label nodes都应该用的一样的

        print(hsg)
        print(hsg.ndata)

        # train sampler
        sampler = dgl.dataloading.ClusterGCNSampler(
            hsg, num_parts)
        loader = dgl.dataloading.DataLoader(
            hsg, th.arange(num_parts), sampler,
            batch_size=args.batch_size, shuffle=True, num_workers=0)

        # validation sampler
        # we do not use full neighbor to save computation resources
        val_sampler = dgl.dataloading.ClusterGCNSampler(
            hsg, num_parts)
        val_loader = dgl.dataloading.DataLoader(
            hsg, th.arange(num_parts), val_sampler,
            batch_size=args.batch_size, shuffle=True, num_workers=0)

        # optimizer
        all_params = itertools.chain(model.parameters(), embed_layer.parameters())
        optimizer = th.optim.Adam(all_params, lr=args.lr, weight_decay=args.l2norm)



        # training loop
        print("start training...")
        dur = []
        for epoch in range(args.n_epochs):
            model.train()
            optimizer.zero_grad()
            if epoch > 3:
                t0 = time.time()
            for it, sg in enumerate(loader):
                sg = sg.to(device)

                # 4. 出来的homogeneous的cluster要通过一个mapping在原来heterogeneous的graph里是不是有target node
                # 5. how to combine the resuls from multiple cluster-gcn sampler
                # 6. 保持target nodes的index是同一批，然后取出来embedding aggregate叠在一起
                # 7. 对于不一样的邻居，cluster-gcn 会return不一样的结果，我们通过majority-voting的方式来确定唯一的cluster
                # Original node types
                print(sg.ndata[dgl.NTYPE])
                # Original type-specific node IDs
                print(sg.ndata[dgl.NID])

                # Original edge types
                # print(sg.edata[dgl.ETYPE])
                # Original type-specific edge IDs
                # print(sg.edata[dgl.EID])
                # print(category)


                input_nodes = one_relation_subgraph.ndata[dgl.NID]
                ### input nodes should have the same size as the seed_nodes
                # print(input_nodes)

                seeds = sg.ndata[dgl.NID]
                batch_tic = time.time()
                emb = extract_embed(node_embed, input_nodes)
                lbl = labels[seeds]
                print(lbl)
                # print(sg.ndata[dgl.NTYPE])
                if use_cuda:
                    emb = {k : e.cuda() for k, e in emb.items()}
                    lbl = lbl.cuda()
                logits = model(emb, sg)[category]
                loss = F.cross_entropy(logits, lbl)
                loss.backward()
                optimizer.step()

                train_acc = th.sum(logits.argmax(dim=1) == lbl).item() / len(seeds)
                print("Epoch {:05d} | Batch {:03d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Time: {:.4f}".
                      format(epoch, it, train_acc, loss.item(), time.time() - batch_tic))



                # print(it)
                # print(sg.edata)
                ################ cluster-gcn #################
                # x = sg.ndata['feat']
                # y = sg.ndata['label']
                # m = sg.ndata['train_mask'].bool()
                # y_hat = model(sg, x)
                # loss = F.cross_entropy(y_hat[m], y[m])
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                # if it % 20 == 0:
                #     acc = MF.accuracy(y_hat[m], y[m])
                #     mem = th.cuda.max_memory_allocated() / 1000000
                #     print('Loss', loss.item(), 'Acc', acc.item(), 'GPU Mem', mem, 'MB')

            ################ rgcn-hetero #################
            # for i, (input_nodes, seeds, blocks) in enumerate(loader):
            #     blocks = [blk.to(device) for blk in blocks]
            #     seeds = seeds[category]     # we only predict the nodes with type "category"
            #     batch_tic = time.time()
            #     emb = extract_embed(node_embed, input_nodes)
            #     lbl = labels[seeds]
            #     if use_cuda:
            #         emb = {k : e.cuda() for k, e in emb.items()}
            #         lbl = lbl.cuda()
            #     logits = model(emb, blocks)[category]
            #     loss = F.cross_entropy(logits, lbl)
            #     loss.backward()
            #     optimizer.step()

            #     train_acc = th.sum(logits.argmax(dim=1) == lbl).item() / len(seeds)
            #     print("Epoch {:05d} | Batch {:03d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Time: {:.4f}".
            #           format(epoch, i, train_acc, loss.item(), time.time() - batch_tic))

            if epoch > 3:
                dur.append(time.time() - t0)

            # val_loss, val_acc = evaluate(model, val_loader, node_embed, labels, category, device)
            # print("Epoch {:05d} | Valid Acc: {:.4f} | Valid loss: {:.4f} | Time: {:.4f}".
            #       format(epoch, val_acc, val_loss, np.average(dur)))
        print()
        if args.model_path is not None:
            th.save(model.state_dict(), args.model_path)

        output = model.inference(
            g, args.batch_size, 'cuda' if use_cuda else 'cpu', 0, node_embed)
        test_pred = output[category][test_idx]
        test_labels = labels[test_idx].to(test_pred.device)
        test_acc = (test_pred.argmax(1) == test_labels).float().mean()
        print("Test Acc: {:.4f}".format(test_acc))
        print()

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
