# Heterogeneous Sampling

## Requirements 

* PyTorch 1.0+
* requests
* rdflib

```
pip install dgl-cu101 dglgo -f https://data.dgl.ai/wheels/repo.html
pip install requests torch rdflib pandas
```

Example code was tested with rdflib 4.2.2 and pandas 0.23.4


## Datasets

The preprocessing is slightly different from the author's code. We directly load and preprocess
raw RDF data. For AIFB, BGS and AM,
all literal nodes are pruned from the graph. For AIFB, some training/testing nodes
thus become orphan and are excluded from the training/testing set. The resulting graph
has fewer entities and relations. As a reference (numbers include reverse edges and relations):

| Dataset | #Nodes | #Edges | #Relations | #Labeled |
| --- | --- | --- | --- | --- |
| AIFB | 8,285 | 58,086 | 90 | 176 |
| AIFB-hetero | 7,262 | 48,810 | 78 | 176 |
| MUTAG | 23,644 | 148,454 | 46 | 340 |
| MUTAG-hetero | 27,163 | 148,100 | 46 | 340 |
| BGS | 333,845 | 1,832,398 | 206 | 146 |
| BGS-hetero | 94,806 | 672,884 | 96 | 146 |
| AM | 1,666,764 | 11,976,642 | 266 | 1000 |
| AM-hetero | 881,680 | 5,668,682 | 96 | 1000 |


## Demo

Check [demo](https://github.com/Eurus-Holmes/Heterogeneous_Sampling/tree/main/code/demo) or [Google Colab](https://colab.research.google.com/drive/1yaMufnRZMcV2rV07blhjbCFV8XFgc3S8?usp=sharing) for more results.


## Usage

> Please put sampling code at `dgl/examples/pytorch/rgcn-hetero/`

For node-wise sampling:

```
python NodeSampler.py -d aifb --testing --gpu 0 --fanout=8
python NodeSampler.py -d mutag --l2norm 5e-4 --testing --gpu 0 --fanout=8
python NodeSampler.py -d bgs --l2norm 5e-4 --n-bases 40 --testing --gpu 0
python NodeSampler.py -d am --l2norm 5e-4 --n-bases 40 --testing --gpu 0  --fanout=16 --batch-size 50
```

For layer-wise sampling:

```
python LayerSampler.py -d aifb --testing --gpu 0 --fanout=8
python LayerSampler.py -d mutag --l2norm 5e-4 --testing --gpu 0 --fanout=8
python LayerSampler.py -d bgs --l2norm 5e-4 --n-bases 40 --testing --gpu 0
python LayerSampler.py -d am --l2norm 5e-4 --n-bases 40 --testing --gpu 0  --fanout=16 --batch-size 50
```

For subgraph-wise sampling (ShaDowKHopSampler):

```
python ShaDowKHopSampler.py -d aifb --testing --gpu 0 --fanout=8
python ShaDowKHopSampler.py -d mutag --l2norm 5e-4 --testing --gpu 0 --fanout=8
python ShaDowKHopSampler.py -d bgs --l2norm 5e-4 --n-bases 40 --testing --gpu 0
python ShaDowKHopSampler.py -d am --l2norm 5e-4 --n-bases 40 --testing --gpu 0  --fanout=16 --batch-size 50
```

For subgraph-wise sampling (ClusterGCNSampler):

See [method2_cluster-gcn](https://github.com/Eurus-Holmes/Heterogeneous_Sampling/tree/main/code/method2_cluster-gcn).


### Instruction

```
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
```



