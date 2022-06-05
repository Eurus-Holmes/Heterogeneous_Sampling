# Sampling for Heterogeneous GNNs

## Abstract

Graph sampling is a popular technique in training large-scale graph neural networks (GNNs), recent sampling-based methods have demonstrated impressive success for homogeneous graphs. However, in practice, the interaction between different entities is often different based on their relationship, i.e., the network in reality is mostly heterogeneous. But only a few of the recent works have paid attention to sampling methods on heterogeneous graphs. In this work, we aim to study sampling for heterogeneous GNNs. We propose two general pipelines for heterogeneous sampling. Based on the proposed pipeline, we evaluate 3 representative sampling methods on heterogeneous graphs, including node-wise sampling, layer-wise sampling, and subgraph-wise sampling. To the best of our knowledge, we are the first to provide a thorough implementation, evaluation, and discussion of each sampling method on heterogeneous graphs. Extensive experiments compared sampling methods from multiple aspects and highlight their characteristics for each category. Evaluation of scalability on larger-scale heterogeneous graphs also shows we achieve the trade-off between efficiency and effectiveness. Last, we also analyze the limitations of our proposed pipeline on heterogeneous sub-graph sampling and provide a detailed comparison with HGSampling.




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


To evaluate the scalability of sampling methods on larger-scale heterogeneous graphs, we also include [OGBN-MAG](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag) dataset, including four types of entities—papers (736,389 nodes), authors (1,134,649 nodes), institutions (8,740 nodes), and fields of study (59,965 nodes).




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




## Acknowledgements

We would like to thank Yewen (Emily) Wang, Prof. [Yizhou Sun](https://web.cs.ucla.edu/~yzsun/) and [DGL](https://github.com/dmlc/dgl) community for helpful discussions and comments.

