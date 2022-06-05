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




## Usage

Take `LayerSampler` as an example:

```
python LayerSampler.py -d aifb --testing --gpu 0 --fanout=8
python LayerSampler.py -d mutag --l2norm 5e-4 --testing --gpu 0 --fanout=8
python LayerSampler.py -d bgs --l2norm 5e-4 --n-bases 40 --testing --gpu 0
python LayerSampler.py -d am --l2norm 5e-4 --n-bases 40 --testing --gpu 0  --fanout=16 --batch-size 50
```

## Demo

Check [demo](https://github.com/Eurus-Holmes/Heterogeneous_Sampling/tree/main/code/demo) or [Google Colab](https://colab.research.google.com/drive/1yaMufnRZMcV2rV07blhjbCFV8XFgc3S8?usp=sharing) for more results.




