Please put the files at dgl/examples/pytorch/rgcn-hetero/

Usage examples:
```
python ecm_aifb.py -d aifb --testing --gpu 0 --fanout=8 --n-epochs=15
python ecm_mutag.py -d mutag --l2norm 5e-4 --n-bases 30 --testing --gpu 0 --batch-size=50 --fanout=-1
python ecm_bgs.py -d bgs --l2norm 5e-4 --n-bases 40 --testing --gpu 0 --fanout=16
python ecm_am.py -d am --l2norm 5e-4 --n-bases 40 --testing --gpu 0 --fanout=-1
```
or:
```
python ecm.py -d am --l2norm 5e-4 --n-bases 40 --testing --gpu 0 --fanout=-1 --num-parts 27 --cluster-size 3
```
