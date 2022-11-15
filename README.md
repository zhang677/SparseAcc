# SparseAcc
A cache simulator for SpGEMM private-cache accelerator

# Run Compiler

```
python train.py --dataset pubmed --epochs 20 --train --path ../trace/model.pt
python trace.py --model gcn --path ../trace/model.pt --dataset pubmed
python train.py --dataset pubmed --path ../trace/model.pt
```

