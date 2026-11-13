# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

SCAD is a research implementation of a lightweight multi-interest next-item recommendation model (PyTorch). Dependencies: Python 3, `torch>=2.0`, `d2l` (used for `d2l.DotProductAttention`). There is no package config, linter, or unit-test suite; `src/test.py` is the evaluation step, not a test suite.

## Environment

Conda env `scad` (Miniconda at `~/miniconda3`, spec in `environment.yml`): Python 3.10, torch 2.5.1+cu121, torchvision 0.20.1 (d2l imports it), d2l 1.0.3 (pins numpy 1.23.5, so Python ≤3.11). Activate with `source ~/miniconda3/bin/activate scad`.

## Running

Run from the **repository root** as `python3 src/main.py ...`. Paths are built relative to the CWD (`./datasets/<name>/...`, `./save_model/`), while imports are flat (`from model.SCAD import SCAD`, `from train import train`) and resolve because the script's directory `src/` is on `sys.path`. The README's `python3 main.py` only works if those relative paths line up. `save_model/` does not exist in the repo, so create it before training or `torch.save` fails.

```bash
mkdir -p save_model
python3 src/main.py --p train --dataset lfm        # train with early stopping, then evaluate on the test split
python3 src/main.py --p test  --dataset yoochoose  # evaluate only; needs save_model/<model_name>.pth
```

- For `--dataset yoochoose` or `lfm`, `main.py` **ignores the hyperparameter CLI flags** and uses hardcoded values: batch 64, GRU 2 layers, dims 128, seq_len 6, lr 1e-3, topN 5, and `k=3` (yoochoose) or `k=6` (lfm). The `--batch_size/--k/--lr/...` flags only apply to other dataset names.
- The checkpoint name is `{dataset}_SCAD_{topn}_k{k}.pth`, so test mode must use the same topN and k as training.
- Training: up to 100 epochs. It keeps the checkpoint with the best validation NDCG and stops after 3 epochs with no improvement in HR or NDCG. Validation metrics are appended to `./process.txt` and test metrics to `./note.txt`.
- Training is slow because `SCAD.forward` loops over the batch in Python and moves tensors between CPU and GPU.

## Data pipeline (`src/utils.py`)

Each dataset directory needs two comma-separated files:
- `user_item_8.csv`: `user_id,item1..item8`
- `user_item_neigh.csv`: `user_id,neigh_id,item1..item8`, with exactly **5 neighbor rows per user**

Implicit assumptions:
- Neighbors are grouped by user through dict insertion order and then zipped positionally with `user_item_8.csv`, so both files must list users in the same order.
- The split is 80/10/10 by file order, with no shuffling before the split.
- The first 7 items are the input sequence and item 8 is the target. Neighbor sequences are also truncated to 7 items.
- Raw item IDs are remapped to contiguous indices. The output layer scores **all** items through the shared embedding matrix.
- `seq_len` (6) must equal the number of neighbors plus 1, because the user vector is stacked on top of the 5 neighbor vectors before routing.
- `datasets/yoochoose/yoo_{train,valid,test}.txt` are not used by the code.

## Model (`src/model/SCAD.py`)

Per sample, `forward(user_seq [B,7], neigh_seqs [B,5,7])`:
1. A shared item embedding feeds a shared GRU. `usr_add_attention` (additive attention against the last hidden state) pools the user sequence into one vector.
2. Each neighbor sequence goes through the same GRU. The user vector is concatenated along time, and the result is pooled the same way. This yields a `[6,128]` matrix: the user row plus 5 neighbor rows.
3. `Self_attention` followed by `Add_norm` (in `modules.py`) is applied over those 6 rows.
4. Capsule-style dynamic routing (3 iterations, with `squashing` and target attention `tar_att1`):
   - `single_dynamic_routing` produces one interest vector.
   - `multi_dynamic_routing` produces `k` interest vectors, initialized with `torch.randn`, so evaluation is not deterministic.
5. Scoring: `score = yita * (single @ E^T) + (1 - yita) * max_k(multi @ E^T)`, where `yita` is a learnable scalar. Training uses `CrossEntropyLoss` over all items. Metrics are HR@N and NDCG@N (`utils.metrics`).

`device` is defined at module level in each file, with `CUDA_VISIBLE_DEVICES=0` hardcoded. `forward` returns scores on the CPU.

`SCAD-master.ipynb` is an earlier, self-contained notebook version (class `GRUModel`, which also computes MRR, with `basic_path='./denny/'`). The `src/` scripts are the maintained version.
