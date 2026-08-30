# ML Architecture Comparison

PyTorch vs TensorFlow vs Scikit-learn experiments. Scratch CNN to fine-tuning pretrained models on Intel's scene classification dataset.

## What's being compared

### Scratch CNN
`src/pytorch/train_scratch.py`, `src/tensorflow/train_scratch.py`

A CNN built from scratch in each framework, swept over a small hyperparameter grid. Params consist of learning rate, conv layer depth, base filters, and dropout.

Same architecture on both sides: a stack of Conv2D + BatchNorm (+ReLU/MaxPool) blocks doubling filter count each layer, flattened into a dense layer with dropout before the final classification layer. 81-run W&B sweep over `learning_rate × num_conv_layers × base_filters × dropout`.

### Transfer learning
`src/pytorch/train_transfer.py`, `src/tensorflow/train_transfer.py`

Three fine-tuning strategies across a handful of pretrained backbones (ResNet18/50/101, MobileNetV3-S/L, EfficientNet-B0/B3):

- **Frozen** — linear probing, backbone weights untouched
- **Gradual** — staged unfreezing, last N layers unfrozen in sequence after a val-acc plateau
- **Finetune** — full fine-tune, whole model unfrozen

Staged unfreezing takes inspiration from ULMFiT's progressive unfreezing (Howard & Ruder, 2018), adapted for backbones with far fewer layers than ULMFiT targeted. Instead of one layer per epoch, the last 3 layer groups are unfrozen in sequence, each run to early stopping.

### Sklearn on frozen features
`src/pytorch/train_sklearn.py`

Feature vectors extracted from a pretrained ResNet50, fed into classic classifiers: SVM (RBF), XGBoost, Logistic Regression, KNN, Random Forest.

## Dataset

[Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) (Kaggle) — 6 scene classes: buildings, forest, glacier, mountain, sea, street.

- `data/raw/` - the raw Kaggle dataset, one folder per class
- `src/organize_splits.py` - stratified 80/20 train/val split out of `data/raw/`, copied into `data/split/`
- `data/split/`, `data/split_medium/`, `data/split_small/` - full, half, and quarter-sized versions of the dataset, used in the transfer learning sweep (`data_size: full/medium/small`) to test how much dataset size mattered. Excluded from the writeup. The models really only "learned something real" on the full dataset, the smaller sizes weren't interesting enough to show.

## Project structure

```
src/
  pytorch/       — scratch CNN, transfer learning, sklearn feature extraction (PyTorch)
  tensorflow/    — scratch CNN, transfer learning (TensorFlow / Keras)
  api/           — FastAPI inference server
hf-space/        — Docker deployment for the Hugging Face Space
data/            — raw images + train/val/test splits
```

## Setup

```bash
pip install -r requirements.txt

# torch / torchvision — install separately
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# tensorflow with GPU — WSL2 only on Windows
pip install tensorflow[and-cuda]
```

TensorFlow dropped native GPU (CUDA) support on Windows a while back. As of this project (mid-2026) it still only runs on GPU via WSL2 or Linux directly, not native Windows. I heard this fact will change soon. PyTorch does not have this restriction.

## Running it

Training runs are driven by [W&B sweeps](https://docs.wandb.ai/guides/sweeps/). Each script defines its own sweep config and launches the full grid:

```bash
wandb login

python -m src.pytorch.train_scratch      # 81-run scratch CNN sweep
python -m src.pytorch.train_transfer     # transfer learning sweep (model x strategy x data_size)
python -m src.pytorch.train_sklearn      # sklearn classifiers on frozen ResNet50 features

python -m src.tensorflow.train_scratch
python -m src.tensorflow.train_transfer
```

Local inference:

```bash
uvicorn src.api.main:app --reload
# POST an image to /predict
```

## Key findings

- Best scratch CNN (PyTorch): **88.8%** test accuracy (6 conv layers, 32 base filters, dropout 0.7, lr 5e-4)
- Transfer learning (PyTorch), best per strategy: frozen ResNet50 **90.8%** -> staged unfreezing ResNet101 **93.6%** -> full fine-tune ResNet101 **93.6%**. Staged unfreezing got nearly all the benefit of full fine-tuning without touching every layer.
- On identical hyperparameters, PyTorch consistently outperformed TensorFlow by a wide margin. PyTorch reached close to 90%, TensorFlow stayed in the high-70s/low-80s. My best guess is that the differences in weight initialization, optimizer implementation details, and GPU tensor handling.
- sklearn classifiers on **frozen** ResNet50 features nearly matched fine-tuning the whole network: SVM 93.3%, XGBoost 92.8%, LogReg 91.9%, KNN 91.9%, RandomForest 91.4%. All within a point or two of the best fine-tuned model, without updating a single backbone weight.
- Most common confusion across all model types: **glacier <-> mountain**. Second most common: **building <-> street**. Buildings are frequently photographed from street level, so the "correct" label is often genuinely ambiguous.

## Limitations

All transfer learning runs share one fixed hyperparameter set (carried over from the best scratch CNN config), rather than a per-model search. So the ranking above says which model wins *under that config*, not which model has the best ceiling. A proper per-model grid search (learning rate, optimizer, dropout, classifier head size, etc.) would likely shift the ordering, but would turn into hundreds of more runs for what's already a wide comparison.

## Deployment

Inference is served via FastAPI, containerized with Docker, and deployed as a Hugging Face Space. The portfolio frontend calls the Space's `/predict` endpoint directly.

## Further research

- Per-model hyperparameter search instead of one shared config
- Deeper tuning of the sklearn classifiers themselves
- More optimizer/criterion/last-layer variations. I focused on the highest-impact knobs (learning rate, dropout) and left the rest unexplored
