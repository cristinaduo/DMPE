# DMPE
DMPE: A Dual-branch Molecular Property Encapsulation Framework with Kolmogorov- Arnold Networks

This document provides a minimal, reliable workflow to set up the environment and run training, prediction, and hyperparameter search.


### Installation

- Recommended Python: 3.9–3.10
- Minimal packages (core):
  - python, torch, tensorboard, rdkit, scikit-learn, hyperopt, numpy, pandas

Conda (recommended for RDKit on macOS/Linux):
```bash
conda create -n dmpe python=3.10 -y
conda activate dmpe
```

Install PyTorch (choose the command for your platform/CUDA):
```bash
# CPU-only example
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
# For CUDA, pick the matching wheels from: https://pytorch.org/get-started/locally/
```

Install RDKit via conda-forge:
```bash
conda install -c conda-forge rdkit=2022.09.5 -y
```

Install core Python packages:
```bash
pip install numpy==1.26.4 pandas==2.2.2 scikit-learn==1.5.2 tensorboard==2.17.0 hyperopt==0.2.7
```

Optional (if you plan to use visualization utilities):
```bash
pip install matplotlib==3.8.4 seaborn==0.13.2 umap-learn==0.5.6 cairosvg==2.7.1
```

Alternatively, see `environment.txt` as a baseline and adjust versions as needed.


### Repository structure (key files)
```
DMPE/
  Data/                      # Example datasets
  fpgnn/                     # Library code (data, model, tools, training)
  model_save/                # Model checkpoints (create if missing)
  log/                       # TensorBoard logs (create if missing)
  result/                    # Prediction outputs (create if desired)
  train.py                   # Training entry point
  predict.py                 # Inference entry point
  hyper_opti.py              # Hyperparameter optimization
```
Create required directories if they do not exist:
```bash
mkdir -p model_save log result
```
Note: `model_save/` is not tracked by default. Create it locally to store checkpoints.


### Quick start

#### Train a model
Classification example on MoleculeNet bace:
```bash
python train.py \
  --data_path Data/MoleculeNet/bace.csv \
  --dataset_type classification \
  --save_path model_save/bace \
  --log_path log/bace
```
Notes:
- `--save_path` will contain subfolders by random seed, e.g., `model_save/bace/Seed_0/model.pt`.
- Track training via TensorBoard:
```bash
tensorboard --logdir log/bace --port 6006
```

#### Predict with a trained model
```bash
python predict.py \
  --predict_path test.csv \
  --model_path model_save/bace/Seed_0/model.pt \
  --result_path result.csv
```
- `test.csv` should match the expected format (including a `smiles` column and required features/targets as defined by the code).
- The script writes predictions to `result.csv`.

#### Hyperparameter optimization
```bash
python hyper_opti.py \
  --data_path Data/MoleculeNet/bace.csv \
  --dataset_type classification \
  --save_path model_save/hyper \
  --log_path log/hyper
```


### Data format
- Input CSVs typically include at least a `smiles` column and one or more target columns.
- Set `--dataset_type` to `classification` or `regression` accordingly.
  

### Citation
If you use DMPE or its components in your research, please cite this repository. A formal citation entry will be added upon publication.


### License
This project is released for research purposes. See the repository LICENSE if provided; otherwise, contact the authors for licensing details.

