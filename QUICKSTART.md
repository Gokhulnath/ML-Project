# Quick Start Guide

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Download the Synapse multi-organ CT dataset from [Synapse Platform](https://www.synapse.org/)
2. Organize it in the following structure:
```
your_data_directory/
├── case0001/
│   ├── img.nii.gz          # CT scan volume (NIfTI format)
│   └── label.nii.gz        # Segmentation mask (NIfTI format)
├── case0002/
│   ├── img.nii.gz
│   └── label.nii.gz
...
├── case0033/
│   ├── img.nii.gz
│   └── label.nii.gz
```

**Key Requirements:**
- Format: NIfTI compressed (`.nii.gz`)
- Naming: `case0001`, `case0002`, ..., `case0033` (zero-padded to 4 digits)
- Files: Each case must have `img.nii.gz` and `label.nii.gz`
- Location: Can be anywhere; you'll specify the path when training

See [DATASET_SETUP.md](DATASET_SETUP.md) for complete details.

## Training Models

### 1. U-Net (Baseline CNN)
```bash
python train.py --model unet \
  --data_dir /path/to/synapse_data \
  --config configs/unet_config.yaml \
  --batch_size 4 \
  --epochs 150
```

### 2. TransUNet (Hybrid CNN-Transformer)
```bash
python train.py --model transunet \
  --data_dir /path/to/synapse_data \
  --config configs/transunet_config.yaml \
  --batch_size 4 \
  --epochs 150
```

### 3. Swin-UNet (Pure Transformer)
```bash
python train.py --model swinunet \
  --data_dir /path/to/synapse_data \
  --config configs/swinunet_config.yaml \
  --batch_size 4 \
  --epochs 150
```

## Evaluation

Evaluate a single model:
```bash
python evaluate.py --model unet \
  --checkpoint checkpoints/unet/best_model.pth \
  --data_dir /path/to/synapse_data \
  --split test
```

## Model Comparison

Compare all trained models:
```bash
python compare_models.py \
  --data_dir /path/to/synapse_data \
  --checkpoints \
    unet:checkpoints/unet/best_model.pth \
    transunet:checkpoints/transunet/best_model.pth \
    swinunet:checkpoints/swinunet/best_model.pth \
  --split test
```

## nnU-Net Setup

For nnU-Net, you need to convert the dataset first:

```bash
# Convert dataset format
python nnunet_setup.py \
  --synapse_dir /path/to/synapse_data \
  --output_dir /path/to/nnUNet_raw_data_base/nnUNet_raw_data

# Set environment variables
export nnUNet_raw_data_base="/path/to/nnUNet_raw_data_base"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export RESULTS_FOLDER="/path/to/nnUNet_trained_models"

# Preprocess
nnUNet_plan_and_preprocess -d 1 --verify_dataset_integrity

# Train
nnUNet_train 3d_fullres nnUNetTrainerV2 1 0
```

## Expected Results

After training, you should see:
- Checkpoints saved in `checkpoints/{model_name}/`
- Evaluation results in `results/`
- Comparison tables and visualizations

## Key Features

- **Preprocessing**: Automatic intensity normalization, resampling, and augmentation
- **Training**: Combined Dice + Cross-Entropy loss, cosine annealing LR schedule
- **Evaluation**: Dice score and Hausdorff distance (95th percentile) per organ
- **Comparison**: Side-by-side model comparison with statistical analysis

## Troubleshooting

1. **CUDA out of memory**: Reduce batch size in config files
2. **Dataset not found**: Check that your data directory structure matches the expected format
3. **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`

