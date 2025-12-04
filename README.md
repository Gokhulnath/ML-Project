# Comparing CNN and Transformer Architectures for Medical Image Segmentation

This project implements and compares four state-of-the-art architectures for medical image segmentation:
- **U-Net**: Pure CNN baseline
- **nnU-Net**: Optimized CNN pipeline
- **TransUNet**: Hybrid CNN-Transformer
- **Swin-UNet**: Pure Transformer

## Dataset

The Synapse multi-organ CT dataset with 8 organs:
- Aorta
- Gallbladder
- Left Kidney
- Right Kidney
- Liver
- Pancreas
- Spleen
- Stomach

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── models/          # Model architectures
├── data/            # Dataset handling and preprocessing
├── training/        # Training scripts
├── evaluation/      # Evaluation metrics and visualization
├── configs/         # Configuration files
└── utils/           # Utility functions
```

## Dataset Setup

1. Download the Synapse multi-organ CT dataset from [Synapse Platform](https://www.synapse.org/)
2. Organize the dataset in the following structure:
   ```
   your_data_directory/
   ├── case0001/
   │   ├── img.nii.gz          # CT scan volume
   │   └── label.nii.gz        # Segmentation mask
   ├── case0002/
   │   ├── img.nii.gz
   │   └── label.nii.gz
   ...
   ├── case0033/
   │   ├── img.nii.gz
   │   └── label.nii.gz
   ```

**Important Details:**
- **Format**: NIfTI compressed files (`.nii.gz`)
- **Naming**: Case directories must be `case0001`, `case0002`, ..., `case0033` (zero-padded)
- **Files**: Each case directory must contain `img.nii.gz` and `label.nii.gz`
- **Split**: Training (cases 1-18), Validation (cases 19-21), Test (cases 22-33)
- **Location**: Dataset can be placed anywhere; specify path with `--data_dir` argument

For detailed setup instructions, see [DATASET_SETUP.md](DATASET_SETUP.md)

## Usage

### Training

Train individual models:

```bash
# U-Net
python train.py --model unet --data_dir /path/to/synapse_data --config configs/unet_config.yaml

# TransUNet
python train.py --model transunet --data_dir /path/to/synapse_data --config configs/transunet_config.yaml

# Swin-UNet
python train.py --model swinunet --data_dir /path/to/synapse_data --config configs/swinunet_config.yaml
```

### Evaluation

Evaluate a trained model:

```bash
python evaluate.py --model unet --checkpoint checkpoints/unet/best_model.pth --data_dir /path/to/synapse_data
```

### Model Comparison

Compare all trained models:

```bash
python compare_models.py --data_dir /path/to/synapse_data \
  --checkpoints unet:checkpoints/unet/best_model.pth \
                transunet:checkpoints/transunet/best_model.pth \
                swinunet:checkpoints/swinunet/best_model.pth
```

### nnU-Net Setup

To use nnU-Net, first convert the dataset:

```bash
python nnunet_setup.py --synapse_dir /path/to/synapse_data --output_dir /path/to/nnUNet_raw_data_base/nnUNet_raw_data
```

Then follow nnU-Net's standard training procedure.

## Results

Results will be saved in `results/` directory with:
- Per-organ Dice scores and Hausdorff distances
- Summary statistics
- Comparison visualizations
- JSON files with detailed metrics

## Model Architectures

- **U-Net**: Classic encoder-decoder CNN with skip connections
- **TransUNet**: ResNet-50 encoder + Vision Transformer + CNN decoder
- **Swin-UNet**: Pure Transformer using shifted window attention
- **nnU-Net**: Self-configuring framework (requires separate setup)

