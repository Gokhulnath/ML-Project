# Dataset Setup Guide

## Dataset Format

The Synapse dataset should be in **NIfTI format** (`.nii.gz` files).

## Directory Structure

Place your dataset in the following structure:

```
your_data_directory/
├── case0001/
│   ├── img.nii.gz          # CT scan volume
│   └── label.nii.gz        # Segmentation mask
├── case0002/
│   ├── img.nii.gz
│   └── label.nii.gz
├── case0003/
│   ├── img.nii.gz
│   └── label.nii.gz
...
├── case0033/
│   ├── img.nii.gz
│   └── label.nii.gz
```

## File Naming Convention

- **Case directories**: Must be named `case0001`, `case0002`, ..., `case0033` (zero-padded to 4 digits)
- **Image files**: Must be named `img.nii.gz` (inside each case directory)
- **Label files**: Must be named `label.nii.gz` (inside each case directory)

## Dataset Split

The code automatically splits the dataset as follows:
- **Training**: cases 1-18 (`case0001` to `case0018`)
- **Validation**: cases 19-21 (`case0019` to `case0021`)
- **Test**: cases 22-33 (`case0022` to `case0033`)

## File Format Details

### Image File (`img.nii.gz`)
- **Format**: NIfTI compressed (.nii.gz)
- **Type**: 3D CT volume
- **Data type**: Float32 (Hounsfield Units)
- **Dimensions**: Variable (will be resampled to 1.5mm isotropic)

### Label File (`label.nii.gz`)
- **Format**: NIfTI compressed (.nii.gz)
- **Type**: 3D segmentation mask
- **Data type**: Integer labels
- **Label values**:
  - `0`: Background
  - `1`: Aorta
  - `2`: Gallbladder
  - `3`: Left Kidney
  - `4`: Right Kidney
  - `5`: Liver
  - `6`: Pancreas
  - `7`: Spleen
  - `8`: Stomach

## Where to Place the Dataset

You can place the dataset **anywhere** on your system. When running training/evaluation scripts, provide the path using the `--data_dir` argument.

### Example Locations:

**Option 1: Inside project directory (not recommended for large datasets)**
```
aml/
├── data/
│   └── synapse_data/       # Your dataset here
│       ├── case0001/
│       ├── case0002/
│       ...
├── models/
├── training/
...
```

**Option 2: Outside project directory (recommended)**
```
/home/user/datasets/
└── synapse_data/           # Your dataset here
    ├── case0001/
    ├── case0002/
    ...
```

**Option 3: On external drive or network storage**
```
/mnt/datasets/synapse_data/  # Your dataset here
    ├── case0001/
    ├── case0002/
    ...
```

## Usage Example

When training, specify the path to your dataset:

```bash
python train.py --model unet \
  --data_dir /path/to/your/synapse_data \
  --config configs/unet_config.yaml
```

For example:
```bash
# If dataset is at /home/user/datasets/synapse_data
python train.py --model unet \
  --data_dir /home/user/datasets/synapse_data \
  --config configs/unet_config.yaml

# If dataset is at ./data/synapse_data (relative path)
python train.py --model unet \
  --data_dir ./data/synapse_data \
  --config configs/unet_config.yaml
```

## Verification

To verify your dataset structure is correct, you can check:

1. **Count cases**: Should have 33 case directories (case0001 to case0033)
2. **Check files**: Each case directory should have exactly 2 files:
   - `img.nii.gz`
   - `label.nii.gz`
3. **Test loading**: The dataset loader will print warnings if files are missing

## Converting from Other Formats

If you have the dataset in a different format, you may need to:

1. **Convert DICOM to NIfTI**: Use tools like `dcm2niix` or `SimpleITK`
2. **Rename files**: Ensure files follow the naming convention above
3. **Organize structure**: Create case directories and move files accordingly

### Example conversion script (if needed):

```python
import os
import shutil
import nibabel as nib
from pathlib import Path

def organize_dataset(source_dir, target_dir):
    """Organize dataset into required structure"""
    os.makedirs(target_dir, exist_ok=True)
    
    # Assuming source files are named differently
    # Adjust this based on your source format
    for i in range(1, 34):
        case_dir = os.path.join(target_dir, f'case{i:04d}')
        os.makedirs(case_dir, exist_ok=True)
        
        # Copy/convert your source files
        # Adjust paths based on your source structure
        src_img = os.path.join(source_dir, f'case{i}_img.nii.gz')
        src_label = os.path.join(source_dir, f'case{i}_label.nii.gz')
        
        if os.path.exists(src_img):
            shutil.copy(src_img, os.path.join(case_dir, 'img.nii.gz'))
        if os.path.exists(src_label):
            shutil.copy(src_label, os.path.join(case_dir, 'label.nii.gz'))
```

## Troubleshooting

**Issue**: "Warning: Missing files for case X"
- **Solution**: Check that case directory exists and contains both `img.nii.gz` and `label.nii.gz`

**Issue**: "File not found" errors
- **Solution**: Verify the `--data_dir` path is correct and accessible

**Issue**: "Invalid NIfTI file"
- **Solution**: Ensure files are valid NIfTI format. You can verify with:
  ```python
  import nibabel as nib
  img = nib.load('path/to/img.nii.gz')
  print(img.shape, img.header)
  ```

