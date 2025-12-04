"""
Script to verify dataset structure and format
"""
import os
import argparse
import glob
import numpy as np


def verify_dataset_npz(data_dir):
    """Verify NPZ format dataset structure and files"""
    print(f"Verifying dataset at: {data_dir}")
    print("=" * 60)
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"ERROR: Directory does not exist: {data_dir}")
        return False
    
    train_npz_dir = os.path.join(data_dir, 'Synapse', 'train_npz')
    test_h5_dir = os.path.join(data_dir, 'Synapse', 'test_vol_h5')
    
    # Check training data
    if not os.path.exists(train_npz_dir):
        print(f"ERROR: Training data directory not found: {train_npz_dir}")
        return False
    
    npz_files = glob.glob(os.path.join(train_npz_dir, '*.npz'))
    print(f"\nFound {len(npz_files)} NPZ files in training directory")
    
    # Group by case
    case_files = {}
    for npz_file in npz_files:
        filename = os.path.basename(npz_file)
        case_id = filename.split('_')[0]
        if case_id not in case_files:
            case_files[case_id] = []
        case_files[case_id].append(npz_file)
    
    found_cases = sorted(case_files.keys())
    print(f"Found {len(found_cases)} unique cases")
    print(f"Cases: {found_cases[:10]}{'...' if len(found_cases) > 10 else ''}")
    
    # Verify file structure
    print("\n" + "=" * 60)
    print("Verifying NPZ file structure...")
    
    valid_files = 0
    invalid_files = []
    
    # Sample a few files to verify
    sample_files = npz_files[:min(10, len(npz_files))]
    for npz_file in sample_files:
        try:
            data = np.load(npz_file)
            if 'image' not in data or 'label' not in data:
                invalid_files.append((os.path.basename(npz_file), 'Missing image or label key'))
                continue
            
            img = data['image']
            label = data['label']
            
            if img.shape != label.shape:
                invalid_files.append((os.path.basename(npz_file), f'Shape mismatch: img={img.shape}, label={label.shape}'))
                continue
            
            if len(img.shape) != 2:
                invalid_files.append((os.path.basename(npz_file), f'Expected 2D image, got shape: {img.shape}'))
                continue
            
            valid_files += 1
            
        except Exception as e:
            invalid_files.append((os.path.basename(npz_file), f'Error loading: {str(e)}'))
    
    print(f"Valid files (sampled): {valid_files}/{len(sample_files)}")
    if invalid_files:
        print(f"Invalid files: {len(invalid_files)}")
        for filename, error in invalid_files[:5]:
            print(f"  {filename}: {error}")
    
    # Dataset split info
    print("\n" + "=" * 60)
    print("Dataset Split Information:")
    train_cases = found_cases[:15] if len(found_cases) >= 15 else found_cases
    val_cases = found_cases[15:18] if len(found_cases) >= 18 else found_cases[-3:] if len(found_cases) >= 3 else []
    test_cases = found_cases[18:] if len(found_cases) > 18 else []
    
    print(f"  Training cases:   {len(train_cases)} cases")
    print(f"  Validation cases: {len(val_cases)} cases")
    print(f"  Test cases:       {len(test_cases)} cases")
    
    # Count slices per split
    train_slices = sum(len(case_files[c]) for c in train_cases)
    val_slices = sum(len(case_files[c]) for c in val_cases)
    test_slices = sum(len(case_files[c]) for c in test_cases)
    
    print(f"\n  Training slices:   {train_slices}")
    print(f"  Validation slices: {val_slices}")
    print(f"  Test slices:       {test_slices}")
    
    # Check test data
    if os.path.exists(test_h5_dir):
        h5_files = glob.glob(os.path.join(test_h5_dir, '*.h5'))
        print(f"\n  Test H5 volumes:    {len(h5_files)} (note: H5 format not yet supported)")
    
    # Overall status
    print("\n" + "=" * 60)
    if valid_files == len(sample_files) and len(found_cases) >= 15:
        print("[OK] Dataset structure is CORRECT!")
        print("  You can proceed with training.")
        return True
    else:
        print("[WARNING] Dataset structure has some issues.")
        if valid_files < len(sample_files):
            print("  Some files failed validation.")
        if len(found_cases) < 15:
            print(f"  Only {len(found_cases)} cases found, expected at least 15.")
        return False


def main():
    parser = argparse.ArgumentParser(description='Verify Synapse dataset structure')
    parser.add_argument('--data_dir', type=str, default='./dataset',
                        help='Path to dataset directory')
    
    args = parser.parse_args()
    
    # Convert to absolute path
    data_dir = os.path.abspath(args.data_dir)
    
    verify_dataset_npz(data_dir)


if __name__ == '__main__':
    main()
