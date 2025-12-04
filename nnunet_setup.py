"""
Setup script for nnU-Net integration with Synapse dataset
nnU-Net requires a specific dataset format
"""
import os
import shutil
import numpy as np
import nibabel as nib
from pathlib import Path


def convert_synapse_to_nnunet_format(synapse_dir, output_dir, task_name='Task001_Synapse'):
    """
    Convert Synapse dataset to nnU-Net format
    
    nnU-Net expects:
    - Dataset folder: nnUNet_raw_data_base/nnUNet_raw_data/TaskXXX_Name/
    - Images: imagesTr/, imagesTs/
    - Labels: labelsTr/
    - Naming: case_XXXX_0000.nii.gz (XXXX is case number, 0000 is modality)
    """
    output_task_dir = os.path.join(output_dir, task_name)
    os.makedirs(output_task_dir, exist_ok=True)
    
    images_tr_dir = os.path.join(output_task_dir, 'imagesTr')
    labels_tr_dir = os.path.join(output_task_dir, 'labelsTr')
    images_ts_dir = os.path.join(output_task_dir, 'imagesTs')
    
    os.makedirs(images_tr_dir, exist_ok=True)
    os.makedirs(labels_tr_dir, exist_ok=True)
    os.makedirs(images_ts_dir, exist_ok=True)
    
    # Training cases: 1-18
    for case_id in range(1, 19):
        case_dir = os.path.join(synapse_dir, f'case{case_id:04d}')
        img_path = os.path.join(case_dir, 'img.nii.gz')
        label_path = os.path.join(case_dir, 'label.nii.gz')
        
        if os.path.exists(img_path) and os.path.exists(label_path):
            # Copy and rename
            shutil.copy(img_path, os.path.join(images_tr_dir, f'case_{case_id:04d}_0000.nii.gz'))
            shutil.copy(label_path, os.path.join(labels_tr_dir, f'case_{case_id:04d}.nii.gz'))
    
    # Test cases: 22-33
    for case_id in range(22, 34):
        case_dir = os.path.join(synapse_dir, f'case{case_id:04d}')
        img_path = os.path.join(case_dir, 'img.nii.gz')
        
        if os.path.exists(img_path):
            shutil.copy(img_path, os.path.join(images_ts_dir, f'case_{case_id:04d}_0000.nii.gz'))
    
    print(f'Converted Synapse dataset to nnU-Net format in {output_task_dir}')
    print(f'Training cases: 18')
    print(f'Test cases: 12')


def create_dataset_json(output_dir, task_name='Task001_Synapse'):
    """Create dataset.json file required by nnU-Net"""
    import json
    
    dataset_json = {
        "channel_names": {
            "0": "CT"
        },
        "labels": {
            "background": 0,
            "aorta": 1,
            "gallbladder": 2,
            "left_kidney": 3,
            "right_kidney": 4,
            "liver": 5,
            "pancreas": 6,
            "spleen": 7,
            "stomach": 8
        },
        "numTraining": 18,
        "file_ending": ".nii.gz",
        "name": "Synapse",
        "description": "Synapse multi-organ CT segmentation dataset",
        "reference": "Synapse dataset",
        "licence": "See dataset website",
        "release": "1.0",
        "tensorImageSize": "3D",
        "modality": {
            "0": "CT"
        },
        "training": [
            {
                "image": f"./imagesTr/case_{i:04d}_0000.nii.gz",
                "label": f"./labelsTr/case_{i:04d}.nii.gz"
            }
            for i in range(1, 19)
        ],
        "test": [
            f"./imagesTs/case_{i:04d}_0000.nii.gz"
            for i in range(22, 34)
        ]
    }
    
    task_dir = os.path.join(output_dir, task_name)
    json_path = os.path.join(task_dir, 'dataset.json')
    
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=2)
    
    print(f'Created dataset.json at {json_path}')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert Synapse dataset to nnU-Net format')
    parser.add_argument('--synapse_dir', type=str, required=True,
                        help='Path to Synapse dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to nnU-Net raw data base directory')
    parser.add_argument('--task_name', type=str, default='Task001_Synapse',
                        help='Task name for nnU-Net')
    
    args = parser.parse_args()
    
    # Convert dataset
    convert_synapse_to_nnunet_format(args.synapse_dir, args.output_dir, args.task_name)
    
    # Create dataset.json
    create_dataset_json(args.output_dir, args.task_name)
    
    print('\nNext steps:')
    print('1. Set environment variables:')
    print('   export nnUNet_raw_data_base="/path/to/nnUNet_raw_data_base"')
    print('   export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"')
    print('   export RESULTS_FOLDER="/path/to/nnUNet_trained_models"')
    print('2. Run nnU-Net preprocessing:')
    print('   nnUNet_plan_and_preprocess -d 1 --verify_dataset_integrity')
    print('3. Train nnU-Net:')
    print('   nnUNet_train 3d_fullres nnUNetTrainerV2 1 0')


if __name__ == '__main__':
    main()

