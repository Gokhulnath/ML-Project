"""
Setup script for the medical image segmentation project
"""
from setuptools import setup, find_packages

setup(
    name='medical-image-segmentation',
    version='1.0.0',
    description='Comparing CNN and Transformer Architectures for Medical Image Segmentation',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.1',
        'torchvision>=0.15.0',
        'monai>=1.2.0',
        'nibabel>=5.0.0',
        'SimpleITK>=2.3.0',
        'albumentations>=1.3.0',
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'scikit-image>=0.20.0',
        'wandb>=0.15.0',
        'timm>=0.9.0',
        'einops>=0.6.0',
        'tqdm>=4.65.0',
        'pandas>=2.0.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'pyyaml>=6.0',
    ],
    python_requires='>=3.8',
)

