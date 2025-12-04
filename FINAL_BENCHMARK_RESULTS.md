# Final Benchmark Results - Medical Image Segmentation

## Algorithms Tested

1. **U-Net** - Pure CNN baseline (encoder-decoder with skip connections)
2. **TransUNet** - Hybrid CNN-Transformer (ResNet-50 encoder + Vision Transformer + CNN decoder)
3. **Swin-UNet** - Pure Transformer (hierarchical shifted window attention)

## Dataset

- **Dataset**: Synapse Multi-Organ CT Segmentation
- **Training**: 1,827 slices (15 cases)
- **Validation**: 384 slices (3 cases)  
- **Test**: Validation set used for testing (test H5 files need conversion)
- **Organs**: 8 organs (aorta, gallbladder, left/right kidney, liver, pancreas, spleen, stomach)

## Results

*Results will be displayed here once training completes. The benchmark script is currently running...*

### Expected Output Format:

| Model | Parameters | Training Time | Val Dice | Test Dice | Test HD95 (mm) | Inference (ms/img) |
|-------|------------|---------------|----------|-----------|----------------|-------------------|
| U-Net | ~17M | XX min | X.XXXX | X.XXXX | XX.XX | XX.XX |
| TransUNet | ~XXM | XX min | X.XXXX | X.XXXX | XX.XX | XX.XX |
| Swin-UNet | ~XXM | XX min | X.XXXX | X.XXXX | XX.XX | XX.XX |

### Per-Organ Performance:

| Organ | U-Net | TransUNet | Swin-UNet |
|-------|-------|-----------|-----------|
| Aorta | X.XXXX | X.XXXX | X.XXXX |
| Gallbladder | X.XXXX | X.XXXX | X.XXXX |
| Left Kidney | X.XXXX | X.XXXX | X.XXXX |
| Right Kidney | X.XXXX | X.XXXX | X.XXXX |
| Liver | X.XXXX | X.XXXX | X.XXXX |
| Pancreas | X.XXXX | X.XXXX | X.XXXX |
| Spleen | X.XXXX | X.XXXX | X.XXXX |
| Stomach | X.XXXX | X.XXXX | X.XXXX |

## Winner

🏆 **BEST ALGORITHM**: [Will be determined after training completes]

**Criterion**: Highest Test Dice Score

---

*Note: Training is in progress. Check progress with: `python check_progress.py`*
*Final results will be saved to: `results/final_benchmark_results.json`*

