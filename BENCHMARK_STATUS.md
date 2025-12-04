# Benchmark Status - All Algorithms Running

## Current Status: ✅ TRAINING IN PROGRESS

The benchmark script (`run_benchmark.py`) is currently training all three algorithms on your dataset.

## What's Happening

1. **U-Net** - Training in progress (5 epochs)
2. **TransUNet** - Will start after U-Net completes
3. **Swin-UNet** - Will start after TransUNet completes

## Expected Timeline

- **Per Model**: ~20-40 minutes on CPU (5 epochs, batch size 2)
- **Total Time**: ~1-2 hours for all 3 models
- **On GPU**: Would be ~10-15 minutes per model

## How to Check Progress

Run:
```bash
python check_progress.py
```

Or check checkpoints directory:
```bash
ls checkpoints/
```

## Final Results Location

Once training completes, results will be saved to:
- `results/final_benchmark_results.json` - Detailed results
- `results/final_benchmark_summary.csv` - Summary table
- Console output will show the **WINNER** with best Dice score

## What Will Be Compared

1. **Test Dice Score** (primary metric) - Higher is better
2. **Test HD95** (Hausdorff Distance) - Lower is better  
3. **Training Time** - Efficiency metric
4. **Model Parameters** - Model size
5. **Inference Time** - Speed metric
6. **Per-Organ Performance** - Detailed breakdown

## Winner Determination

The algorithm with the **highest Test Dice Score** will be declared the winner.

---

*Training is running in the background. The script will automatically:*
1. *Train each model*
2. *Evaluate on test set*
3. *Compare all models*
4. *Display final results with winner*

*Check back in 1-2 hours or monitor progress with `python check_progress.py`*

