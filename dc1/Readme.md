# JBG040 Advanced Data Challenge  
## Thoracic Disease Classification from Chest X-ray Images

This project builds and evaluates a deep learning pipeline for multi-class thoracic disease classification from chest X-ray images. The system is designed so that all experiments, evaluations, and Grad-CAM analyses can be launched from a single entry point, main.py.
---

## Project Overview

Chest X-ray interpretation is an important but challenging task in medical imaging. In this project, we train convolutional neural networks to classify thoracic diseases from X-ray images and compare different training strategies.

The experiments are organized into three main stages:

1. **Optimizer Experiment**  
   Compare different optimizers and select the best one.

2. **Imbalance Handling Experiment**  
   Based on the best optimizer, test different class imbalance strategies.

3. **Threshold Selection Experiment**  
   Based on the best previous settings, optimize class decision thresholds.

After these experiments, the project also includes:

- **Experiment Evaluation** for comparing saved models and reporting results
- **Grad-CAM Analysis** for visual explanation and model comparison

---

## Main Design

The project uses `main.py` as the **central controller**.

Running `main.py` will automatically connect all experiment and evaluation scripts so that the complete workflow can be executed from one command.

This means:

- all experiments are launched in sequence
- results are saved automatically
- evaluation is performed automatically
- Grad-CAM analysis can also be generated automatically

So the project supports a **single entry point** for the full pipeline.

---

## Project Structure

```text
dc1/
│
├── main.py
├── main_train_val_monitor.py
├── main_experiment_optimizer.py
├── main_experiment_imbalance.py
├── main_experiment_threshold.py
├── experiment_evaluation.py
├── run_gradcam_experiment.py
├── gradcam.py
├── net.py
├── train_test.py
├── image_dataset.py
├── batch_sampler.py
│
├── check_data/                             # Data inspection and label analysis utilities
│   ├── __init__.py
│   ├── data_check.py
│   ├── data_check_update.py
│   ├── data_mapping.py
│   ├── OutputOfLabels.py
│   └── Distribution of image labels.png
│
├── data/                                   # Prepared NumPy datasets
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── Y_train.npy
│   └── Y_test.npy
│
├── experiments/                            # Saved training runs for each experiment stage
│   ├── experiment_optimizer/
│   │   ├── adam/
│   │   ├── adamw/
│   │   ├── sgd/
│   │   └── comparison/
│   ├── experiment_imbalance/
│   │   ├── experiment_no_imbalance_adamw/
│   │   ├── experiment_balanced_batch_adamw/
│   │   ├── experiment_severity_weighted_loss_adamw/
│   │   └── comparison/
│   └── experiment_threshold/
│       ├── balanced_batch_adamw/
│       ├── severity_weighted_loss_adamw/
│       └── comparison/
│
├── artifacts/                              # Generated reports and selected baseline outputs
│   ├── experiment_evaluation_result/
│   │   └── eval_<timestamp>/
│   │       ├── ranking.csv
│   │       ├── aggregate_summary.txt
│   │       ├── aggregate_summary.json
│   │       ├── experiment_optimizer/
│   │       ├── experiment_imbalance/
│   │       └── experiment_threshold/
│   ├── baseline_best/
│   └── baseline_final/
│
└── gradcam_result_experiment/              # Grad-CAM visualizations and comparison summaries
    ├── comparisons/
    ├── experiment_optimizer/
    ├── experiment_imbalance/
    ├── experiment_threshold/
    └── batch_gradcam_summary_<timestamp>.csv