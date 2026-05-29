# Multi-Dataset Comparison Figures

## Generated Publication-Ready Visualizations

### Figure 1: AUROC Performance Comparison
- **Left panel:** Centralized baseline AUROC (MIMIC-IV: 0.8816 vs eICU-CRD: 0.8441)
- **Right panel:** Federated FedAvg AUROC with loss percentages
- **Key insight:** 0.38% loss on MIMIC-IV, 1.23% loss on eICU-CRD (both well below 3% target)

### Figure 2: Expected Calibration Error (ECE) Before & After Platt Scaling
- **Left panel:** Original model ECE (shows calibration need)
- **Right panel:** After Platt scaling (both datasets achieve ECE < 0.02 target)
- **Key insight:** Platt scaling reliably improves calibration across datasets

### Figure 3: Federated Learning Performance Degradation Analysis
- **Bar chart:** AUROC loss percentage for each dataset
- **Green dashed line:** 3% target threshold
- **Key insight:** Both datasets achieve <3% federated loss (0.38% vs 1.23%)

### Figure 4: Comprehensive Performance Summary Table
- **Structured comparison:** All key metrics side-by-side
- **Color-coded:** MIMIC-IV (blue), eICU-CRD (orange), differences (gray)
- **Checkmarks:** Indicates successful target achievement

## Usage in Manuscript

These figures should be included in the Results section:
- Figure 1 → Replace existing AUROC comparison
- Figure 2 → New calibration subsection
- Figure 3 → Federated robustness demonstration
- Figure 4 → Comprehensive results summary

All figures are saved to `results/plots/figures/` in high-resolution PNG format (300 DPI).
