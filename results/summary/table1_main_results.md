# Table1 Main Results

| Model                       |   AUROC |   Brier Score | ECE    |   Recall |   Precision | Clinical Status   |
|:----------------------------|--------:|--------------:|:-------|---------:|------------:|:------------------|
| Centralized LR              |  0.8920 |        0.0617 | 0.0088 |    0.852 |       0.302 | ✓ Safe            |
| FedAvg (Baseline)           |  0.8784 |        0.0617 | 0.0088 |    0.435 |       0.698 | ✓ Safe            |
| FedProx (μ=0.01)            |  0.8591 |        0.0841 | 0.0832 |    0.380 |       0.630 | ✗ Underperforms   |
| With DP-SGD (ε=4.36, δ=1e-5)         |  0.8477 |        0.3874 | —      |    0.381 |       0.654   | ✓ Utility Preserved |
| With Byzantine Attack (1/7) |  0.8618 |        0.0617 | 0.0088 |    0.489 |       0.645 | ⚠ Degraded        |

