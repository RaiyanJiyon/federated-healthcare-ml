# Table6 Robustness

| Attack Scenario               |   Byzantine Clients | Fraction   |   AUROC |   Recall |   AUROC Loss (%) | Status      |
|:------------------------------|--------------------:|:-----------|--------:|---------:|-----------------:|:------------|
| Clean (Baseline)              |                   0 | 0%         |  0.885  |    0.417 |              0   | ✓ Baseline  |
| Light Attack (1/7 Byzantine)  |                   1 | 14.3%      |  0.8618 |    0.489 |              2.6 | ✓ Resilient |
| Severe Attack (2/7 Byzantine) |                   2 | 28.6%      |  0.8268 |    0.412 |              6.6 | ⚠ Degraded  |

