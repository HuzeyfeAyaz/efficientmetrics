# EfficientMetrics

EfficientMetrics is a Python package for efficient calculation of classification metrics and skicit-learn's classification_report.

## Installation

```bash
pip install efficientmetrics
```

## Sample Usage

```python
from efficientmetrics import EfficientMetrics
import numpy as np

y_true = np.array([0, 1, 2, 2, 0, 1])
y_preds = np.array([0, 2, 2, 2, 0, 0])
classes = np.array([0, 1, 2])

eff_metrics = EfficientMetrics(y_true, y_preds)
eff_metrics.calculate_confusion_matrix()
eff_metrics.classification_report()
print(eff_metrics.confmat)
print(eff_metrics.report)
```
