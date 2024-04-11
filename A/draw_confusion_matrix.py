#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Confusion Matrix"""

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    classification_report,
)

# Load real labels of the base model
with open("y.txt", "r") as f:
    y = [int(line) for line in f.readlines()]

# Load predicted labels of the base model
with open("y_pred.txt", "r") as f:
    y_pred = [int(line) for line in f.readlines()]

labels = [
    "Cassava Bacterial Blight (CBB)",
    "Cassava Brown Streak Disease (CBSD)",
    "Cassava Green Mottle (CGM)",
    "Cassava Mosaic Disease (CMD)",
    "Healthy",
]

print(classification_report(y, y_pred))

cm = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    # display_labels=labels,
)
disp.plot()
plt.tight_layout()
plt.savefig("confusion_matrix_of_base_model.png")
