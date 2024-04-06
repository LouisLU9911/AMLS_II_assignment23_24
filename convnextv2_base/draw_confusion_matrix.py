import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

with open("y.txt", "r") as f:
    y = [int(line) for line in f.readlines()]

with open("y_pred.txt", "r") as f:
    y_pred = [int(line) for line in f.readlines()]

labels = [
    "Cassava Bacterial Blight (CBB)",
    "Cassava Brown Streak Disease (CBSD)",
    "Cassava Green Mottle (CGM)",
    "Cassava Mosaic Disease (CMD)",
    "Healthy",
]

cm = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels,
)
disp.plot()

plt.savefig("test.png")
