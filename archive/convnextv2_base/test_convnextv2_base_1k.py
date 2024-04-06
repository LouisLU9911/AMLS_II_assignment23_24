import os

from datasets import load_dataset
from sklearn.metrics import confusion_matrix, classification_report
from transformers import pipeline

BATCH_SIZE = 180
NUM_TRAIN_EPOCHS = 20
NUM_WORKERS = 15
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
PIPELINE_DEVICE = int(CUDA_VISIBLE_DEVICES.split(",")[0])

cwd = os.getcwd()
train_image_folder = os.path.join(cwd, "Datasets", "imagefolder")

print(f"begin loading {train_image_folder} ...")
dataset = load_dataset("imagefolder", data_dir=train_image_folder)
print("dataset setup successfully!")

model_checkpoint = "louislu9911/convnextv2-base-1k-224-finetuned-cassava-leaf-disease"

splits = dataset["train"].train_test_split(
    test_size=0.1, seed=42, stratify_by_column="label"
)
train_ds = splits["train"]
val_ds = splits["test"]

image = val_ds["image"]

y = val_ds["label"]

with open("y.txt", "w") as f:
    f.write("\n".join([str(i) for i in y]))

pipe = pipeline("image-classification", model_checkpoint, device=PIPELINE_DEVICE)

y_pred = [d[0]["label"] for d in pipe.predict(image)]

with open("y_pred.txt", "w") as f:
    f.write("\n".join([str(i) for i in y_pred]))

print(confusion_matrix(y, y_pred))
# [[  49   12    2    8   38]
#  [  15  172    4   10   18]
#  [   2    4  189   25   18]
#  [   1    9   17 1270   19]
#  [  17   10    9   21  201]]

# array([[0.45, 0.11, 0.02, 0.07, 0.35],
#        [0.07, 0.79, 0.02, 0.05, 0.08],
#        [0.01, 0.02, 0.79, 0.11, 0.08],
#        [0.  , 0.01, 0.01, 0.97, 0.01],
#        [0.07, 0.04, 0.03, 0.08, 0.78]], dtype=float32)

# 0, 4
# 1, 2, 3

print(classification_report(y, y_pred))
#               precision    recall  f1-score   support

#            0       0.58      0.45      0.51       109
#            1       0.83      0.79      0.81       219
#            2       0.86      0.79      0.82       238
#            3       0.95      0.97      0.96      1316
#            4       0.68      0.78      0.73       258

#     accuracy                           0.88      2140
#    macro avg       0.78      0.75      0.77      2140
# weighted avg       0.88      0.88      0.88      2140
