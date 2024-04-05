import os

from datasets import load_dataset
from sklearn.metrics import confusion_matrix,classification_report
from transformers import pipeline

BATCH_SIZE = 180
NUM_TRAIN_EPOCHS = 20
NUM_WORKERS = 15

cwd = os.getcwd()
train_image_folder = os.path.join(cwd, "Datasets", "imagefolder")

print(f"begin loading {train_image_folder} ...")
dataset = load_dataset("imagefolder", data_dir=train_image_folder)
print("dataset setup successfully!")

model_checkpoint = "louislu9911/convnextv2-base-1k-224-finetuned-cassava-leaf-disease"

splits = dataset["train"].train_test_split(test_size=0.1, seed = 42, stratify_by_column="label")
train_ds = splits["train"]
val_ds = splits["test"]

image = val_ds['image']

y = val_ds['label']
# print(f"{y=}")

pipe = pipeline("image-classification", model_checkpoint, device=3)
y_pred = [d[0]['label'] for d in pipe.predict(image)]

# print(f"{y_pred=}")
# print(confusion_matrix(y, y_pred))
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
