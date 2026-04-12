# skin_disease_classifier_optimized.py

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

layers = tf.keras.layers
models = tf.keras.models
EarlyStopping = tf.keras.callbacks.EarlyStopping
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint

ARTIFACTS_DIR = Path(__file__).resolve().parent / "training_artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# 1️⃣ Dataset
# -------------------------------
ds = load_dataset("ahmed-ai/skin-lesions-classification-dataset")

val_key = "validation" if "validation" in ds else "test"

# -------------------------------
# 2️⃣ Params
# -------------------------------
img_height, img_width = 224, 224
batch_size = 16
AUTOTUNE = tf.data.AUTOTUNE

label_feature = ds["train"].features["label"]
class_names = list(label_feature.names)
num_classes = len(class_names)

# Save class names
(ARTIFACTS_DIR / "class_names.json").write_text(
    json.dumps(class_names, indent=2), encoding="utf-8"
)

# -------------------------------
# 3️⃣ Class weights (IMPORTANT)
# -------------------------------
labels_list = [ds["train"][i]["label"] for i in range(len(ds["train"]))]

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=labels_list,
)

class_weights = dict(enumerate(class_weights))

# -------------------------------
# 4️⃣ tf.data
# -------------------------------
def make_tf_dataset(split, shuffle):
    split_ds = ds[split]
    n = len(split_ds)

    def generator():
        for i in range(n):
            ex = split_ds[i]
            img = ex["image"].convert("RGB").resize((img_width, img_height))

            x = preprocess_input(np.asarray(img, dtype=np.float32))
            y = np.int32(ex["label"])

            yield x, y

    sig = (
        tf.TensorSpec(shape=(img_height, img_width, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    ds_tf = tf.data.Dataset.from_generator(generator, output_signature=sig)

    if shuffle:
        ds_tf = ds_tf.shuffle(min(10000, n))

    return ds_tf.batch(batch_size).prefetch(AUTOTUNE)


train_dataset = make_tf_dataset("train", True)
val_dataset = make_tf_dataset(val_key, False)

# -------------------------------
# 5️⃣ Augmentation
# -------------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.2),
])

# -------------------------------
# 6️⃣ Model
# -------------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(img_height, img_width, 3),
)

base_model.trainable = False

inputs = layers.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# -------------------------------
# 7️⃣ Callbacks
# -------------------------------
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint(
        filepath=str(ARTIFACTS_DIR / "best_model.h5"),
        save_best_only=True,
    ),
]

# -------------------------------
# 8️⃣ Training (Phase 1)
# -------------------------------
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=15,
    class_weight=class_weights,
    callbacks=callbacks,
)

# -------------------------------
# 9️⃣ Fine-tuning
# -------------------------------
base_model.trainable = True

# نفتح غير آخر layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks,
)

# -------------------------------
# 🔟 Evaluation
# -------------------------------
def collect_predictions(model, dataset):
    y_true, y_pred = [], []

    for x_batch, y_batch in dataset:
        preds = model.predict(x_batch, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(y_batch.numpy())

    return np.array(y_true), np.array(y_pred)


y_true, y_pred = collect_predictions(model, val_dataset)

print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# -------------------------------
# 1️⃣1️⃣ Save model
# -------------------------------
model.save("skin_model_final.h5")

print("✅ Training complete")
