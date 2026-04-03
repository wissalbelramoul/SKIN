# ==============================
# Google Colab Compatible Version
# ==============================

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------
# 📁 Paths (Colab)
# -------------------------------
BASE_DIR = Path("/content")
ARTIFACTS_DIR = BASE_DIR / "training_artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# 1️⃣ Dataset Hugging Face
# -------------------------------
ds = load_dataset("ahmed-ai/skin-lesions-classification-dataset")
print(ds)

val_key = "validation" if "validation" in ds else "test"
if val_key not in ds:
    raise KeyError("No validation/test split found.")

# -------------------------------
# 2️⃣ Parameters
# -------------------------------
img_height, img_width = 224, 224
batch_size = 16  # 🚀 أفضل من 10

label_feature = ds["train"].features["label"]
class_names = list(label_feature.names)
num_classes = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE

# Save class names
_names_json = json.dumps(class_names, ensure_ascii=False, indent=2)
(ARTIFACTS_DIR / "class_names.json").write_text(_names_json, encoding="utf-8")

# -------------------------------
# 3️⃣ tf.data Dataset
# -------------------------------
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def make_tf_dataset(split: str, shuffle: bool):
    split_ds = ds[split]
    n = len(split_ds)

    def generator():
        for i in range(n):
            ex = split_ds[i]
            img = ex["image"].convert("RGB").resize((img_width, img_height))
            x = preprocess_input(np.asarray(img))
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
# 4️⃣ Model (MobileNetV2)
# -------------------------------
layers = tf.keras.layers
MobileNetV2 = tf.keras.applications.MobileNetV2

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

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
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)  # 🔥 تحسين
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# -------------------------------
# Callbacks
# -------------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ARTIFACTS_DIR / "best_model.h5"),
        save_best_only=True,
    ),
]

# -------------------------------
# 5️⃣ Training (Phase 1)
# -------------------------------
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=callbacks,
)

# -------------------------------
# 6️⃣ Fine-tuning
# -------------------------------
base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5,
    callbacks=callbacks,
)

# -------------------------------
# 7️⃣ Plot curves
# -------------------------------
def plot_training(h1, h2):
    acc = h1.history["accuracy"] + h2.history["accuracy"]
    val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss = h1.history["loss"] + h2.history["loss"]
    val_loss = h1.history["val_loss"] + h2.history["val_loss"]

    epochs = range(len(loss))

    plt.figure()
    plt.plot(epochs, acc, label="train acc")
    plt.plot(epochs, val_acc, label="val acc")
    plt.legend()
    plt.savefig(ARTIFACTS_DIR / "accuracy.png")

    plt.figure()
    plt.plot(epochs, loss, label="train loss")
    plt.plot(epochs, val_loss, label="val loss")
    plt.legend()
    plt.savefig(ARTIFACTS_DIR / "loss.png")

plot_training(history, history_ft)

# -------------------------------
# 8️⃣ Metrics
# -------------------------------
def get_preds(model, dataset):
    y_true, y_pred = [], []

    for x, y in dataset:
        preds = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(y.numpy())

    return np.array(y_true), np.array(y_pred)

y_true, y_pred = get_preds(model, val_dataset)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Report
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

(ARTIFACTS_DIR / "report.txt").write_text(report)

# -------------------------------
# 9️⃣ Save model
# -------------------------------
model.save("/content/skin_model.h5")

print("✅ Model saved in /content/")
print("📁 Results in:", ARTIFACTS_DIR)
