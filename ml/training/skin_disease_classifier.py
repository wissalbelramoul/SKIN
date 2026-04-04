# skin_disease_classifier.py (Production Ready, Optimized)

import json
from pathlib import Path

import tensorflow as tf
from datasets import load_dataset
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -------------------------------
# 📁 Paths
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR.parent / "training_artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "skin_20model.h5"

# -------------------------------
# ⚙️ Params
# -------------------------------
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# -------------------------------
# 📊 Dataset
# -------------------------------
print("📥 Loading dataset...")
ds = load_dataset("isic_2020")
val_key = "validation" if "validation" in ds else "test"

label_feature = ds["train"].features["label"]
class_names = list(label_feature.names)
NUM_CLASSES = len(class_names)
print("Classes:", class_names)

# -------------------------------
# ⚖️ Class weights
# -------------------------------
labels_list = [ex["label"] for ex in ds["train"]]
class_weights = compute_class_weight(
    "balanced",
    classes=np.unique(labels_list),
    y=labels_list
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# -------------------------------
# 🔄 Preprocessing
# -------------------------------
def preprocess_example(example):
    img = example["image"].convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    x = preprocess_input(np.array(img))
    y = np.int32(example["label"])
    return x, y

def make_tf_dataset(split, shuffle=True):
    ds_split = ds[split]

    def generator():
        for ex in ds_split:
            yield preprocess_example(ex)

    sig = (
        tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    tf_ds = tf.data.Dataset.from_generator(generator, output_signature=sig)
    if shuffle:
        tf_ds = tf_ds.shuffle(buffer_size=1000).repeat()  # ✅ repeat to avoid OUT_OF_RANGE
    else:
        tf_ds = tf_ds.repeat()
    return tf_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_dataset = make_tf_dataset("train", True)
val_dataset = make_tf_dataset(val_key, False)

# Compute steps_per_epoch
steps_per_epoch = len(ds["train"]) // BATCH_SIZE
validation_steps = len(ds[val_key]) // BATCH_SIZE

# -------------------------------
# 🔁 Augmentation
# -------------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2),
])

# -------------------------------
# 🧠 Model
# -------------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
)
base_model.trainable = False

inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.6)(x)  # slightly higher dropout
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# -------------------------------
# ⏹️ Callbacks
# -------------------------------
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint(str(MODEL_PATH), save_best_only=True),
]

# -------------------------------
# 🚀 Training
# -------------------------------
print("🚀 Training...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    class_weight=class_weights,
    callbacks=callbacks
)

# -------------------------------
# 🔥 Fine-tuning
# -------------------------------
print("🔥 Fine-tuning...")
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    class_weight=class_weights,
    callbacks=callbacks
)

# -------------------------------
# 📊 Evaluation
# -------------------------------
print("📊 Evaluating...")
y_true, y_pred = [], []

for x, y in val_dataset.take(validation_steps):
    preds = model.predict(x, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(y.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

report = classification_report(y_true, y_pred, target_names=class_names)
cm = confusion_matrix(y_true, y_pred)

print(report)
print("Confusion Matrix:\n", cm)

# -------------------------------
# 💾 Save results
# -------------------------------
(ARTIFACTS_DIR / "classification_report.txt").write_text(report)
np.save(ARTIFACTS_DIR / "confusion_matrix.npy", cm)

with open(ARTIFACTS_DIR / "class_names.json", "w") as f:
    json.dump(class_names, f)

# -------------------------------
# 💾 Save model FINAL
# -------------------------------
model.save(MODEL_PATH)
print("\n✅ Model saved:", MODEL_PATH)
print("📁 Results saved in:", ARTIFACTS_DIR)
