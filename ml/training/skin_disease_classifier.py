import json
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

layers = tf.keras.layers
models = tf.keras.models

# =========================
# ⚡ SPEED + STABILITY BOOST
# =========================
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# =========================
# 1️⃣ Dataset
# =========================
ds = load_dataset("ahmed-ai/skin-lesions-classification-dataset")
val_key = "validation" if "validation" in ds else "test"

img_size = 224
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

class_names = ds["train"].features["label"].names
num_classes = len(class_names)

# =========================
# 2️⃣ Class weights (IMBALANCE FIX)
# =========================
labels = [ds["train"][i]["label"] for i in range(len(ds["train"]))]

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=labels
)
class_weights = dict(enumerate(class_weights))

# =========================
# 3️⃣ Efficient tf.data pipeline (FAST)
# =========================
def preprocess(example):
    img = example["image"].convert("RGB").resize((img_size, img_size))
    img = np.array(img, dtype=np.float32)
    label = example["label"]
    return img, label


def make_dataset(split, shuffle=False):
    hf_ds = ds[split]

    tf_ds = tf.data.Dataset.from_generator(
        lambda: map(preprocess, hf_ds),
        output_signature=(
            tf.TensorSpec((img_size, img_size, 3), tf.float32),
            tf.TensorSpec((), tf.int32),
        ),
    )

    tf_ds = tf_ds.cache()

    if shuffle:
        tf_ds = tf_ds.shuffle(10000, reshuffle_each_iteration=True)

    tf_ds = tf_ds.batch(batch_size)
    tf_ds = tf_ds.prefetch(AUTOTUNE)

    return tf_ds


train_ds = make_dataset("train", shuffle=True)
val_ds = make_dataset(val_key)

# =========================
# 4️⃣ Strong Augmentation (PRO)
# =========================
data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.25),
    layers.RandomContrast(0.3),
])

# =========================
# 5️⃣ Focal Loss (🔥 IMPORTANT for imbalance)
# =========================
loss_fn = tf.keras.losses.SparseCategoricalFocalCrossentropy(
    gamma=2.0,
    label_smoothing=0.1
)

# =========================
# 6️⃣ Model (EfficientNet PRO)
# =========================
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(img_size, img_size, 3),
)

base_model.trainable = False

inputs = layers.Input(shape=(img_size, img_size, 3))

x = data_aug(inputs)

# EfficientNet preprocessing (IMPORTANT FIX)
x = tf.keras.applications.efficientnet.preprocess_input(x)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.5)(x)

x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

model = models.Model(inputs, outputs)

# =========================
# 7️⃣ Learning Rate Schedule (PRO)
# =========================
lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=1e-3,
    first_decay_steps=2000
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=loss_fn,
    metrics=[
        "accuracy",
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)
    ]
)

# =========================
# 8️⃣ Callbacks (PRO MONITORING)
# =========================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=6,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_model_PRO_MAX.keras",
        save_best_only=True
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir="logs_PRO_MAX"
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        patience=3,
        factor=0.5,
        verbose=1
    )
]

# =========================
# 9️⃣ TRAIN PHASE 1
# =========================
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    class_weight=class_weights,
    callbacks=callbacks
)

# =========================
# 🔟 FINE TUNING (DEEP UNFREEZE)
# =========================
base_model.trainable = True

for layer in base_model.layers[:-60]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=loss_fn,
    metrics=[
        "accuracy",
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)
    ]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=12,
    class_weight=class_weights,
    callbacks=callbacks
)

# =========================
# 1️⃣1️⃣ Evaluation
# =========================
def evaluate(model, dataset):
    y_true, y_pred = [], []

    for x, y in dataset:
        preds = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(y.numpy())

    return np.array(y_true), np.array(y_pred)


y_true, y_pred = evaluate(model, val_ds)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))

# =========================
# 💾 SAVE FINAL MODEL
# =========================
model.save("skin_disease.keras")

print("\n✅ PRO MAX training completed")
