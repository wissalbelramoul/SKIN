# skin_disease_classifier_colab.py
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------
# TensorFlow shortcuts
layers = tf.keras.layers
models = tf.keras.models
EarlyStopping = tf.keras.callbacks.EarlyStopping
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
MobileNetV2 = tf.keras.applications.MobileNetV2

# -------------------------------
# Artifacts
ARTIFACTS_DIR = Path("/content/training_artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# 1️⃣ Dataset Hugging Face
ds = load_dataset("isic_2020")
val_key = "validation" if "validation" in ds else "test"
if val_key not in ds:
    raise KeyError("Aucun split validation/test trouvé dans le dataset.")

# -------------------------------
# 2️⃣ Paramètres
img_height, img_width = 224, 224
batch_size = 32
label_feature = ds["train"].features["label"]
class_names = list(label_feature.names)
num_classes = len(class_names)
AUTOTUNE = tf.data.AUTOTUNE

# Export class names
_names_json = json.dumps(class_names, ensure_ascii=False, indent=2)
(ARTIFACTS_DIR / "class_names.json").write_text(_names_json, encoding="utf-8")

# -------------------------------
# 3️⃣ tf.data depuis HF (avec repeat)
def make_tf_dataset(split: str, shuffle: bool) -> tf.data.Dataset:
    split_ds = ds[split]
    n = len(split_ds)

    def generator():
        for i in range(n):
            ex = split_ds[i]
            img = ex["image"].convert("RGB").resize((img_width, img_height))
            x = np.asarray(img, dtype=np.float32) / 255.0
            y = np.int32(ex["label"])
            yield x, y

    sig = (
        tf.TensorSpec(shape=(img_height, img_width, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    td = tf.data.Dataset.from_generator(generator, output_signature=sig)
    
    if shuffle:
        td = td.shuffle(min(10_000, n), reshuffle_each_iteration=True)
    
    td = td.repeat()  # <- répète indéfiniment
    td = td.batch(batch_size).prefetch(AUTOTUNE)
    return td

train_dataset = make_tf_dataset("train", shuffle=True)
val_dataset = make_tf_dataset(val_key, shuffle=False)

steps_per_epoch = len(ds["train"]) // batch_size
validation_steps = len(ds[val_key]) // batch_size

# -------------------------------
# 4️⃣ Augmentation + MobileNetV2
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomHeight(0.1),
    layers.RandomWidth(0.1),
], name="data_augmentation")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(img_height, img_width, 3)
)
base_model.trainable = False

inputs = layers.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(
        filepath=str(ARTIFACTS_DIR / "skin_model_best.h5"),
        monitor="val_loss",
        save_best_only=True,
    ),
]

# -------------------------------
# 5️⃣ Entraînement phase 1
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=15,
    callbacks=callbacks,
)

# -------------------------------
# 6️⃣ Fine-tuning
base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model.fit(
    train_dataset,
    validation_data=val_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=5,
    callbacks=callbacks,
)

# -------------------------------
# 7️⃣ Courbes
def plot_training_curves(h1, h2):
    def merge_series(key: str):
        return list(h1.history.get(key, [])) + list(h2.history.get(key, []))

    acc = merge_series("accuracy")
    val_acc = merge_series("val_accuracy")
    loss = merge_series("loss")
    val_loss = merge_series("val_loss")
    epochs_range = range(1, len(loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs_range, loss, label="train loss")
    axes[0].plot(epochs_range, val_loss, label="val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Loss (transfert + fine-tuning)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, acc, label="train acc")
    axes[1].plot(epochs_range, val_acc, label="val acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].set_title("Accuracy")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = ARTIFACTS_DIR / "training_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

curves_path = plot_training_curves(history, history_ft)
print(f"Courbes sauvegardées : {curves_path}")

# -------------------------------
# 8️⃣ Matrice de confusion + classification report
def collect_predictions(m, vds):
    y_true = []
    y_pred = []
    for x_batch, y_batch in vds:
        probs = m.predict(x_batch, verbose=0)
        y_pred.append(np.argmax(probs, axis=1))
        y_true.append(y_batch.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)

y_true, y_pred = collect_predictions(model, val_dataset)
cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

report_txt = classification_report(y_true, y_pred, target_names=class_names, digits=4)
report_path = ARTIFACTS_DIR / "classification_report.txt"
report_path.write_text(report_txt, encoding="utf-8")
print(report_txt)

def plot_confusion(cm, labels):
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Vrai label",
        xlabel="Prédiction",
        title="Matrice de confusion (validation)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    fig.tight_layout()
    out = ARTIFACTS_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out

cm_path = plot_confusion(cm, class_names)
print(f"Matrice de confusion : {cm_path}")

# -------------------------------
# 9️⃣ Sauvegarde modèle final
final_path = Path("/content/skin_20model_best.h5")
model.save(str(final_path))
print(f"✅ Modèle final : {final_path}")
print(f"✅ Noms de classes : {ARTIFACTS_DIR / 'class_names.json'}")
