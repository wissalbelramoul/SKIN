# skin_disease_classifier.py
# Dataset HF -> tf.data, transfer learning (EfficientNet/MobileNet), métriques TP.
import json
from pathlib import Path

import yaml
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix

layers = tf.keras.layers
models = tf.keras.models
EarlyStopping = tf.keras.callbacks.EarlyStopping
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
MobileNetV2 = tf.keras.applications.MobileNetV2
EfficientNetB0 = tf.keras.applications.EfficientNetB0
EfficientNetB3 = tf.keras.applications.EfficientNetB3


ARTIFACTS_DIR = Path(__file__).resolve().parent / "training_artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def _load_training_config() -> dict:
    if not _CONFIG_PATH.is_file():
        return {}
    data = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


CFG = _load_training_config()

# -------------------------------



# 1️⃣ Dataset Hugging Face
# -------------------------------
_hf_id = CFG.get("dataset", {}).get("huggingface_id", "ahmed-ai/skin-lesions-classification-dataset")
ds = load_dataset(_hf_id)
print(ds)

val_key = "validation" if "validation" in ds else "test"
if val_key not in ds:
    raise KeyError("Aucun split validation/test trouvé dans le dataset.")

# -------------------------------
# 2️⃣ Paramètres (config.yaml + défauts)
# -------------------------------
_img = CFG.get("image", {})
_tr = CFG.get("training", {})
_mod = CFG.get("model", {})
_aug = CFG.get("augmentation", {})
_prep = CFG.get("preprocessing", {})
_out = CFG.get("output", {})

img_height = int(_img.get("height", 224))
img_width = int(_img.get("width", 224))
batch_size = int(_tr.get("batch_size", 32))
epochs_head = int(_tr.get("epochs_head", 15))
epochs_finetune = int(_tr.get("epochs_finetune", 5))
ft_lr = float(_tr.get("fine_tune_learning_rate", 1e-5))
es_patience = int(_tr.get("early_stopping_patience", 5))
dense_units = int(_mod.get("dense_units", 128))
dropout_rate = float(_mod.get("dropout_rate", 0.3))
imagenet_weights = _mod.get("imagenet_weights", "imagenet")
backbone_name = str(_mod.get("backbone", "EfficientNetB0"))
label_feature = ds["train"].features["label"]
class_names = list(label_feature.names)
num_classes = len(class_names)
AUTOTUNE = tf.data.AUTOTUNE
use_imagenet_preprocess = bool(_prep.get("use_imagenet_preprocess", True))
use_class_weights = bool(_tr.get("use_class_weights", True))
model_output_path = Path(
    _out.get(
        "model_path",
        str(ARTIFACTS_DIR / "skin_model_best.keras"),
    )
)
if not model_output_path.is_absolute():
    model_output_path = (Path(__file__).resolve().parent / model_output_path).resolve()
if model_output_path.suffix.lower() != ".keras":
    # Force the Keras v3 format for checkpoints and final model export.
    model_output_path = model_output_path.with_suffix(".keras")
model_output_path.parent.mkdir(parents=True, exist_ok=True)


def _remove_previous_keras_checkpoint() -> None:
    """Supprime le dernier modèle .keras pour repartir d'un entraînement propre."""
    if model_output_path.exists():
        try:
            model_output_path.unlink()
            print(f"🧹 Checkpoint précédent supprimé: {model_output_path}")
        except OSError as exc:
            print(f"⚠️ Impossible de supprimer {model_output_path}: {exc}")


_remove_previous_keras_checkpoint()

# Export pour aligner server.py / inférence
_names_json = json.dumps(class_names, ensure_ascii=False, indent=2)
(ARTIFACTS_DIR / "class_names.json").write_text(_names_json, encoding="utf-8")
# Copie à la racine du projet (plus simple pour Docker / déploiement)
Path(__file__).resolve().parent.joinpath("class_names.json").write_text(
    _names_json, encoding="utf-8"
)

# -------------------------------
# 3️⃣ tf.data depuis HF (sparse labels + loss sparse_categorical_crossentropy)
# -------------------------------
def _preprocess_image_for_model(x: np.ndarray) -> np.ndarray:
    # Use proper ImageNet preprocessing depending on selected backbone.
    if use_imagenet_preprocess:
        if backbone_name == "MobileNetV2":
            return tf.keras.applications.mobilenet_v2.preprocess_input(x)
        if backbone_name in ("EfficientNetB0", "EfficientNetB3"):
            return tf.keras.applications.efficientnet.preprocess_input(x)
    return x / 255.0


def make_tf_dataset(split: str, shuffle: bool) -> tf.data.Dataset:
    split_ds = ds[split]
    n = len(split_ds)

    def generator():
        for i in range(n):
            ex = split_ds[i]
            img = ex["image"].convert("RGB").resize((img_width, img_height))
            x = np.asarray(img, dtype=np.float32)
            x = _preprocess_image_for_model(x)
            y = np.int32(ex["label"])
            yield x, y

    sig = (
        tf.TensorSpec(shape=(img_height, img_width, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    td = tf.data.Dataset.from_generator(generator, output_signature=sig)
    if shuffle:
        td = td.shuffle(min(10_000, n), reshuffle_each_iteration=True)
    return td.batch(batch_size).prefetch(AUTOTUNE)


train_dataset = make_tf_dataset("train", shuffle=True)
val_dataset = make_tf_dataset(val_key, shuffle=False)


def compute_class_weights() -> dict[int, float]:
    labels = np.array(ds["train"]["label"], dtype=np.int32)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    total = float(np.sum(counts))
    weights = {}
    for i in range(num_classes):
        c = counts[i] if counts[i] > 0 else 1.0
        weights[i] = total / (num_classes * c)
    return weights


class_weights = compute_class_weights() if use_class_weights else None
if class_weights:
    print("Class weights activés:", class_weights)


def save_class_distribution_artifacts() -> tuple[Path, Path]:
    labels = np.array(ds["train"]["label"], dtype=np.int32)
    counts = np.bincount(labels, minlength=num_classes).astype(int)
    dist = {class_names[i]: int(counts[i]) for i in range(num_classes)}

    json_path = ARTIFACTS_DIR / "class_distribution.json"
    json_path.write_text(json.dumps(dist, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(class_names, counts)
    ax.set_title("Distribution des classes (train)")
    ax.set_ylabel("Nombre d'images")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    plot_path = ARTIFACTS_DIR / "class_distribution.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return json_path, plot_path

# -------------------------------
# 4️⃣ Augmentation + Backbone (transfer learning)
# -------------------------------
_flip = _aug.get("random_flip", "horizontal")
data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip(_flip),
        layers.RandomRotation(float(_aug.get("random_rotation", 0.1))),
        layers.RandomZoom(float(_aug.get("random_zoom", 0.1))),
        layers.RandomHeight(float(_aug.get("random_height", 0.1))),
        layers.RandomWidth(float(_aug.get("random_width", 0.1))),
    ],
    name="data_augmentation",
)

def build_backbone(name: str):
    if name == "MobileNetV2":
        return MobileNetV2(
            weights=imagenet_weights,
            include_top=False,
            input_shape=(img_height, img_width, 3),
        )
    if name == "EfficientNetB0":
        return EfficientNetB0(
            weights=imagenet_weights,
            include_top=False,
            input_shape=(img_height, img_width, 3),
        )
    if name == "EfficientNetB3":
        return EfficientNetB3(
            weights=imagenet_weights,
            include_top=False,
            input_shape=(img_height, img_width, 3),
        )
    raise ValueError(
        f"Backbone non supporté: {name}. Choix: MobileNetV2, EfficientNetB0, EfficientNetB3"
    )


base_model = build_backbone(backbone_name)
base_model.trainable = False

inputs = layers.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(dense_units, activation=str(_mod.get("top_activation", "relu")))(x)
x = layers.Dropout(dropout_rate)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=es_patience, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
    ModelCheckpoint(
        filepath=str(model_output_path),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
    ),
]

# -------------------------------
# 5️⃣ Entraînement — phase 1 (tête seule)
# -------------------------------
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs_head,
    callbacks=callbacks,
    class_weight=class_weights,
)

# -------------------------------
# 6️⃣ Fine-tuning
# -------------------------------
base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(ft_lr),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs_finetune,
    callbacks=callbacks,
    class_weight=class_weights,
)


# -------------------------------
# 7️⃣ Courbes loss / accuracy
# -------------------------------
def plot_training_curves(h1: tf.keras.callbacks.History, h2: tf.keras.callbacks.History) -> Path:
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


# -------------------------------
# 8️⃣ Matrice de confusion + précision / rappel / F1
# -------------------------------
def collect_predictions(m: tf.keras.Model, vds: tf.data.Dataset):
    y_true = []
    y_pred = []
    for x_batch, y_batch in vds:
        probs = m.predict(x_batch, verbose=0)
        y_pred.append(np.argmax(probs, axis=1))
        y_true.append(y_batch.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def plot_confusion(cm: np.ndarray, labels: list[str]) -> Path:
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
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    fig.tight_layout()
    out = ARTIFACTS_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


curves_path = plot_training_curves(history, history_ft)
print(f"Courbes sauvegardées : {curves_path}")

y_true, y_pred = collect_predictions(model, val_dataset)
cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

report_txt = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
)
report_path = ARTIFACTS_DIR / "classification_report.txt"
report_path.write_text(report_txt, encoding="utf-8")
print(report_txt)

cm_path = plot_confusion(cm, class_names)
print(f"Matrice de confusion : {cm_path}")
print(f"Rapport P/R/F1 : {report_path}")
dist_json_path, dist_plot_path = save_class_distribution_artifacts()
print(f"Distribution classes (json) : {dist_json_path}")
print(f"Distribution classes (plot) : {dist_plot_path}")

# -------------------------------
# 9️⃣ Sauvegarde modèle pour l’API
# -------------------------------
from google.colab import drive
drive.mount('/content/drive')

model.save('/content/drive/MyDrive/model.h5')
model.save(str(model_output_path))
print(f"✅ Modèle final : {model_output_path}")
print(f"✅ Noms de classes : {ARTIFACTS_DIR / 'class_names.json'}")

# Rapport markdown prêt à coller dans le rapport PDF
report_md = ARTIFACTS_DIR / "report_results.md"
report_md.write_text(
    "\n".join(
        [
            "# Résultats Deep Learning",
            "",
            "## Modèle",
            f"- Architecture: {backbone_name} (transfer learning)",
            f"- Nombre de classes: {num_classes}",
            f"- Classes: {', '.join(class_names)}",
            "",
            "## Fichiers de résultats",
            f"- Courbes Accuracy/Loss: `{ARTIFACTS_DIR / 'training_curves.png'}`",
            f"- Matrice de confusion: `{ARTIFACTS_DIR / 'confusion_matrix.png'}`",
            f"- Rapport Precision/Recall/F1: `{ARTIFACTS_DIR / 'classification_report.txt'}`",
            f"- Distribution des classes (plot): `{dist_plot_path}`",
            f"- Distribution des classes (json): `{dist_json_path}`",
            "",
            "## Remarque",
            "- Insérer ces figures et ce résumé dans le rapport final PDF.",
        ]
    ),
    encoding="utf-8",
)
print(f"✅ Résumé rapport généré : {report_md}")
