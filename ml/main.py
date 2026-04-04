"Server FastAPI pour la classification d'images de peau avec un modèle TensorFlow"
from contextlib import asynccontextmanager
from typing import Optional
import json
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import tensorflow as tf

STATIC_DIR = Path(__file__).resolve().parent

# Try multiple possible paths for the model file
_possible_paths = [
    STATIC_DIR / "training_artifacts" / "skin_model_best.h5",
    Path("/app/training_artifacts/skin_model_best.h5"),  # Docker default
    Path("/usr/src/app/training_artifacts/skin_model_best.h5"),  # Alternative Docker path
]

MODEL_PATH = next((p for p in _possible_paths if p.exists()), _possible_paths[0])


model: Optional[object] = None
model_load_error: Optional[str] = None


def _load_class_names() -> list:
    for path in (
        STATIC_DIR / "class_names.json",
        STATIC_DIR / "training_artifacts" / "class_names.json",
    ):
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return [
        "Melanoma",
        "Nevus",
        "Basal Cell Carcinoma",
        "Actinic Keratosis",
    ]


class_names = _load_class_names()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_load_error
    try:
        custom_objects = {
            "RandomHeight": tf.keras.layers.RandomHeight,
            "RandomWidth": tf.keras.layers.RandomWidth,
            "RandomFlip": tf.keras.layers.RandomFlip,
            "RandomRotation": tf.keras.layers.RandomRotation,
            "RandomZoom": tf.keras.layers.RandomZoom,
            "Rescaling": tf.keras.layers.Rescaling,
        }
        model = tf.keras.models.load_model(
            str(MODEL_PATH),
            custom_objects=custom_objects,
            compile=False,
        )
        print(f"✅ OK — Modèle chargé avec succès : {MODEL_PATH}")
    except Exception as e:
        model_load_error = f"{type(e).__name__}: {e}"
        print("=" * 60)
        print("⚠️ ATTENTION — Erreur lors du chargement du modèle.")
        print(f"Fichier attendu : {MODEL_PATH}")
        print(f"Erreur : {model_load_error}")
        print("=" * 60)
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "error": model_load_error,
    }


@app.post("/api/predict/")
async def predict(image: UploadFile = File(..., description="Image à classifier")):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Modèle non chargé.",
                "path": str(MODEL_PATH),
                "error": model_load_error,
            },
        )

    try:
        raw = await image.read()
        img = tf.keras.preprocessing.image.load_img(BytesIO(raw), target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        n = int(model.output_shape[-1])
        if len(class_names) != n:
            raise HTTPException(
                status_code=500,
                detail=f"Mismatch: class_names({len(class_names)}) vs model output({n})"
            )

        preds = model.predict(img_array, verbose=0)[0]
        top3_idx = preds.argsort()[-3:][::-1]
        top3 = [
            {"label": class_names[i], "confidence": round(float(preds[i]) * 100, 2)}
            for i in top3_idx
        ]

        return {"top3": top3}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="site")


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("server:app", host=host, port=port, reload=False)
