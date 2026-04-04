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
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# -------------------------------
# 📁 Paths
# -------------------------------
STATIC_DIR = Path(__file__).resolve().parent

MODEL_PATH = STATIC_DIR / "training_artifacts" / "skin_model.h5"


model: Optional[object] = None
model_load_error: Optional[str] = None


# -------------------------------
# 📚 Load class names
# -------------------------------
def _load_class_names():
    path = STATIC_DIR / "training_artifacts" / "class_names.json"
    if path.exists():
        return json.loads(path.read_text())
    return []


class_names = _load_class_names()


# -------------------------------
# 🚀 Load model
# -------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_load_error
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"✅ Model loaded: {MODEL_PATH}")
    except Exception as e:
        model_load_error = str(e)
        print("❌ Error loading model:", model_load_error)
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# 🧪 Health check
# -------------------------------
@app.get("/api/health")
def health():
    return {
        "model_loaded": model is not None,
        "error": model_load_error
    }


# -------------------------------
# 🔮 Prediction
# -------------------------------
@app.post("/api/predict/")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        raw = await image.read()

        # ✅ نفس preprocessing تاع training
        img = tf.keras.preprocessing.image.load_img(
            BytesIO(raw), target_size=(224, 224)
        )

        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array, verbose=0)[0]

        top3_idx = preds.argsort()[-3:][::-1]

        top3 = [
            {
                "label": class_names[i],
                "confidence": float(preds[i])
            }
            for i in top3_idx
        ]

        return {"predictions": top3}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------
# 🌐 Frontend static
# -------------------------------
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="site")


# -------------------------------
# ▶️ Run
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
