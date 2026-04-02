"""API métier : CRUD patients + analyses, file RabbitMQ, validation JWT."""
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Optional

import httpx
import pika
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg2://skin:skin_db@localhost:5432/skinapp"
)
JWT_SECRET = os.environ.get("JWT_SECRET", "change-me-super-secret-key-min-32-chars!!")
JWT_ALG = "HS256"
RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
QUEUE_NAME = "skin_predictions"
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/data/uploads")
WORKER_SECRET = os.environ.get("WORKER_SECRET", "worker-shared-secret")
CONSUL_ADDR = os.environ.get("CONSUL_HTTP_ADDR", "")
SERVICE_NAME = os.environ.get("SERVICE_NAME", "api")
SERVICE_PORT = int(os.environ.get("SERVICE_PORT", "8000"))

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
security = HTTPBearer()


class Patient(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    notes = Column(Text, nullable=True)
    created_by = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    analyses = relationship(
        "Analysis",
        back_populates="patient",
        cascade="all, delete-orphan",
    )


class Analysis(Base):
    __tablename__ = "analyses"
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    user_id = Column(Integer, nullable=False)
    file_path = Column(String(512), nullable=False)
    status = Column(String(32), default="pending")
    result_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    patient = relationship("Patient", back_populates="analyses")


class PatientOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    notes: Optional[str] = None
    created_by: int
    created_at: datetime


class PatientCreate(BaseModel):
    name: str
    notes: Optional[str] = None


class AnalysisOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    patient_id: int
    user_id: int
    status: str
    result_json: Optional[str] = None
    created_at: datetime


class CompleteBody(BaseModel):
    top3: Optional[list] = None
    error: Optional[str] = None


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def decode_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        uid = int(payload.get("sub"))
        role = payload.get("role", "user")
        return {"id": uid, "role": role}
    except (JWTError, ValueError, TypeError):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token invalide")


CurrentUser = Annotated[dict, Depends(decode_token)]


def publish_job(analysis_id: int):
    params = pika.URLParameters(RABBITMQ_URL)
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ch.queue_declare(queue=QUEUE_NAME, durable=True)
    ch.basic_publish(
        exchange="",
        routing_key=QUEUE_NAME,
        body=json.dumps({"analysis_id": analysis_id}),
        properties=pika.BasicProperties(delivery_mode=2),
    )
    conn.close()


async def register_consul():
    if not CONSUL_ADDR:
        return
    payload = {
        "ID": f"{SERVICE_NAME}-{uuid.uuid4().hex[:8]}",
        "Name": SERVICE_NAME,
        "Address": SERVICE_NAME,
        "Port": SERVICE_PORT,
        "Tags": ["skin-ai", "api", "traefik.enable=false"],
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.put(f"{CONSUL_ADDR.rstrip('/')}/v1/agent/service/register", json=payload)
    except Exception as e:
        print("Consul:", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    await register_consul()
    yield


app = FastAPI(title="API Métier Skin AI", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "service": "api"}


@app.get("/patients", response_model=list[PatientOut])
def list_patients(user: CurrentUser, db: Session = Depends(get_db)):
    q = db.query(Patient)
    if user["role"] != "admin":
        q = q.filter(Patient.created_by == user["id"])
    return q.order_by(Patient.id.desc()).all()


@app.post("/patients", response_model=PatientOut)
def create_patient(user: CurrentUser, body: PatientCreate, db: Session = Depends(get_db)):
    p = Patient(name=body.name, notes=body.notes, created_by=user["id"])
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


@app.get("/patients/{pid}", response_model=PatientOut)
def get_patient(pid: int, user: CurrentUser, db: Session = Depends(get_db)):
    p = db.query(Patient).filter(Patient.id == pid).first()
    if not p:
        raise HTTPException(404)
    if user["role"] != "admin" and p.created_by != user["id"]:
        raise HTTPException(403)
    return p


@app.put("/patients/{pid}", response_model=PatientOut)
def update_patient(pid: int, user: CurrentUser, body: PatientCreate, db: Session = Depends(get_db)):
    p = db.query(Patient).filter(Patient.id == pid).first()
    if not p:
        raise HTTPException(404)
    if user["role"] != "admin" and p.created_by != user["id"]:
        raise HTTPException(403)
    p.name = body.name
    p.notes = body.notes
    db.commit()
    db.refresh(p)
    return p


@app.delete("/patients/{pid}", status_code=204)
def delete_patient(pid: int, user: CurrentUser, db: Session = Depends(get_db)):
    if user["role"] != "admin":
        raise HTTPException(403, "Réservé aux administrateurs")
    p = db.query(Patient).filter(Patient.id == pid).first()
    if not p:
        raise HTTPException(404)
    db.delete(p)
    db.commit()


@app.post("/analyses", response_model=AnalysisOut)
async def create_analysis(
    user: CurrentUser,
    patient_id: int = Form(...),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    p = db.query(Patient).filter(Patient.id == patient_id).first()
    if not p:
        raise HTTPException(404, "Patient introuvable")
    if user["role"] != "admin" and p.created_by != user["id"]:
        raise HTTPException(403)

    ext = os.path.splitext(image.filename or "")[1] or ".jpg"
    a = Analysis(
        patient_id=patient_id,
        user_id=user["id"],
        file_path="",
        status="pending",
    )
    db.add(a)
    db.flush()
    fname = f"{a.id}{ext}"
    fpath = os.path.join(UPLOAD_DIR, fname)
    with open(fpath, "wb") as f:
        f.write(await image.read())
    a.file_path = fpath
    db.commit()
    db.refresh(a)
    try:
        publish_job(a.id)
    except Exception as e:
        a.status = "error"
        db.commit()
        raise HTTPException(500, f"File RabbitMQ: {e}")
    return a


@app.get("/analyses", response_model=list[AnalysisOut])
def list_analyses(user: CurrentUser, db: Session = Depends(get_db)):
    q = db.query(Analysis)
    if user["role"] != "admin":
        q = q.filter(Analysis.user_id == user["id"])
    return q.order_by(Analysis.id.desc()).all()
