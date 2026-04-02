import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlalchemy import Boolean, Column, DateTime, Integer, String, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# ------------------------
# CONFIG
# ------------------------
# fichier SQLite sera créé dans le dossier courant si aucune variable d'env n'est définie
SQLITE_PATH = os.environ.get("SQLITE_PATH", "auth.db")
JWT_SECRET = os.environ.get("JWT_SECRET", "change-me-super-secret-key-min-32-chars!!")
JWT_ALG = "HS256"
ACCESS_MINUTES = 60 * 24  # 1 jour
CONSUL_ADDR = os.environ.get("CONSUL_HTTP_ADDR", "")
SERVICE_NAME = os.environ.get("SERVICE_NAME", "auth")
SERVICE_PORT = int(os.environ.get("SERVICE_PORT", "8000"))

# ------------------------
# DB SETUP
# ------------------------
engine = create_engine(f"sqlite:///{SQLITE_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login", auto_error=False)

print("SQLite DB path:", SQLITE_PATH)  # <-- pour vérifier où la DB sera créée

# ------------------------
# MODELS
# ------------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(32), default="user")
    disabled = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    role: str = "user"


class UserOut(BaseModel):
    id: int
    email: str
    role: str

    class Config:
        orm_mode = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

# ------------------------
# UTILS
# ------------------------
def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def hash_password(p: str) -> str:
    p = p.encode('utf-8')[:72].decode('utf-8', errors='ignore')
    return pwd_context.hash(p)


def create_token(sub: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_MINUTES)
    return jwt.encode({"sub": sub, "role": role, "exp": expire}, JWT_SECRET, algorithm=JWT_ALG)


async def register_consul():
    if not CONSUL_ADDR:
        return
    svc_id = f"{SERVICE_NAME}-{uuid.uuid4().hex[:8]}"
    payload = {
        "ID": svc_id,
        "Name": SERVICE_NAME,
        "Address": SERVICE_NAME,
        "Port": SERVICE_PORT,
        "Tags": ["auth", "traefik.enable=false"],
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.put(f"{CONSUL_ADDR.rstrip('/')}/v1/agent/service/register", json=payload)
            print("Consul register auth:", r.status_code)
    except Exception as e:
        print("Consul register skip:", e)

# ------------------------
# LIFESPAN
# ------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)  # <-- crée les tables si elles n'existent pas
    await register_consul()
    yield

app = FastAPI(title="Auth Service", lifespan=lifespan)

# ------------------------
# DEPENDENCIES
# ------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ------------------------
# ROUTES
# ------------------------
@app.post("/register", response_model=UserOut)
def register(body: UserCreate, db: Session = Depends(get_db)):
    if body.role not in ("admin", "user"):
        raise HTTPException(400, "role doit être admin ou user")
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(400, "Email déjà utilisé")
    u = User(email=body.email, hashed_password=hash_password(body.password), role=body.role)
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


@app.post("/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    u = db.query(User).filter(User.email == form.username).first()
    if not u or not verify_password(form.password, u.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Identifiants invalides")
    if u.disabled:
        raise HTTPException(403, "Compte désactivé")
    token = create_token(str(u.id), u.role)
    return Token(access_token=token)


@app.get("/me", response_model=UserOut)
def me(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    if not token:
        raise HTTPException(401)
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        uid = int(payload.get("sub"))
    except (JWTError, ValueError):
        raise HTTPException(401, "Token invalide")
    u = db.query(User).filter(User.id == uid).first()
    if not u:
        raise HTTPException(401)
    return u


@app.get("/health")
def health():
    return {"status": "ok", "service": "auth"}


@app.get("/verify")
def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return {"valid": True, "sub": payload.get("sub"), "role": payload.get("role")}
    except JWTError:
        raise HTTPException(401, "Token invalide")