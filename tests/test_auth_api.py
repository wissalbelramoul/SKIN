import importlib.util
import os

def load_module_from_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def auth_client(monkeypatch, tmp_path):
    sqlite_file = tmp_path / "auth_test.db"
    monkeypatch.setenv("SQLITE_PATH", str(sqlite_file))
    monkeypatch.setenv("JWT_SECRET", "test-secret-jwt-1234567890")

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    auth_main_path = os.path.join(root, "auth", "app", "main.py")
    auth_mod = load_module_from_path(auth_main_path, "auth_main")
    monkeypatch.setattr(auth_mod, "hash_password", lambda p: p)
    monkeypatch.setattr(auth_mod, "verify_password", lambda plain, hashed: plain == hashed)

    with TestClient(auth_mod.app) as client:
        yield client


@pytest.fixture
def api_client(monkeypatch, tmp_path):
    sqlite_file = tmp_path / "api_test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{sqlite_file}")
    monkeypatch.setenv("JWT_SECRET", "test-secret-jwt-1234567890")

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    api_main_path = os.path.join(root, "api", "app", "main.py")
    api_mod = load_module_from_path(api_main_path, "api_main")

    with TestClient(api_mod.app) as client:
        yield client


def test_auth_register_login_verify(auth_client):
    r = auth_client.post("/register", json={"email": "test@exemple.com", "password": "Password123!", "role": "user"})
    assert r.status_code == 200, r.text
    assert r.json()["email"] == "test@exemple.com"

    r = auth_client.post("/login", data={"username": "test@exemple.com", "password": "Password123!"})
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]
    assert token

    h = {"Authorization": f"Bearer {token}"}

    r = auth_client.get("/me", headers=h)
    assert r.status_code == 200
    assert r.json()["email"] == "test@exemple.com"

    r = auth_client.get("/verify", headers=h)
    assert r.status_code == 200
    assert r.json()["valid"] is True


def test_api_patients_crud(auth_client, api_client):
    # Create auth token
    r = auth_client.post("/register", json={"email": "apiuser@exemple.com", "password": "Password123!", "role": "user"})
    assert r.status_code == 200
    r = auth_client.post("/login", data={"username": "apiuser@exemple.com", "password": "Password123!"})
    assert r.status_code == 200
    token = r.json()["access_token"]

    headers = {"Authorization": f"Bearer {token}"}

    r = api_client.get("/health")
    assert r.status_code == 200
    assert r.json()["service"] == "api"

    r = api_client.post("/patients", headers=headers, json={"name": "Patient1", "notes": "note1"})
    assert r.status_code == 200
    pid = r.json()["id"]

    r = api_client.get("/patients", headers=headers)
    assert r.status_code == 200
    assert any(p["id"] == pid for p in r.json())

    r = api_client.get(f"/patients/{pid}", headers=headers)
    assert r.status_code == 200
    assert r.json()["name"] == "Patient1"
