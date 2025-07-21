import os, json
from pathlib import Path
from cryptography.fernet import Fernet
from pydantic import  Field
from pydantic_settings import BaseSettings

class SecureConfig(BaseSettings):
    encryption_key: bytes = Field(default_factory=lambda: Fernet.generate_key())
    db_key: bytes          = Field(default_factory=lambda: Fernet.generate_key())
    model_path: str        = "models/llama-2-7b-chat.Q4_K_M.gguf"
    camera_index: int      = 0
    force_cpu: bool        = False
    data_retention_days: int = 7

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

secure_cfg = SecureConfig()

def encrypt_file(src: Path, dst: Path, key: bytes):
    Fernet(key).encrypt(src.read_bytes())

def decrypt_file(src: Path, dst: Path, key: bytes):
    Fernet(key).decrypt(src.read_bytes())