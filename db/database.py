import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from memnai.storage.manager import StorageManager
from memnai.db.models import Base

storage_manager = StorageManager()

def get_engine():
    db_dir = storage_manager.get_path("db")
    db_path = os.path.join(db_dir, "sessions.db")
    
    sqlite_url = f"sqlite:///{db_path}"
    engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    
    return engine

def get_session():
    engine = get_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()
