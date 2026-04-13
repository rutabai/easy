from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///instagram_tool.db")

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)

class Base(DeclarativeBase):
    pass

def get_db():
    """Grąžina DB sesiją kaip generatorių. Naudoti su next() arba Flask route'uose per Depends()."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Sukuria visas lenteles pagal dabartinį models.py."""
    # Importuojam visus modelius kad SQLAlchemy juos užregistruotų
    from database.models import Upload, Post, PostUpload, TrainingImage, TrainingSession
    Base.metadata.create_all(bind=engine)
    print("✅ Duomenų bazė sukurta!")

print("DB absolute path:", os.path.abspath(engine.url.database))