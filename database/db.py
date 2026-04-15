from sqlalchemy import create_engine  #create_engine sukuria rysi du db. be jos nebutu galima prisijungti prie db
from sqlalchemy.orm import sessionmaker, DeclarativeBase #session maker naudojamas sukurti db sesiju kureja (daryti uzklausas, irasyti duomenis, commit pakeitimus), o declerative base  naudojamas bazinei klasei modeliams, 
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///instagram_tool.db")

engine = create_engine(DATABASE_URL, echo=False) #engine yra pagrindinis rysio objektas tarp programos ir db. echo=Falso - sql komandos nebus spauzdinamos i terminala
SessionLocal = sessionmaker(bind=engine) #sukuria sesiju kureja. objektas, kuris kuria sesijas naujas. 

class Base(DeclarativeBase): #sukuria mazine klase visoms lentelems 
    pass

#uzkomentuojam nes DB sesijos valdomos su @app.before_request ir @appteardown_request
# def get_db():  #sukuria db sesija ir veliau ja uzdaro
#     """Grąžina DB sesiją kaip generatorių. Naudoti su next() arba Flask route'uose per Depends()."""
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

def init_db():
    """Sukuria visas lenteles pagal dabartinį models.py."""
    # Importuojam visus modelius kad SQLAlchemy juos užregistruotų
    from database.models import Upload, Post, PostUpload, TrainingImage, TrainingSession
    Base.metadata.create_all(bind=engine) #sukuria visas lenteles db pagal aprasymus, jei ju dar nera
    print("✅ Duomenų bazė sukurta!")

print("DB absolute path:", os.path.abspath(engine.url.database))