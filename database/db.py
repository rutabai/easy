from sqlalchemy import create_engine #importuoja funkcija, kuri sukuria prisijungima prie duomenu bazes
from sqlalchemy.orm import sessionmaker, DeclarativeBase #sessionmaker leis kurti DB sesijas, declerativebase bus baze modeliams/lentelems
from dotenv import load_dotenv #leidzia uzkrauti .env failo kintamuosius
import os #reikalingas pasiimti aplinkos kintamuosius (pvz DATABASE_URL)
 
load_dotenv() #uzkrauna .env faila, kad programa galetu nuskaityti jo reiksmes
 
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///instagram_tool.db")
#bando paimti DATABASE_URL is .env failo. jeineranda, naudoja numatyta reiksme "sqlote://,,," tai reiskia vietine sqlite db instagram_tool.db
 
engine = create_engine(DATABASE_URL, echo=False) #sukuria rysi su db pagal DATABASE_URL. echo=False reiškia, kad SQL užklausos nebus rodomos terminale
SessionLocal = sessionmaker(bind=engine) #sukuria sesiju gamykla, veliau su sessionlocal() bus galima atsidaryti DB sesija
 
#bazine klase visiems modeliams
class Base(DeclarativeBase):  #sukuria bazine klase, nuo kurios paveldes visi modeliai: image, prediction, trainingsession
    pass

 #funkcija gauti DB
def get_db():
    db = SessionLocal()  #sukuriama nauja DB sesija
    try:
        yield db     #sesija perduodama naudoti kitai programos daliai
    finally:
        db.close()  #pasibaigus naudojimui, sesija visada uzdaroma

 #funkcija sukurti lenteles duomenu bazeje
def init_db():
    from database.models import Image, Prediction, TrainingSession  #importuojami modeliai, kad SQLAlchemy zinotu, kokias lenteles reikia kurti
    Base.metadata.create_all(bind=engine)  #sukuria visas lenteles pagal modelius, jei ju dar nera
    print("✅ Duomenų bazė sukurta!") #isveda zinute terminale, jog DB jau sukurta

#yield skiriasi nuo return, kuris uzbaigia funkcija visam laikui. yield susttabdo funkcija atiduoda reiksme lauk ir veliau gali testi darba nuo tos pacios vietos
#po yield funkcija grizta atgal ir vyksta finally. tai reiskia, kad sesija bus uzdaryta net jei ivyko klaida