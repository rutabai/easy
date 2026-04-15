from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Text, DateTime, Float, Integer, String, ForeignKey, Boolean, UniqueConstraint
from sqlalchemy.sql import func #importuojama del func.now() (jei rasytume datetime.now() butu fiksuojama kada programa paleidziama, o dabar fiksuojama, kai sukuriamas konkretus irasas)
from database.db import Base 
#-----
#orm - object relational mapper leidzia dirbti su db nerasant sql

class Upload(Base):
    """Vartotojo įkeltos nuotraukos"""
    __tablename__ = "uploads"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_name: Mapped[str] = mapped_column(String(255), nullable=False) #nusako fialo tipa jpeg ar png. str|None reiskia, jog gali buti tekstas arba gali nieko nebutu=i
    filepath: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)
    mime_type: Mapped[str | None] = mapped_column(String(50))
    file_size: Mapped[int | None] = mapped_column(Integer)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[DateTime] = mapped_column(DateTime, default=func.now())

    post_links: Mapped[list["PostUpload"]] = relationship(back_populates="upload") #leidzia is upload pasiekti prie kokiu postu ji prijungta

  


class Post(Base):
    """Sugeneruoti Instagram postai"""
    __tablename__ = "posts"

    id: Mapped[int] = mapped_column(primary_key=True)

    # story / carousel
    post_type: Mapped[str] = mapped_column(String(20), nullable=False)

    # pending / processing / completed / failed
    status: Mapped[str] = mapped_column(String(20), default="pending")

    # ── Vartotojo intencija ────────────────────────────────────────
    topic: Mapped[str | None] = mapped_column(Text)
    goal: Mapped[str | None] = mapped_column(String(50))
    cta_type: Mapped[str | None] = mapped_column(String(50))
    additional_notes: Mapped[str | None] = mapped_column(Text)

    # ── Modelio spėjimas ───────────────────────────────────────────
    predicted_category: Mapped[str | None] = mapped_column(String(50))  #kokia kategorija modelis atspejo
    confidence: Mapped[float | None] = mapped_column(Float)  #kokiu tikslumu atspejo
    model_used: Mapped[str | None] = mapped_column(String(20))   #koks modelis tai padare

    # ── Sugeneruotas tekstas ───────────────────────────────────────
    hook: Mapped[str | None] = mapped_column(Text)
    story: Mapped[str | None] = mapped_column(Text)
    cta: Mapped[str | None] = mapped_column(Text)
    caption: Mapped[str | None] = mapped_column(Text)

    filter_name: Mapped[str | None] = mapped_column(String(20), default="original")  #koks filtras buvo parinktas (original, warm ar kitas)

    # Story atveju — vieno failo kelias
    output_path: Mapped[str | None] = mapped_column(String(500))  #kur issaugotas sugeneruotas galutinis paveikslelis

    created_at: Mapped[DateTime] = mapped_column(DateTime, default=func.now())  #kada sis postas sukurtas

    upload_links: Mapped[list["PostUpload"]] = relationship(back_populates="post") #ci svarbu carusel logikai, nes vienas postas gali turti daugiau susijusiu irasu




class PostUpload(Base):
    """Tarpinė lentelė — susieja Post ir Upload"""
    __tablename__ = "post_uploads"

    __table_args__ = (
        UniqueConstraint("post_id", "position", name="uq_post_position"), #viename poste ta pati pozicija negali kartotis
        UniqueConstraint("post_id", "upload_id", name="uq_post_upload"),    #ta pati nuotrauka negali buti dukart prideta prie to paties posto
    )
#table_args yra specialus klases atributas, kuri SQLAlchemy supranta kaip papldomus lenteles nusttymus


    id: Mapped[int] = mapped_column(primary_key=True)  #unikalus tarpinio rysio ID
    post_id: Mapped[int] = mapped_column(ForeignKey("posts.id"), nullable=False)  #nuoroda i posts lentele
    upload_id: Mapped[int] = mapped_column(ForeignKey("uploads.id"), nullable=False)  #nuoroda i uploads lenele
    position: Mapped[int] = mapped_column(Integer, default=1)  #kuri carusel tai yra

    # single / hook / story / cta
    slide_type: Mapped[str | None] = mapped_column(String(20)) #kokio tipo tai skaidre ir tai padeda nuspresti koki teksta deti
    slide_text: Mapped[str | None] = mapped_column(Text) #ir cia jau konkrecios skaidres tekstas

    # Kiekvienos skaidrės sugeneruoto vaizdo kelias
    output_image_path: Mapped[str | None] = mapped_column(String(500)) #kur issaugotas buten tos skaidres sugeneruotas paveikslelis

    post: Mapped["Post"] = relationship(back_populates="upload_links")  
    upload: Mapped["Upload"] = relationship(back_populates="post_links")




class TrainingImage(Base):
    """Treniravimo duomenų bazė"""
    __tablename__ = "training_images"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    filepath: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    split: Mapped[str] = mapped_column(String(10), nullable=False) #kam priklauso: train/val/test
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    file_size: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str | None] = mapped_column(String(100)) #is kur duomenys gauti
    is_augmented: Mapped[bool] = mapped_column(Boolean, default=False) #ar nuotrauka yra augmentuota
    hog_features: Mapped[str | None] = mapped_column(Text) # Saugosime kaip JSON stringą 
    created_at: Mapped[DateTime] = mapped_column(DateTime, default=func.now())
   




class TrainingSession(Base):
    """Modelio treniravimo istorija ir rezultatai"""
    __tablename__ = "training_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_type: Mapped[str] = mapped_column(String(20), nullable=False)

    learning_rate: Mapped[float | None] = mapped_column(Float)
    batch_size: Mapped[int | None] = mapped_column(Integer)
    epochs: Mapped[int | None] = mapped_column(Integer)
    optimizer: Mapped[str | None] = mapped_column(String(50))
    dropout: Mapped[float | None] = mapped_column(Float)

    train_accuracy: Mapped[float | None] = mapped_column(Float)
    val_accuracy: Mapped[float | None] = mapped_column(Float)
    test_accuracy: Mapped[float | None] = mapped_column(Float)
    train_loss: Mapped[float | None] = mapped_column(Float)
    val_loss: Mapped[float | None] = mapped_column(Float)

    precision: Mapped[float | None] = mapped_column(Float)
    recall: Mapped[float | None] = mapped_column(Float)
    f1_score: Mapped[float | None] = mapped_column(Float)

    model_path: Mapped[str | None] = mapped_column(String(500)) #kur issaugotas modelis
    report_path: Mapped[str | None] = mapped_column(String(500)) #kur ataskaita
    notes: Mapped[str | None] = mapped_column(Text)                 # pastabos apie teniravima
    created_at: Mapped[DateTime] = mapped_column(DateTime, default=func.now())

 