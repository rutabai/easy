from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Text, DateTime, Float, Integer, String, ForeignKey, Boolean, UniqueConstraint
from sqlalchemy.sql import func
from database.db import Base


class Upload(Base):
    """Vartotojo įkeltos nuotraukos"""
    __tablename__ = "uploads"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_name: Mapped[str] = mapped_column(String(255), nullable=False)
    filepath: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)
    mime_type: Mapped[str | None] = mapped_column(String(50))
    file_size: Mapped[int | None] = mapped_column(Integer)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[DateTime] = mapped_column(DateTime, default=func.now())

    post_links: Mapped[list["PostUpload"]] = relationship(back_populates="upload")

    def __repr__(self):
        return f"<Upload {self.original_name} | {self.created_at}>"


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
    predicted_category: Mapped[str | None] = mapped_column(String(50))
    confidence: Mapped[float | None] = mapped_column(Float)
    model_used: Mapped[str | None] = mapped_column(String(20))

    # ── Sugeneruotas tekstas ───────────────────────────────────────
    hook: Mapped[str | None] = mapped_column(Text)
    story: Mapped[str | None] = mapped_column(Text)
    cta: Mapped[str | None] = mapped_column(Text)
    caption: Mapped[str | None] = mapped_column(Text)

    filter_name: Mapped[str | None] = mapped_column(String(20), default="original")

    # Story atveju — vieno failo kelias
    output_path: Mapped[str | None] = mapped_column(String(500))

    created_at: Mapped[DateTime] = mapped_column(DateTime, default=func.now())

    upload_links: Mapped[list["PostUpload"]] = relationship(back_populates="post")

    def __repr__(self):
        return f"<Post {self.post_type} | {self.predicted_category} | {self.status}>"


class PostUpload(Base):
    """Tarpinė lentelė — susieja Post ir Upload"""
    __tablename__ = "post_uploads"

    __table_args__ = (
        UniqueConstraint("post_id", "position", name="uq_post_position"),
        UniqueConstraint("post_id", "upload_id", name="uq_post_upload"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    post_id: Mapped[int] = mapped_column(ForeignKey("posts.id"), nullable=False)
    upload_id: Mapped[int] = mapped_column(ForeignKey("uploads.id"), nullable=False)
    position: Mapped[int] = mapped_column(Integer, default=1)

    # single / hook / story / cta
    slide_type: Mapped[str | None] = mapped_column(String(20))
    slide_text: Mapped[str | None] = mapped_column(Text)

    # Kiekvienos skaidrės sugeneruoto vaizdo kelias
    output_image_path: Mapped[str | None] = mapped_column(String(500))

    post: Mapped["Post"] = relationship(back_populates="upload_links")
    upload: Mapped["Upload"] = relationship(back_populates="post_links")

    def __repr__(self):
        return f"<PostUpload pos={self.position} | type={self.slide_type}>"


class TrainingImage(Base):
    """Treniravimo duomenų bazė"""
    __tablename__ = "training_images"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    filepath: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    split: Mapped[str] = mapped_column(String(10), nullable=False)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    file_size: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str | None] = mapped_column(String(100))
    is_augmented: Mapped[bool] = mapped_column(Boolean, default=False)
    hog_features: Mapped[str | None] = mapped_column(Text) # Saugosime kaip JSON stringą
    created_at: Mapped[DateTime] = mapped_column(DateTime, default=func.now())
   

    def __repr__(self):
        return f"<TrainingImage {self.filename} | {self.category} | {self.split}>"


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

    model_path: Mapped[str | None] = mapped_column(String(500))
    report_path: Mapped[str | None] = mapped_column(String(500))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[DateTime] = mapped_column(DateTime, default=func.now())

    def __repr__(self):
        return f"<TrainingSession {self.model_type} | val_acc={self.val_accuracy}>"