from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Text, DateTime, Float, Integer, String, ForeignKey, Boolean
from sqlalchemy.sql import func
from database.db import Base


class Upload(Base):
    """Vartotojo įkeltos nuotraukos"""
    __tablename__ = "uploads"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_name: Mapped[str] = mapped_column(String(255), nullable=False)
    filepath: Mapped[str] = mapped_column(String(500), nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(50))        # image/jpeg, image/png...
    file_size: Mapped[int | None] = mapped_column(Integer)           # baitais
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[DateTime] = mapped_column(DateTime, default=func.now())

    # Ryšys su post_uploads
    post_links: Mapped[list["PostUpload"]] = relationship(back_populates="upload")

    def __repr__(self):
        return f"<Upload {self.original_name} | {self.created_at}>"


class Post(Base):
    """Sugeneruoti Instagram postai"""
    __tablename__ = "posts"

    id: Mapped[int] = mapped_column(primary_key=True)
    post_type: Mapped[str] = mapped_column(String(20), nullable=False)   # story / carousel
    status: Mapped[str] = mapped_column(String(20), default="pending")   # pending / processing / completed / failed

    # Modelio spėjimas
    predicted_category: Mapped[str | None] = mapped_column(String(50))  # food, portrait...
    confidence: Mapped[float | None] = mapped_column(Float)
    model_used: Mapped[str | None] = mapped_column(String(20))           # cnn / vit / knn

    # Sugeneruotas tekstas
    hook: Mapped[str | None] = mapped_column(Text)
    story: Mapped[str | None] = mapped_column(Text)
    cta: Mapped[str | None] = mapped_column(Text)
    caption: Mapped[str | None] = mapped_column(Text)                    # visas tekstas kartu

    # Galutinis rezultatas
    output_path: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[DateTime] = mapped_column(DateTime, default=func.now())

    # Ryšys su post_uploads
    upload_links: Mapped[list["PostUpload"]] = relationship(back_populates="post")

    def __repr__(self):
        return f"<Post {self.post_type} | {self.predicted_category} | {self.status}>"


class PostUpload(Base):
    """Tarpinė lentelė — susieja Post ir Upload (su eiliškumu ir role)"""
    __tablename__ = "post_uploads"

    id: Mapped[int] = mapped_column(primary_key=True)
    post_id: Mapped[int] = mapped_column(ForeignKey("posts.id"), nullable=False)
    upload_id: Mapped[int] = mapped_column(ForeignKey("uploads.id"), nullable=False)
    position: Mapped[int] = mapped_column(Integer, default=1)            # eiliškumas carousel viduje

    # Skaidrės rolė:
    # single = 1 nuotrauka (story arba viena carousel)
    # hook   = pirma carousel skaidrė
    # story  = vidurinės carousel skaidrės
    # cta    = paskutinė carousel skaidrė
    slide_type: Mapped[str | None] = mapped_column(String(20))
    slide_text: Mapped[str | None] = mapped_column(Text)                 # tekstas ant tos skaidrės

    # Ryšiai
    post: Mapped["Post"] = relationship(back_populates="upload_links")
    upload: Mapped["Upload"] = relationship(back_populates="post_links")

    def __repr__(self):
        return f"<PostUpload pos={self.position} | type={self.slide_type}>"


class TrainingImage(Base):
    """Treniravimo duomenų bazė"""
    __tablename__ = "training_images"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    filepath: Mapped[str] = mapped_column(String(500), nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False)    # food, portrait...
    split: Mapped[str] = mapped_column(String(10), nullable=False)       # train / val / test
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    file_size: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str | None] = mapped_column(String(100))              # kaggle, manual...
    is_augmented: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime, default=func.now())

    def __repr__(self):
        return f"<TrainingImage {self.filename} | {self.category} | {self.split}>"


class TrainingSession(Base):
    """Modelio treniravimo istorija ir rezultatai"""
    __tablename__ = "training_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_type: Mapped[str] = mapped_column(String(20), nullable=False)  # cnn / vit / knn

    # Hyperparametrai
    learning_rate: Mapped[float | None] = mapped_column(Float)
    batch_size: Mapped[int | None] = mapped_column(Integer)
    epochs: Mapped[int | None] = mapped_column(Integer)
    optimizer: Mapped[str | None] = mapped_column(String(50))
    dropout: Mapped[float | None] = mapped_column(Float)

    # Rezultatai
    train_accuracy: Mapped[float | None] = mapped_column(Float)
    val_accuracy: Mapped[float | None] = mapped_column(Float)
    test_accuracy: Mapped[float | None] = mapped_column(Float)
    train_loss: Mapped[float | None] = mapped_column(Float)
    val_loss: Mapped[float | None] = mapped_column(Float)

    # Papildomos metrikos
    precision: Mapped[float | None] = mapped_column(Float)
    recall: Mapped[float | None] = mapped_column(Float)
    f1_score: Mapped[float | None] = mapped_column(Float)

    # Failai
    model_path: Mapped[str | None] = mapped_column(String(500))          # išsaugoto modelio kelias
    report_path: Mapped[str | None] = mapped_column(String(500))         # classification report

    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[DateTime] = mapped_column(DateTime, default=func.now())

    def __repr__(self):
        return f"<TrainingSession {self.model_type} | val_acc={self.val_accuracy}>"