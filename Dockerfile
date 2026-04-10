FROM python:3.11-slim

# Sisteminės bibliotekos (reikia PIL, OpenCV)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Vartotojas (HuggingFace reikalavimas)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Įdiek priklausomybes
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Nukopijuok projektą
COPY --chown=user . /app

# Sukurk reikalingus aplankus
RUN mkdir -p static/uploads static/outputs saved_models

# HuggingFace naudoja port 7860
CMD ["python", "-m", "gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "app:create_app()"]