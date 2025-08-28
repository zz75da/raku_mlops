# === Base image légère avec Python ===
FROM python:3.10-slim

# === Installer dépendances système ===
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# === Définir le répertoire de travail ===
WORKDIR /app

# === Copier et installer les dépendances Python ===
# Copy the whole requirements folder
COPY requirements/base.txt ./base.txt
COPY requirements/mlflow.txt ./requirements.txt

# Install MLflow-specific requirements (which itself may include -r base.txt)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .


# === Exposer le port MLflow ===
EXPOSE 5000

# === Commande par défaut (sera écrasée par docker-compose.yml) ===
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "./mlruns", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
