# Utilise une image Python officielle
FROM python:3.11-slim

# Définit le répertoire de travail
WORKDIR /app

# Copie le requirements.txt et installe les dépendances
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copie tout le code de ton projet
COPY . .

# Définit la variable d'environnement pour Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Expose le port que Fly.io utilisera
EXPOSE 8000

# Commande pour démarrer ton application Flask
CMD ["python", "app.py"]
