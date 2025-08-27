# Utilise une image Python officielle
FROM python:3.11-slim

# Définit le répertoire de travail
WORKDIR /app

# Copie le fichier requirements.txt et installe les dépendances
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copie tout le code de ton projet dans l'image
COPY . .

# Définit les variables d'environnement pour Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8080

# Expose le port que Fly.io utilisera
EXPOSE 8080

# Commande pour démarrer l'application Flask
CMD ["python", "app.py"]
