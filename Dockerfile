# Utilise une image Python officielle
FROM python:3.11-slim

# Définit le répertoire de travail
WORKDIR /app

# Copie les dépendances et installe
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie tout le code de ton projet
COPY . .

# Définir les variables d'environnement pour Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8000

# Expose le port 8000
EXPOSE 8000

# Commande pour démarrer l'application
CMD ["flask", "run"]
