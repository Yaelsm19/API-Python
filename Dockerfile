# Utilise une image Python officielle
FROM python:3.11-slim

# Définit le répertoire de travail
WORKDIR /app

# Copie les dépendances et installe
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie tout le code de ton projet
COPY . .

# Expose le port utilisé
EXPOSE 8000

# Commande pour démarrer l'application
CMD ["python", "app.py"]
