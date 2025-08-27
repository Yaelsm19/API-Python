# test_app_local.py
import requests
import json

# URL locale où Flask tourne par défaut
url = "http://127.0.0.1:8000/optimiser"

# Exemple de données à envoyer
data = {
    "titres": ["AAPL", "GOOGL", "MSFT"],
    "date_debut": "2023-01-01",
    "date_fin": "2023-12-31",
    "methode": "sharpe",
    "taux_benchmark": 0.017
}

# Faire la requête POST
response = requests.post(url, json=data)

# Afficher la réponse
print("Status code:", response.status_code)
print("Réponse JSON:", json.dumps(response.json(), indent=2))
