# ----------- Importation des bibliothèques -----------------#



from flask import Flask, request, jsonify
from datetime import date, timedelta, datetime
from zoneinfo import ZoneInfo
import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import sys
import json
import yfinance as yf
import time
from typing import Any
import mysql.connector
from typing import Optional
from urllib import response
from matplotlib import pyplot as plt
import pandas_market_calendars as mcal
from sqlalchemy import null
from flask import Flask, request, jsonify
import io
import base64
from flask import Response
import signal
# ----------- Importation des fonctions-----------------#


from optimisation_portefeuille import calculer_rentabilite1, calculer_matrice_rentabilite1, calculer_covariance1, calculer_ecart_type1, calculer_matrice_covariance1, calculer_risque_portefeuille1, calculer_co_semi_variance1, calculer_matrice_semi_variance1, calculer_semi_risque_portefeuille1, calculer_skewness_matrice1, calculer_kurtosis_matrice1, utilite_exponentielle1, gradient_utilite1, maximiser_ratio_sharpe1, maximiser_ratio_sortino1, optimiser_utilite_CARA1, process_une_action1, process_tout_action1
from recuperer_donnes import recuperer_prix_cloture2, verifier_et_supprimer_fichiers_tout2, filtrer_json_selon_csv2, enrichir_et_json_vers_sql2, tout_faire2, completer_prix_csv_aujourdhui2, init_database2, recuperer_prix_cloture_1_symbole2, verifier_et_supprimer_fichiers2, ajouter_symbole_json2, ajouter_action_base_donnees2, ajouter_action_complete2
from simulation_simple import simuler_rendement4, charger_donnees4, est_jour_boursier4, verifier_presence_date4, get_prix_cloture4, calculer_rentabilite_1_titre4, calculer_rentabilite_n_titres4
from simulation_dynamique import simuler_rendement_long3, simuler_rendement_rapide3, maximiser_ratio_sharpe3, maximiser_ratio_sortino3, optimiser_utilite_CARA3, calculer_rentabilite3, calculer_matrice_rentabilite3, calculer_covariance3, calculer_matrice_covariance3, calculer_risque_portefeuille3, calculer_co_semi_variance3, calculer_matrice_semi_variance3, calculer_semi_risque_portefeuille3, calculer_skewness_matrice3, calculer_kurtosis_matrice3, utilite_exponentielle3, gradient_utilite3, charger_donnees3, est_jour_boursier3, verifier_presence_date3, get_prix_cloture3, calculer_rentabilite_1_titre3, calculer_rentabilite_n_titres3







app = Flask(__name__)

# ----------- GESTION DE TIMEOUT -----------------#

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Temps d'exécution dépassé")

@app.before_request
def before_request():
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(100)

@app.after_request
def after_request(response):
    signal.alarm(0)
    return response

@app.errorhandler(TimeoutException)
def handle_timeout(error):
    return jsonify({"error": "Temps d'exécution dépassé"}), 504






# ----------- ROUTE TEST -----------------#
# Route racine pour vérifier que Flask fonctionne
@app.route("/")
def home():
    return "Flask fonctionne !"

# ----------- ROUTE OPTIMISATION -----------------
# Route pour optimiser un portefeuille selon Sharpe, Sortino ou utilité CARA
@app.route("/optimiser", methods=["POST"])
def optimiser():
    try:
        data = request.get_json()
        titres = data.get("titres")
        date_debut = datetime.strptime(data.get("date_debut"), "%Y-%m-%d")
        date_fin = datetime.strptime(data.get("date_fin"), "%Y-%m-%d")
        methode = data.get("methode")
        taux_benchmark = float(data.get("taux_benchmark", 0.0))

        if methode not in ['sharpe', 'moment_ordre_superieur', 'sortino']:
            return jsonify({"erreur": "Méthode non reconnue."}), 400

        if methode == "sharpe":
            df_optimisation, df_portefeuille = maximiser_ratio_sharpe1(titres, date_debut, date_fin, taux_benchmark)
        elif methode == "moment_ordre_superieur":
            lamb = float(data.get("lambda"))
            df_optimisation, df_portefeuille = optimiser_utilite_CARA1(titres, date_debut, date_fin, lamb)
        else:
            df_optimisation, df_portefeuille = maximiser_ratio_sortino1(titres, date_debut, date_fin, taux_benchmark)

        df_process = process_tout_action1(titres, date_debut, date_fin)

        reponse = {
            "optimisation": df_optimisation.to_dict(orient="records"),
            "process": df_process.to_dict(orient="records"),
            "portefeuille": df_portefeuille.to_dict(orient="records")
        }

        return jsonify(reponse)

    except Exception as e:
        return jsonify({"erreur": str(e)}), 500

# ----------- ROUTES RECUPERATION -----------------
# Route pour récupérer toutes les actions et les insérer dans la base
@app.route("/recuperer_tout", methods=["GET"])
def recuperer_tout():
    date_debut = request.args.get("date_debut")
    tz_paris = ZoneInfo("Europe/Paris")
    date_actuelle = datetime.now(tz_paris).date()
    tout_faire2(date_debut, date_actuelle, "../euronext_nettoye.json", "sql_file", "../fichier_python/historique_action")
    return jsonify({"message": "Récupération terminée", "date_debut": date_debut, "date_fin": str(date_actuelle)})

# Route pour récupérer une seule action
@app.route("/recuperer_un", methods=["GET"])
def recuperer_un():
    date_debut = request.args.get("date_debut")
    nom_titre = request.args.get("nom_titre")
    symbole_titre = request.args.get("symbole_titre")
    tz_paris = ZoneInfo("Europe/Paris")
    date_actuelle = datetime.now(tz_paris).date()
    ajouter_action_complete2(nom_titre, symbole_titre, date_debut, date_actuelle)
    return jsonify({"message": f"Action {nom_titre} ({symbole_titre}) ajoutée", "date_debut": date_debut, "date_fin": str(date_actuelle)})

# Route pour compléter tous les CSV avec les prix du jour
@app.route("/completer_tout", methods=["GET"])
def completer_tout():
    completer_prix_csv_aujourdhui2("../euronext_nettoye.json")
    return jsonify({"message": "Complétion terminée"})

# ----------- ROUTE SIMULATION DYNAMIQUE -----------------
# Route pour lancer une simulation dynamique (rolling window) avec différentes méthodes
@app.route("/simulation_dynamique", methods=["GET"])
def simulation_dynamique_api():
    try:
        date_debut = datetime.strptime(request.args.get("date_debut"), "%Y-%m-%d")
        date_fin = datetime.strptime(request.args.get("date_fin"), "%Y-%m-%d")
        duree_estimation = int(request.args.get("duree_estimation"))
        duree_investissement = int(request.args.get("duree_investissement"))
        titres = request.args.get("titres").split(',')
        niveau_risque = float(request.args.get("niveau_risque"))
        indice = request.args.get("indice")
        user_id = request.args.get("user_id")
        montant = float(request.args.get("montant"))
        taux_sans_risque_annuel = float(request.args.get("taux_sans_risque_annuel"))
        taux_benchmark = float(request.args.get("taux_benchmark"))
        nom_simulation = request.args.get("nom_simulation")
        graphique_option = request.args.get("graphique_option")
        methode = request.args.get("methode") 
        lamb = request.args.get("lambda", None)
        if lamb is not None:
            lamb = float(lamb)
    except Exception as e:
        return jsonify({"error": f"Paramètres incorrects: {str(e)}"}), 400

    if methode not in ['sharpe', 'moment_ordre_superieur', 'sortino']:
        return jsonify({"error": "Méthode non reconnue. Utilisez 'sharpe', 'moment_ordre_superieur' ou 'sortino'."}), 400

    try:
        if methode == "sharpe":
            if graphique_option == "rapide":
                reponse = simuler_rendement_rapide3(date_debut, date_fin, duree_estimation,
                                                    duree_investissement, titres, niveau_risque,
                                                    indice, montant, taux_sans_risque_annuel,
                                                    taux_benchmark, user_id, nom_simulation, methode, None)
            else:
                reponse = simuler_rendement_long3(date_debut, date_fin, duree_estimation,
                                                 duree_investissement, titres, niveau_risque,
                                                 indice, montant, taux_sans_risque_annuel,
                                                 taux_benchmark, user_id, nom_simulation, methode, None)
        elif methode == "moment_ordre_superieur":
            if lamb is None:
                return jsonify({"error": "Le paramètre lambda est requis pour la méthode 'moment_ordre_superieur'"}), 400
            if graphique_option == "rapide":
                reponse = simuler_rendement_rapide3(date_debut, date_fin, duree_estimation,
                                                    duree_investissement, titres, niveau_risque,
                                                    indice, montant, taux_sans_risque_annuel,
                                                    taux_benchmark, user_id, nom_simulation, methode, lamb)
            else:
                reponse = simuler_rendement_long3(date_debut, date_fin, duree_estimation,
                                                 duree_investissement, titres, niveau_risque,
                                                 indice, montant, taux_sans_risque_annuel,
                                                 taux_benchmark, user_id, nom_simulation, methode, lamb)
        else: 
            if graphique_option == "rapide":
                reponse = simuler_rendement_rapide3(date_debut, date_fin, duree_estimation,
                                                    duree_investissement, titres, niveau_risque,
                                                    indice, montant, taux_sans_risque_annuel,
                                                    taux_benchmark, user_id, nom_simulation, methode, None)
            else:
                reponse = simuler_rendement_long3(date_debut, date_fin, duree_estimation,
                                                 duree_investissement, titres, niveau_risque,
                                                 indice, montant, taux_sans_risque_annuel,
                                                 taux_benchmark, user_id, nom_simulation, methode, None)
    except Exception as e:
        return jsonify({"error": f"Erreur pendant la simulation: {str(e)}"}), 500

    return jsonify(reponse)

# ----------- ROUTE SIMULATION SIMPLE -----------------
# Route pour lancer une simulation simple avec un portefeuille fixe
@app.route("/simulation_simple", methods=["GET"])
def simulation_simple_api():
    try:
        date_debut = datetime.strptime(request.args.get("date_debut"), "%Y-%m-%d")
        date_fin = datetime.strptime(request.args.get("date_fin"), "%Y-%m-%d")
        w = request.args.get("w")
        titres = request.args.get("titres").split(',')
        poids_str = float(request.args.get("poids_str"))
        indice = request.args.get("indice")
        user_id = request.args.get("user_id")
        montant = float(request.args.get("montant", 1000))
        if montant <= 0:
            return jsonify({"error": "Montant doit être positif."}), 400
        taux_sans_risque = float(request.args.get("taux_sans_risque", 0.000092))
        nom_simulation = request.args.get("nom_simulation")

        w_liste = [float(p.strip())/100 for p in w.split(',')]

        reponse = simuler_rendement4(date_debut, date_fin, montant, w_liste, titres, poids_str, taux_sans_risque, indice, user_id, nom_simulation)

    except Exception as e:
        return jsonify({"error": f"Paramètres invalides ou erreur: {str(e)}"}), 400

    return jsonify(reponse)

# ----------- BOUCLE PRINCIPALE -----------------
# Démarre l'application Flask sur le port défini dans l'environnement ou 8000
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
