import yfinance as yf
import os
import pandas as pd
import time
import json
from typing import Any
import sys
from zoneinfo import ZoneInfo
import mysql.connector
from datetime import datetime, timedelta
from typing import Optional
from mysql.connector import Error
from flask import Flask, request, jsonify
#################################################################################################################################################################################
#################################################################################################################################################################################
######################################################################## Récupérer les données de tout les symboles depuis Yahoo Finance #########################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################


# Télécharge les prix de clôture et les rendements journaliers pour toutes les actions listées dans le fichier JSON, puis les enregistre sous forme de fichiers CSV dans le dossier 'historique_action'.

def recuperer_prix_cloture2(start_date, end_date, json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        actions = json.load(f)

    os.makedirs('historique_action', exist_ok=True)

    nb_erreurs = 0
    nb_succes = 0
    erreurs_list = []

    for action in actions:
        nom_fichier = action['nom']
        symbole_complet = action['symbole']
        symbole = symbole_complet

        try:
            df = yf.download(symbole, start=start_date, end=end_date, progress=False, auto_adjust=True)

            if df.empty or 'Close' not in df.columns:
                print(f"NON : Donnees manquantes ou ticker invalide pour {symbole_complet}")
                nb_erreurs += 1
                erreurs_list.append(symbole_complet)
                continue

            df_cloture = pd.DataFrame()
            df_cloture['Date'] = df.index
            df_cloture['Cours_Cloture'] = df['Close'].values
            df_cloture['Rentabilite_Journaliere'] = df_cloture['Cours_Cloture'].pct_change()

            nom_fichier_nettoye = nom_fichier.replace("'", "-").replace("’", "-").replace(" ", "-")

            df_cloture.to_csv(
                f"historique_action\\{nom_fichier_nettoye}_cloture.csv",
                index=False,
                encoding='utf-8'
            )
            nb_succes += 1
            time.sleep(0.5)


        except Exception as e:
            print(f"NON : Erreur lors du telechargement pour {symbole_complet} : {e}")
            nb_erreurs += 1
            erreurs_list.append(symbole_complet)

    print(f"\nResume : {nb_succes} actions ajoutees, {nb_erreurs} erreurs.")
    if erreurs_list:
        print("Actions en erreur :", erreurs_list)


# Vérifie tous les fichiers CSV existants dans 'historique_action' et supprime ceux qui ne couvrent pas correctement la période spécifiée.

def verifier_et_supprimer_fichiers_tout2(date_debut, date_actuelle):
    dossier = "historique_action"
    date_min_ref = pd.to_datetime(date_debut) + timedelta(days=10)
    date_max_ref = pd.to_datetime(date_actuelle) - timedelta(days=10)

    fichiers = [f for f in os.listdir(dossier) if f.endswith(".csv")]

    supprimes = 0
    conserves = 0

    for fichier in fichiers:
        chemin = os.path.join(dossier, fichier)
        try:
            df = pd.read_csv(chemin, parse_dates=['Date'])
            if df.empty or 'Date' not in df.columns:
                print(f"⚠️ Fichier {fichier} vide ou sans colonne 'Date' -> Suppression")
                os.remove(chemin)
                supprimes += 1
                continue

            date_min = df['Date'].min()
            date_max = df['Date'].max()

            if date_min < date_min_ref and date_max > date_max_ref:
                conserves += 1
            else:
                print(f"NON : {fichier} supprime : date_min = {date_min.date()}, date_max = {date_max.date()}")
                os.remove(chemin)
                supprimes += 1

        except Exception as e:
            print(f"NON : Erreur avec le fichier {fichier} ({e}) -> Suppression")
            os.remove(chemin)
            supprimes += 1

# Filtre le fichier JSON pour ne conserver que les actions pour lesquelles un fichier CSV existe dans le dossier spécifié.
def filtrer_json_selon_csv2(json_file, dossier_csv):
    with open(json_file, 'r', encoding='utf-8') as f:
        actions = json.load(f)

    fichiers_csv = [f for f in os.listdir(dossier_csv) if f.endswith('.csv')]

    noms_csv = set()
    for fichier in fichiers_csv:
        if fichier.endswith('_cloture.csv'):
            nom_action = fichier[:-len('_cloture.csv')]
            noms_csv.add(nom_action)
    actions_filtrees = [action for action in actions if action['nom'] in noms_csv]
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(actions_filtrees, f, indent=2, ensure_ascii=False)

    print(f"{len(actions_filtrees)} actions gardees dans {json_file} selon les fichiers CSV.")

# Prépare une valeur pour l'insertion SQL en gérant les chaînes, les valeurs NULL et l’échappement des apostrophes.
def sql_val(val: Any) -> str:
    if val is None or val == "" or val == "NULL":
        return "NULL"
    if isinstance(val, str):
        return "'" + val.replace("'", "''") + "'"
    return str(val)


DB_CONFIG = {
    'host': 'localhost',
    'database': 'Optimisation',
    'user': 'root',
    'password': '',
    'charset': 'utf8'
}
# Initialise la base de données MySQL et crée la table 'actions' si elle n'existe pas.

def init_database2() -> None:
    """Initialise la table actions si elle n'existe pas."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS actions (
                id INT PRIMARY KEY AUTO_INCREMENT,
                nom_complet VARCHAR(100) NOT NULL,
                symbole VARCHAR(20) NOT NULL UNIQUE,
                secteur VARCHAR(50),
                pays VARCHAR(50),
                marche VARCHAR(50)
            ) CHARACTER SET utf8 COLLATE utf8_general_ci
        ''')
        conn.commit()
        print("Base de donnees MySQL initialisee : Optimisation")
    except Error as e:
        print(f"Erreur lors de l'initialisation de la base de donnees : {e}")
        raise
    finally:
        cursor.close()
        conn.close()

# Lit les actions depuis le fichier JSON, récupère les informations via Yahoo Finance,
# et insère les données dans la table MySQL 'actions', en gérant les erreurs et les tentatives multiples.

def enrichir_et_json_vers_sql2(json_file: str, max_retries: int = 3, delay: float = 0.1) -> None:
    """Charge les actions depuis JSON et les insère directement dans MySQL."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            actions = json.load(f)
        if not isinstance(actions, list):
            raise ValueError("Le fichier JSON doit contenir une liste d'objets.")
    except FileNotFoundError:
        print(f"Le fichier {json_file} n'existe pas.")
        return
    except json.JSONDecodeError as e:
        print(f"Erreur de formatage JSON dans {json_file} : {e}")
        return
    except ValueError as e:
        print(f"Erreur de validation des donnees JSON : {e}")
        return

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM actions;")
        cursor.execute("ALTER TABLE actions AUTO_INCREMENT = 1;")
        conn.commit()
        print("Table actions videe pour nouvelle insertion.")
    except Error as e:
        print(f"Erreur lors de la connexion ou suppression des donnees : {e}")
        return

    insert_count = 0
    for action in actions:
        symbole = action.get("symbole", "")
        nom_complet = action.get("nom", "")

        if not symbole or not isinstance(symbole, str):
            print(f"Symbole manquant ou invalide pour l'action : {nom_complet}")
            continue

        secteur = pays = marche = None
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbole)
                info = ticker.info
                secteur = info.get('sector')
                pays = info.get('country')
                marche = info.get('exchange')
                break
            except Exception as e:
                print(f"Tentative {attempt + 1}/{max_retries} echouee pour {symbole}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                else:
                    print(f"Echec definitif pour {symbole} après {max_retries} tentatives.")

        try:
            cursor.execute(
                f"INSERT INTO actions (nom_complet, symbole, secteur, pays, marche) "
                f"VALUES ({sql_val(nom_complet)}, {sql_val(symbole)}, {sql_val(secteur)}, {sql_val(pays)}, {sql_val(marche)})"
            )
            conn.commit()
            insert_count += 1
        except Error as e:
            print(f"NON : Erreur SQL pour {symbole} : {e}")
        time.sleep(delay)

    cursor.close()
    conn.close()
    print(f"Termine : {insert_count} actions inserees dans la base.")

 # Exécute automatiquement toutes les étapes : téléchargement des prix, nettoyage des CSV, filtrage du JSON et insertion en base MySQL.
def tout_faire2(start_date, end_date, json_file, sql_file, dossier_actions) :
    recuperer_prix_cloture2(start_date, end_date, json_file)
    verifier_et_supprimer_fichiers_tout2(start_date, end_date)
    filtrer_json_selon_csv2(json_file, dossier_actions)
    enrichir_et_json_vers_sql2(json_file)

#################################################################################################################################################################################
#################################################################################################################################################################################
################################ Mise à jour des fichiers CSV avec les prix boursiers les plus récents depuis Yahoo Finance (jusqu’à aujourd’hui) ################################
#################################################################################################################################################################################
#################################################################################################################################################################################

# Complète les fichiers CSV existants pour chaque action jusqu'à la veille du jour actuel, en téléchargeant uniquement les nouvelles données manquantes.
def completer_prix_csv_aujourdhui2(json_file):
    """Complète tous les CSV existants pour chaque action jusqu'à aujourd'hui."""
    with open(json_file, 'r', encoding='utf-8') as f:
        actions = json.load(f)

    dossier_csv = 'historique_action'
    os.makedirs(dossier_csv, exist_ok=True)

    nb_maj = 0
    nb_erreurs = 0
    erreurs_list = []

    date_veille = pd.to_datetime("today").normalize() - pd.Timedelta(days=1)


    for action in actions:
        nom_fichier = action['nom']
        symbole_complet = action['symbole']
        nom_fichier_nettoye = nom_fichier.replace("'", "-").replace("’", "-").replace(" ", "-")
        chemin_csv = os.path.join(dossier_csv, f"{nom_fichier_nettoye}_cloture.csv")

        try:
            if os.path.exists(chemin_csv):
                df_existing = pd.read_csv(chemin_csv, parse_dates=['Date'])
                if df_existing.empty:
                    start_date = pd.to_datetime("2005-01-01")
                else:
                    last_date = df_existing['Date'].max()
                    if last_date >= date_veille:
                        continue
                    start_date = last_date + timedelta(days=1)
            else:
                df_existing = pd.DataFrame()
                start_date = pd.to_datetime("2000-01-01")

            df_new = yf.download(symbole_complet, start=start_date, end=date_veille + timedelta(days=1),
                                 progress=False, auto_adjust=True)
            if df_new.empty or 'Close' not in df_new.columns:
                print(f"NON : Données manquantes pour {symbole_complet}")
                nb_erreurs += 1
                erreurs_list.append(symbole_complet)
                continue

            df_new_cloture = pd.DataFrame()
            df_new_cloture['Date'] = df_new.index
            df_new_cloture['Cours_Cloture'] = df_new['Close'].values
            df_new_cloture['Rentabilite_Journaliere'] = df_new_cloture['Cours_Cloture'].pct_change()

            df_final = pd.concat([df_existing, df_new_cloture], ignore_index=True)
            df_final.drop_duplicates(subset=['Date'], inplace=True)
            df_final.sort_values('Date', inplace=True)

            df_final['Rentabilite_Journaliere'] = df_final['Cours_Cloture'].pct_change()

            df_final.to_csv(chemin_csv, index=False, encoding='utf-8')

            nb_maj += 1
            time.sleep(0.5)

        except Exception as e:
            print(f"NON : Erreur lors de la mise à jour pour {symbole_complet} : {e}")
            nb_erreurs += 1
            erreurs_list.append(symbole_complet)

    print(f"\nRésumé : {nb_maj} fichiers mis à jour, {nb_erreurs} erreurs.")
    if erreurs_list:
        print("Actions en erreur :", erreurs_list)


#################################################################################################################################################################################
#################################################################################################################################################################################
######################################################################## Récupérer les données d’un seul symbole depuis Yahoo Finance ###########################################
#################################################################################################################################################################################
#################################################################################################################################################################################

# Télécharge les prix de clôture et rendements pour une seule action, et les enregistre dans un fichier CSV dédié dans 'historique_action'.

def recuperer_prix_cloture_1_symbole2(start_date: str, end_date: str, nom: str, symbole: str) -> Optional[str]:
    os.makedirs('historique_action', exist_ok=True)
    nom_fichier = nom
    print(f"Telechargement des donnees pour {nom_fichier} ({symbole})...")
    try:
        df = yf.download(symbole, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if df.empty or 'Close' not in df.columns:
            print(f"NON : Donnees manquantes ou ticker invalide pour {symbole}")
            return 0
        df_cloture = pd.DataFrame()
        df_cloture['Date'] = df.index
        df_cloture['Cours_Cloture'] = df['Close'].values
        df_cloture['Rentabilite_Journaliere'] = df_cloture['Cours_Cloture'].pct_change()
        nom_fichier_nettoye = nom_fichier.replace("'", "-").replace("’", "-").replace(" ", "-")
        fichier_csv = f"historique_action/{nom_fichier_nettoye}_cloture.csv"
        df_cloture.to_csv(fichier_csv, index=False, encoding='utf-8')
        print(f"OUI : {nom_fichier}_cloture.csv sauvegarde proprement.")
        return nom_fichier_nettoye
    except Exception as e:
        print(f"NON : Erreur lors du telechargement pour {symbole} : {e}")
        return 0

# Vérifie un fichier CSV pour une action spécifique et le supprime s’il ne couvre pas correctement la période demandée.

def verifier_et_supprimer_fichiers2(nom_fichier: str, start_date: str, end_date: str) -> int:
    date_min_ref = pd.to_datetime(start_date) + timedelta(days=10)
    date_max_ref = pd.to_datetime(end_date) - timedelta(days=10)
    fichier = f"historique_action/{nom_fichier}_cloture.csv"

    try:
        df = pd.read_csv(fichier, parse_dates=['Date'])
        if df.empty or 'Date' not in df.columns:
            print(f"⚠️ Fichier {fichier} vide ou sans colonne 'Date' -> Suppression")
            if os.path.exists(fichier):
                os.remove(fichier)
            return 0
        date_min = df['Date'].min()
        date_max = df['Date'].max()

        if date_min < date_min_ref and date_max > date_max_ref:
            print(f"OUI : {fichier} conserve : date_min = {date_min.date()}, date_max = {date_max.date()}")
            return 1
        else:
            print(f"NON : {fichier} supprime : date_min = {date_min.date()}, date_max = {date_max.date()}")
            if os.path.exists(fichier):
                os.remove(fichier)
            return 0
    except Exception as e:
        print(f"NON : Erreur avec le fichier {fichier} ({e}) -> Suppression")
        if os.path.exists(fichier):
            os.remove(fichier)
        return 0

# Ajoute une action (nom et symbole) au fichier JSON si elle n’y est pas déjà présente.

def ajouter_symbole_json2(nom: str, symbole: str, fichier_json: str) -> bool:
    try:
        try:
            with open(fichier_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                print(f"Le fichier {fichier_json} doit contenir une liste.")
                return False
        except FileNotFoundError:
            data = []
        except json.JSONDecodeError as e:
            print(f"Erreur de format JSON dans {fichier_json} : {e}")
            return False

        for entry in data:
            if entry.get("symbole") == symbole:
                print(f"Symbole {symbole} deja present dans {fichier_json}. Ignore.")
                return False

        new_entry = {"nom": nom, "symbole": symbole}
        data.append(new_entry)

        with open(fichier_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Ajoute {nom} ({symbole}) a {fichier_json}")
        return True
    except Exception as e:
        print(f"Erreur lors de l'ajout a {fichier_json} : {e}")
        return False

 # Récupère les informations d’une action via Yahoo Finance et l’insère dans la table MySQL 'actions'.

def ajouter_action_base_donnees2(nom: str, symbole: str) -> bool:

    try:

        ticker = yf.Ticker(symbole)
        info = ticker.info
        secteur = info.get('sector')
        pays = info.get('country')
        marche = info.get('exchange')
        print(f"Informations recuperees pour {symbole}: secteur={secteur}, pays={pays}, marche={marche}")

        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO actions (nom_complet, symbole, secteur, pays, marche)
            VALUES (%s, %s, %s, %s, %s)
        ''', (nom, symbole, secteur, pays, marche))
        conn.commit()
        print(f"Action ajoutee a la base de donnees : {nom} ({symbole})")
        return True
    except mysql.connector.errors.IntegrityError:
        print(f"Symbole {symbole} deja existant dans la base de donnees.")
        return False
    except Error as e:
        print(f"Erreur lors de l'ajout de {symbole} a la base de donnees : {e}")
        return False
    except Exception as e:
        print(f"Erreur Yahoo Finance pour {symbole} : {e}")
        return False
    finally:
        cursor.close()
        conn.close()

# Ajoute une action complète en automatisant toutes les étapes : téléchargement des données historiques, vérification du CSV, ajout au JSON et insertion dans la base MySQL.

def ajouter_action_complete2(nom: str, symbole: str, start_date: str, end_date: str, fichier_json: str = "euronext_nettoye.json") -> bool:
    try:
        init_database2()
    except Exception as e:
        print(f"echec de l'initialisation de la base de donnees : {e}")
        return False

    nom_fichier = recuperer_prix_cloture_1_symbole2(start_date, end_date, nom, symbole)
    if not nom_fichier:
        print(f"echec du telechargement des donnees historiques pour {nom} ({symbole}).")
        return False
    
    if not verifier_et_supprimer_fichiers2(nom_fichier, start_date, end_date):
        print(f"Le fichier CSV pour {nom} ({symbole}) ne respecte pas les critères de dates.")
        return False
    
    if not ajouter_symbole_json2(nom, symbole, fichier_json):
        print(f"echec de l'ajout de {nom} ({symbole}) au fichier JSON.")
        return False

    if not ajouter_action_base_donnees2(nom, symbole):
            print(f"echec de l'ajout de {nom} ({symbole}) a la base de donnees.")
            return False
    
    print(f"Action {nom} ({symbole}) ajoutee avec succès (JSON, base de donnees, donnees historiques).")
    return True

