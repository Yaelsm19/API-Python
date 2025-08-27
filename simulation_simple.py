from urllib import response
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime
import pandas_market_calendars as mcal
import os
import sys
import json
from flask import Flask, request, jsonify


#################################################################################################################################################################################
#################################################################################################################################################################################
######################################################### Récupération des données pour tout les titres #########################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################

def charger_donnees4(symboles):
    donnees = {}
    for symbole in symboles:
        chemin = f"historique_action/{symbole}_cloture.csv"
        try:
            df = pd.read_csv(chemin, encoding='utf-8', parse_dates=['Date'], index_col='Date')
            if 'Cours_Cloture' not in df.columns:
                print(f"Colonne 'Cours_Cloture' introuvable pour {symbole}.")
                continue
            donnees[symbole] = df
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {symbole}: {e}")
    return donnees

#################################################################################################################################################################################
#################################################################################################################################################################################
######################################################### Vérification de la disponibilité des dates ############################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################

def est_jour_boursier4(date):
    cal = mcal.get_calendar('XPAR')
    return cal.valid_days(start_date=date, end_date=date).size > 0


def verifier_presence_date4(symboles, date_str, donnees_symboles):
    date = pd.to_datetime(date_str)
    for symbole in symboles:
        if symbole not in donnees_symboles:
            print(f"Les données pour {symbole} n'ont pas été chargées.")
            return False
        
        df = donnees_symboles[symbole]
        if date not in df.index:
            return False
    
    return True

#################################################################################################################################################################################
#################################################################################################################################################################################
######################################################### Récupération du prix de cloture d'un symbole ##########################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################

def get_prix_cloture4(symbole, date_str, donnees_symboles):
    try:
        df = donnees_symboles[symbole]
        date = pd.to_datetime(date_str)

        if date in df.index:
            return df.loc[date, 'Cours_Cloture']
        else:
            return None
    except Exception as e:
        print("Erreur lors de la récupération :", e)
        return None
    
#################################################################################################################################################################################
#################################################################################################################################################################################
######################################################### Calcul de rentabilité entre deux dates ################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################

def calculer_rentabilite_1_titre4(symbole, start_date, end_date, donnees_symboles) :
    prix_debut = get_prix_cloture4(symbole, start_date, donnees_symboles)
    prix_fin = get_prix_cloture4(symbole, end_date, donnees_symboles)
    if prix_debut==None or prix_fin==None :
        return 0
    rentabilite = (prix_fin-prix_debut)/prix_debut
    return rentabilite


def calculer_rentabilite_n_titres4(poids, symboles, start_date, end_date, donnees_symboles) :
    rentabilite_total = 0
    for i in range(len(symboles)) :
        rentabilite_total += poids[i] * calculer_rentabilite_1_titre4(symboles[i], start_date, end_date, donnees_symboles)
    return rentabilite_total

#################################################################################################################################################################################
#################################################################################################################################################################################
######################################################### Simulation d'investissement d'un portefeuille #########################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################

def simuler_rendement4(start_date, end_date, montant, poids_symboles, symboles, poids_risque, taux_sans_risque, indice, user_id, nom_simulation):
    titres = [indice] + symboles
    donnees_titres = charger_donnees4(titres)

    montant_risque = montant * poids_risque
    montant_sans_risque = montant * (1 - poids_risque)

    valeurs_titres = {symbole: montant_risque * poids_symboles[i] for i, symbole in enumerate(symboles)}

    valeur_tresorerie = montant_sans_risque

    valeur_portefeuille = {}
    valeur_indice = {}

    date_actuelle = pd.to_datetime(start_date)
    ancienne_date = date_actuelle
    date_actuelle += pd.Timedelta(days=1)
    date_fin = pd.to_datetime(end_date)

    valeur_actuelle_indice = montant

    while not verifier_presence_date4(symboles, ancienne_date, donnees_titres):
        ancienne_date += pd.Timedelta(days=1)
        date_actuelle = ancienne_date + pd.Timedelta(days=1)

    while date_actuelle <= date_fin:
        if est_jour_boursier4(date_actuelle) and verifier_presence_date4(symboles, date_actuelle, donnees_titres):
            valeur_portefeuille[ancienne_date] = sum(valeurs_titres.values()) + valeur_tresorerie
            valeur_indice[ancienne_date] = valeur_actuelle_indice

            for symbole in symboles:
                rendement = calculer_rentabilite_1_titre4(symbole, ancienne_date, date_actuelle, donnees_titres)
                valeurs_titres[symbole] *= (1 + rendement)

            nb_jours = (date_actuelle - ancienne_date).days
            valeur_tresorerie *= (1 + taux_sans_risque) ** (nb_jours/365)

            rendement_indice = calculer_rentabilite_1_titre4(indice, ancienne_date, date_actuelle, donnees_titres)
            valeur_actuelle_indice *= (1 + rendement_indice)

            ancienne_date = date_actuelle

        date_actuelle += pd.Timedelta(days=1)

    valeur_portefeuille[ancienne_date] = sum(valeurs_titres.values()) + valeur_tresorerie
    valeur_indice[ancienne_date] = valeur_actuelle_indice

    df_portefeuille = pd.DataFrame(list(valeur_portefeuille.items()), columns=['Date', 'Valeur'])
    df_portefeuille['Date'] = pd.to_datetime(df_portefeuille['Date'])
    df_indice = pd.DataFrame(list(valeur_indice.items()), columns=['Date', 'Valeur'])
    df_indice['Date'] = pd.to_datetime(df_indice['Date'])

    rentabilite_portefeuille_absolue = (df_portefeuille['Valeur'].iloc[-1] - df_portefeuille['Valeur'].iloc[0]) / df_portefeuille['Valeur'].iloc[0]
    rentabilite_indice_absolue = (df_indice['Valeur'].iloc[-1] - df_indice['Valeur'].iloc[0]) / df_indice['Valeur'].iloc[0]

    nombre_jours_boursiers = len(df_portefeuille)
    duree_annees = (df_portefeuille['Date'].iloc[-1] - df_portefeuille['Date'].iloc[0]).days / 365.25

    if duree_annees <= 0:
        rentabilite_portefeuille_annuelle = 0
        rentabilite_indice_annuelle = 0
    else:
        rentabilite_portefeuille_annuelle = (df_portefeuille['Valeur'].iloc[-1] / df_portefeuille['Valeur'].iloc[0]) ** (1 / duree_annees) - 1
        rentabilite_indice_annuelle = (df_indice['Valeur'].iloc[-1] / df_indice['Valeur'].iloc[0]) ** (1 / duree_annees) - 1

    def max_drawdown(series):
        roll_max = series.cummax()
        drawdown = (series - roll_max) / roll_max
        return drawdown.min()

    max_drawdown_portefeuille = max_drawdown(df_portefeuille['Valeur'])
    max_drawdown_indice = max_drawdown(df_indice['Valeur'])

    valeur_minimum_portefeuille = df_portefeuille['Valeur'].min()
    valeur_maximum_portefeuille = df_portefeuille['Valeur'].max()
    valeur_minimum_indice = df_indice['Valeur'].min()
    valeur_maximum_indice = df_indice['Valeur'].max()

    plt.figure(figsize=(10, 6))
    plt.plot(df_portefeuille['Date'], df_portefeuille['Valeur'], color='b', label='Valeur du portefeuille')
    plt.plot(df_indice['Date'], df_indice['Valeur'], color='r', label="Valeur de l'indice")
    plt.title("Évolution de la valeur du portefeuille comparé à l'indice du marché")
    plt.xlabel("Date")
    plt.ylabel("Valeur (€)")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    dossier = "image/graphique"
    os.makedirs(dossier, exist_ok=True)
    nom_fichier = f"simulation_rendement_{user_id}_{nom_simulation}.png"
    chemin_fichier = os.path.join(dossier, nom_fichier)
    plt.savefig(chemin_fichier, dpi=300)
    plt.close()

    return {
        "rentabilite_portefeuille_absolue": rentabilite_portefeuille_absolue/100,
        "rentabilite_indice_absolue": rentabilite_indice_absolue/100,
        "rentabilite_portefeuille_annuelle": rentabilite_portefeuille_annuelle/100,
        "rentabilite_indice_annuelle": rentabilite_indice_annuelle/100,
        "max_drawdown_portefeuille": max_drawdown_portefeuille/100,
        "max_drawdown_indice": max_drawdown_indice/100,
        "nombre_jours_boursiers": nombre_jours_boursiers,
        "valeur_minimum_portefeuille": valeur_minimum_portefeuille,
        "valeur_maximum_portefeuille": valeur_maximum_portefeuille,
        "valeur_minimum_indice": valeur_minimum_indice,
        "valeur_maximum_indice": valeur_maximum_indice,
    }

#################################################################################################################################################################################
#################################################################################################################################################################################
######################################################### Traitement des arguments ##############################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################
