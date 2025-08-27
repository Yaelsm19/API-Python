from urllib import response
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime
import pandas_market_calendars as mcal
import os
import sys
import json
from datetime import date, timedelta, datetime
import numpy as np
from scipy.optimize import minimize
from sqlalchemy import null

from flask import Flask, request, jsonify



#################################################################################################################################################################################
#################################################################################################################################################################################
################################################################# Calcul de la rentabilité ######################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################

def calculer_rentabilite3(symbole, start_date, end_date):
    df = pd.read_csv(
        f"../fichier_python/historique_action/{symbole}_cloture.csv",
        encoding='utf-8',
        parse_dates=['Date'],
        index_col='Date'
    )
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    df_periode = df.loc[start_date:end_date]

    if df_periode.empty:
        print(f"Aucune donnée disponible pour {symbole} entre {start_date.date()} et {end_date.date()}")
        return None

    moyenne_rentabilite = df_periode['Rentabilite_Journaliere'].mean()
    return moyenne_rentabilite

def calculer_matrice_rentabilite3(symboles, start_date, end_date):
    rentabilites = []
    for s in symboles:
        r = calculer_rentabilite3(s, start_date, end_date)
        if r is None:
            print(f"Attention : rentabilité manquante pour {s}, valeur 0 par défaut")
            r = 0.0
        rentabilites.append(r)
    return np.array(rentabilites)

#################################################################################################################################################################################
#################################################################################################################################################################################
################################################################# Calcul de la variance, covariance, écart-type #################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################

def calculer_covariance3(symbole1, symbole2, start_date, end_date):
    df1 = pd.read_csv(
        rf"../fichier_python/historique_action/{symbole1}_cloture.csv",
        encoding='utf-8',
        parse_dates=['Date'],
        index_col='Date'
    )
    df2 = pd.read_csv(
        rf"../fichier_python/historique_action/{symbole2}_cloture.csv",
        encoding='utf-8',
        parse_dates=['Date'],
        index_col='Date'
    )

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    df1 = df1.loc[start_date:end_date]
    df2 = df2.loc[start_date:end_date]

    df = pd.concat([
        df1['Rentabilite_Journaliere'],
        df2['Rentabilite_Journaliere']
    ], axis=1, join='inner', keys=[symbole1, symbole2]).dropna()

    if len(df) < 2:
        return 0.0

    s1 = df[symbole1]
    s2 = df[symbole2]

    if symbole1 == symbole2:
        var = s1.var(ddof=1)
        if hasattr(var, "__len__") and not np.isscalar(var):
            var = var.iloc[0] if len(var) > 0 else 0.0
        return float(var) if np.isscalar(var) and not np.isnan(var) else 0.0

    cov = s1.cov(s2)
    return float(cov) if np.isscalar(cov) and not np.isnan(cov) else 0.0


def calculer_matrice_covariance3(symboles, start_date, end_date):
    n = len(symboles)
    matrice_covariance = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            cov = calculer_covariance3(symboles[i], symboles[j], start_date, end_date)
            matrice_covariance[i][j] = cov
            matrice_covariance[j][i] = cov
    return matrice_covariance

def calculer_risque_portefeuille(w, symboles, start_date, end_date):
    w_transpose = np.transpose(w)
    matrice_covariance = calculer_matrice_covariance3(symboles, start_date, end_date)
    variance_portefeuille = np.dot(np.dot(w_transpose, matrice_covariance), w)
    return variance_portefeuille**0.5

#################################################################################################################################################################################
#################################################################################################################################################################################
################################################################# Calcul de la semi-variance, semi-covariance, semi-écart-type ##################################################
#################################################################################################################################################################################
#################################################################################################################################################################################

def calculer_co_semi_variance3(symbole1, symbole2, start_date, end_date, taux_benchmark):
    taux_benchmark_journalier = (1+taux_benchmark)**(1/365)-1
    df1 = pd.read_csv(
        f"../fichier_python/historique_action/{symbole1}_cloture.csv",
        encoding='utf-8',
        parse_dates=['Date'],
        index_col='Date'
    )
    df2 = None
    if symbole1 == symbole2:
        df2 = df1.copy()
    else:
        df2 = pd.read_csv(
            f"../fichier_python/historique_action/{symbole2}_cloture.csv",
            encoding='utf-8',
            parse_dates=['Date'],
            index_col='Date'
        )

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    df1 = df1.loc[start_date:end_date]
    df2 = df2.loc[start_date:end_date]

    if symbole1 == symbole2:
        df = df1[['Rentabilite_Journaliere']].copy()
        df.columns = [symbole1]
    else:
        df = pd.concat([
            df1['Rentabilite_Journaliere'],
            df2['Rentabilite_Journaliere']
        ], axis=1, join='inner')
        df.columns = [symbole1, symbole2]

    df = df.dropna()

    T = len(df)
    if T < 2:
        print("Pas assez de données après nettoyage")
        return 0.0

    r1 = df[symbole1]
    if symbole1 == symbole2:
        r2 = r1
    else:
        r2 = df[symbole2]

    d1 = np.minimum(r1 - taux_benchmark_journalier, 0)
    d2 = np.minimum(r2 - taux_benchmark_journalier, 0)

    co_semi_var = (d1 * d2).sum() / (T - 1)

    if pd.isna(co_semi_var):
        co_semi_var = 0.0

    return float(co_semi_var)


def calculer_matrice_semi_variance3(symboles, start_date, end_date, taux_benchmark):
    n = len(symboles)
    matrice_covariance = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            cov = calculer_co_semi_variance3(symboles[i], symboles[j], start_date, end_date, taux_benchmark)
            matrice_covariance[i][j] = cov
            matrice_covariance[j][i] = cov
    return matrice_covariance



def calculer_semi_risque_portefeuille3(w, symboles, start_date, end_date, taux_benchmark):
    w_transpose = np.transpose(w)
    matrice_covariance = calculer_matrice_semi_variance3(symboles, start_date, end_date, taux_benchmark)
    variance_portefeuille = np.dot(np.dot(w_transpose, matrice_covariance), w)
    return variance_portefeuille**0.5

#################################################################################################################################################################################
#################################################################################################################################################################################
################################################################################# Calcul de la matrice du skewness ##############################################################
#################################################################################################################################################################################
#################################################################################################################################################################################

def calculer_skewness_matrice3(symboles, start_date, end_date):
    donnees = []
    for symbole in symboles:
        chemin = rf"../fichier_python/historique_action/{symbole}_cloture.csv"
        df = pd.read_csv(chemin, parse_dates=['Date'], index_col='Date', encoding='utf-8')

        if 'Rentabilite_Journaliere' not in df.columns:
            df['Rentabilite_Journaliere'] = df['Cours_Cloture'].pct_change()

        df = df.loc[start_date:end_date]['Rentabilite_Journaliere'].dropna()
        donnees.append(df)

    df_rendements = pd.concat(donnees, axis=1, join='inner')
    if df_rendements.empty:
        raise ValueError("Aucune donnée commune aux symboles sur la période donnée.")

    df_rendements.columns = symboles

    rendements = df_rendements.T.values
    n_actifs, n_jours = rendements.shape



    rendements_centres = rendements - rendements.mean(axis=1, keepdims=True)


    coskew = np.zeros((n_actifs, n_actifs, n_actifs))
    for t in range(n_jours):
        r_t = rendements_centres[:, t]
        coskew += np.einsum('i,j,k->ijk', r_t, r_t, r_t)
    coskew /= n_jours

    return coskew

#################################################################################################################################################################################
#################################################################################################################################################################################
######################################################################## Calcul de la matrice du kurtosis #######################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################

def calculer_kurtosis_matrice3(symboles, start_date, end_date):
    donnees = []
    for symbole in symboles:
        chemin = rf"../fichier_python/historique_action/{symbole}_cloture.csv"
        df = pd.read_csv(chemin, parse_dates=['Date'], index_col='Date', encoding='utf-8')

        if 'Rentabilite_Journaliere' not in df.columns:
            df['Rentabilite_Journaliere'] = df['Cours_Cloture'].pct_change()

        df = df.loc[start_date:end_date]['Rentabilite_Journaliere'].dropna()
        donnees.append(df)

    df_rendements = pd.concat(donnees, axis=1, join='inner')
    if df_rendements.empty:
        raise ValueError("Aucune donnée commune aux symboles sur la période donnée.")

    df_rendements.columns = symboles

    rendements = df_rendements.T.values
    n_actifs, n_jours = rendements.shape

    rendements_centres = rendements - rendements.mean(axis=1, keepdims=True)

    cokurt = np.zeros((n_actifs, n_actifs, n_actifs, n_actifs))
    for t in range(n_jours):
        r_t = rendements_centres[:, t]
        cokurt += np.einsum('i,j,k,l->ijkl', r_t, r_t, r_t, r_t)
    cokurt /= n_jours

    return cokurt

#################################################################################################################################################################################
#################################################################################################################################################################################
######################################################### Méthode ratio de sharpe ###############################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################

def maximiser_ratio_sharpe3(symboles, start_date, end_date, taux_benchmark):
    taux_benchmark_journalier = (1 + taux_benchmark)**(1/365) - 1
    rendements = calculer_matrice_rentabilite3(symboles, start_date, end_date).flatten()
    covariance = calculer_matrice_covariance3(symboles, start_date, end_date)
    n = len(rendements)

    def ratio_sharpe3(w):
        rendement = np.dot(w, rendements)
        risque = np.sqrt(np.dot(w.T, np.dot(covariance, w)))
        if risque == 0:
            return 0
        return (rendement - taux_benchmark_journalier) / risque

    def gradient_ratio_sharpe3(w):
        w = np.array(w)
        rendement = np.dot(w, rendements)
        risque_carre = np.dot(w.T, np.dot(covariance, w))
        risque = np.sqrt(risque_carre)

        if risque == 0:
            return np.zeros_like(w)

        num1 = rendements * risque
        num2 = np.dot(rendements, w) - taux_benchmark_journalier
        num3 = np.dot(covariance, w) * (num2 / risque)
        gradient = (num1 - num3) / risque_carre
        return -gradient

    contrainte_somme = {
        'type': 'eq',
        'fun': lambda w: float(np.sum(w) - 1)
    }

    bornes = [(0, 1) for _ in range(n)]
    w0 = np.ones(n) / n

    resultat = minimize(
        lambda w: -ratio_sharpe3(w),
        w0,
        method='SLSQP',
        bounds=bornes,
        constraints=contrainte_somme,
        jac=gradient_ratio_sharpe3
    )

    if resultat.success:
        poids = resultat.x
        return poids
    else:
        raise ValueError("Échec de l'optimisation :", resultat.message)
    
#################################################################################################################################################################################
#################################################################################################################################################################################
################################################################# Méthode ratio de sortino ######################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################

def maximiser_ratio_sortino3(symboles, start_date, end_date, taux_benchmark):
    taux_benchmark_journalier = ( 1 + taux_benchmark)**(1/365)-1
    rendement_matrice = calculer_matrice_rentabilite3(symboles, start_date, end_date).flatten()
    covariance_matrice = calculer_matrice_semi_variance3(symboles, start_date, end_date, taux_benchmark)
    n = len(rendement_matrice)

    def ratio_sortino3(w):
        rendement = np.dot(w, rendement_matrice)
        risque = np.sqrt(np.dot(w.T, np.dot(covariance_matrice, w)))
        return (rendement - taux_benchmark_journalier) / risque

    def gradient_ratio_sortino3(w):
        w = np.array(w)
        risque_carre = np.dot(w.T, np.dot(covariance_matrice, w))
        risque = np.sqrt(risque_carre)

        num1 = rendement_matrice * risque
        num2 = np.dot(w.T, rendement_matrice - taux_benchmark_journalier)
        num3 = np.dot(covariance_matrice, w) * (num2 / risque)
        gradient = (num1 - num3) / risque_carre
        return -gradient

    contrainte_somme = {
        'type': 'eq',
        'fun': lambda w: float(np.sum(w) - 1)
    }

    bornes = [(0, 1) for _ in range(n)]
    w0 = np.ones(n) / n

    resultat = minimize(
        lambda w: -ratio_sortino3(w),
        w0,
        method='SLSQP',
        bounds=bornes,
        constraints=contrainte_somme,
        jac=gradient_ratio_sortino3
    )

    if resultat.success:
        poids = resultat.x
        return poids
    else:
        raise ValueError("Échec de l'optimisation :", resultat.message)

#################################################################################################################################################################################
#################################################################################################################################################################################
######################################################### Méthode utilite exponentielle #########################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################

def gradient_utilite3(w, lambd, matrice_rentabilite, matrice_covariance, matrice_skewness, matrice_kurtosis):
    n = len(w)
    E_w = np.dot(w.T, matrice_rentabilite)
    M_w = np.dot(np.dot(w.T, matrice_covariance), w)
    S_w = np.einsum('i,j,k,ijk->', w, w, w, matrice_skewness)
    K_w = np.einsum('i,j,k,l,ijkl->', w, w, w, w, matrice_kurtosis)
    ajustement = 1 + (lambd**2 / 2) * M_w - (lambd**3 / 6) * S_w + (lambd**4 / 24) * K_w
    
    grad_E_w = matrice_rentabilite.flatten()
    grad_M_w = 2 * np.dot(matrice_covariance, w)
    grad_S_w = 3 * np.einsum('j,k,ijk->i', w, w, matrice_skewness)
    grad_K_w = 4 * np.einsum('j,k,l,ijkl->i', w, w, w, matrice_kurtosis)
    
    grad_A = (lambd**2 / 2) * grad_M_w - (lambd**3 / 6) * grad_S_w + (lambd**4 / 24) * grad_K_w
    term1 = -lambd * grad_E_w * ajustement
    grad_U = np.exp(-lambd * E_w) * (term1 + grad_A)
    
    return grad_U
def utilite_exponentielle3(w, lambd, matrice_rentabilite, matrice_covariance, matrice_skewness, matrice_kurtosis):
    E_w = np.dot(w.T, matrice_rentabilite)
    M_w = np.dot(np.dot(w.T, matrice_covariance), w)
    S_w = np.einsum('i,j,k,ijk->', w, w, w, matrice_skewness)
    K_w = np.einsum('i,j,k,l,ijkl->', w, w, w, w, matrice_kurtosis)

    ajustement = 1 + ((lambd**2) / 2) * M_w - ((lambd**3) / 6) * S_w + ((lambd**4) / 24) * K_w
    utilite = -np.exp(-lambd * E_w) * ajustement
    return -utilite


def optimiser_utilite_CARA3(symboles, start_date, end_date, lambd):
    matrice_rentabilite = calculer_matrice_rentabilite3(symboles, start_date, end_date).flatten()
    matrice_covariance = calculer_matrice_covariance3(symboles, start_date, end_date)
    matrice_skewness = calculer_skewness_matrice3(symboles, start_date, end_date)
    matrice_kurtosis = calculer_kurtosis_matrice3(symboles, start_date, end_date)
    n = len(symboles)
    w0 = np.ones(n) / n
    contraintes = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1, 'jac': lambda w: np.ones_like(w)}]
    bornes = [(0, 1) for _ in range(n)]

    def objectif(w, lambd, matrice_rentabilite, matrice_covariance, matrice_skewness, matrice_kurtosis):
        return utilite_exponentielle3(w, lambd, matrice_rentabilite, matrice_covariance, matrice_skewness, matrice_kurtosis)

    resultat = minimize(objectif, w0, method='SLSQP',
                        jac=gradient_utilite3,
                        args=(lambd, matrice_rentabilite, matrice_covariance, matrice_skewness, matrice_kurtosis),
                        bounds=bornes, constraints=contraintes,
                        options={'disp': False, 'maxiter': 1000})

    if not resultat.success:
        raise ValueError("Optimisation échouée : " + resultat.message)
    return resultat.x, -resultat.fun

#################################################################################################################################################################################
#################################################################################################################################################################################
######################################################### Récupération des données pour tout les titres #########################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################

def charger_donnees3(symboles):
    donnees = {}
    for symbole in symboles:
        chemin = rf"../fichier_python/historique_action/{symbole}_cloture.csv"
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
######################################################### Vérification de la disponibilité des dates ################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################

def est_jour_boursier3(date):
    cal = mcal.get_calendar('XPAR')
    return cal.valid_days(start_date=date, end_date=date).size > 0


def verifier_presence_date3(symboles, date_str, donnees_symboles):
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

def get_prix_cloture3(symbole, date_str, donnees_symboles):
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

def calculer_rentabilite_1_titre3(symbole, start_date, end_date, donnees_symboles) :
    prix_debut = get_prix_cloture3(symbole, start_date, donnees_symboles)
    prix_fin = get_prix_cloture3(symbole, end_date, donnees_symboles)
    if prix_debut==None or prix_fin==None :
        return 0
    rentabilite = (prix_fin-prix_debut)/prix_debut
    return rentabilite


def calculer_rentabilite_n_titres3(poids, symboles, start_date, end_date, donnees_symboles) :
    rentabilite_total = 0
    for i in range(len(symboles)) :
        rentabilite_total += poids[i] * calculer_rentabilite_1_titre3(symboles[i], start_date, end_date, donnees_symboles)
    return rentabilite_total

#################################################################################################################################################################################
#################################################################################################################################################################################
############################################# Simulation d'investissement dynamique d'un portefeuille avec un graphique précis ###################################################
#################################################################################################################################################################################
#################################################################################################################################################################################

def simuler_rendement_long3(start_date, end_date, duree_estimation, duree_investissement, symboles, poids_risque, indice, montant, taux_sans_risque, taux_benchmark, user_id, nom_simulation, methode, lamb):
    titres = [indice] + symboles
    donnees_titres = charger_donnees3(titres)
    fin_investissement = pd.to_datetime(end_date)
    debut_investissement_periode = pd.to_datetime(start_date)

    valeur_portefeuille_total = {}
    valeur_indice_total = {}

    montant_portefeuille = montant
    valeur_actuelle_indice = montant

    ancienne_date = debut_investissement_periode
    while not verifier_presence_date3(titres, ancienne_date, donnees_titres):
        ancienne_date += pd.Timedelta(days=1)
    premiere_date = ancienne_date
    date_actuelle = ancienne_date + pd.Timedelta(days=1)
    valeurs_titres = {}
    debut_investissement_periode = ancienne_date
    if verifier_presence_date3(titres, premiere_date, donnees_titres):
        valeur_portefeuille_total[premiere_date] = montant
        valeur_indice_total[premiere_date] = montant
    while debut_investissement_periode < fin_investissement:
        fin_estimation_periode = debut_investissement_periode - pd.Timedelta(days=1)
        debut_estimation_periode = fin_estimation_periode - pd.Timedelta(days=int(duree_estimation))
        fin_investissement_periode = debut_investissement_periode + pd.Timedelta(days=int(duree_investissement))
        if fin_investissement_periode > fin_investissement:
            fin_investissement_periode = fin_investissement

        montant_risque = montant_portefeuille * poids_risque
        montant_sans_risque = montant_portefeuille * (1 - poids_risque)
        while not verifier_presence_date3(titres, fin_estimation_periode, donnees_titres):
            fin_estimation_periode -= pd.Timedelta(days=1)
            if fin_estimation_periode < debut_estimation_periode:
                break
        while not verifier_presence_date3(titres, debut_estimation_periode, donnees_titres):
            debut_estimation_periode += pd.Timedelta(days=1)
            if debut_estimation_periode > debut_investissement_periode:
                break
        if methode == "sharpe" :
            poids_symboles = maximiser_ratio_sharpe3(symboles, debut_estimation_periode, fin_estimation_periode, taux_benchmark)
        elif methode == "moment_ordre_superieur" :
            poids_symboles, score = optimiser_utilite_CARA3(symboles, debut_estimation_periode, fin_estimation_periode, lamb)
        elif methode == 'sortino' :
            poids_symboles = maximiser_ratio_sortino3(symboles, debut_estimation_periode, fin_estimation_periode, taux_benchmark)
        valeurs_titres = {symbole: montant_risque * poids_symboles[i] for i, symbole in enumerate(symboles)}
        valeur_tresorerie = montant_sans_risque
        if date_actuelle > fin_investissement_periode:
            break

        while date_actuelle <= fin_investissement_periode:
            if verifier_presence_date3(titres, date_actuelle, donnees_titres):
                for symbole in symboles:
                    rendement = calculer_rentabilite_1_titre3(symbole, ancienne_date, date_actuelle, donnees_titres)
                    valeurs_titres[symbole] *= (1 + rendement)

                days = (date_actuelle - ancienne_date).days
                valeur_tresorerie *= (1 + taux_sans_risque) ** (days / 365)

                rendement_indice = calculer_rentabilite_1_titre3(indice, ancienne_date, date_actuelle, donnees_titres)
                valeur_actuelle_indice *= (1 + rendement_indice)

                valeur_portefeuille_total[date_actuelle] = sum(valeurs_titres.values()) + valeur_tresorerie
                valeur_indice_total[date_actuelle] = valeur_actuelle_indice

                ancienne_date = date_actuelle

            date_actuelle += pd.Timedelta(days=1)

        debut_investissement_periode = fin_investissement_periode

        montant_portefeuille = sum(valeurs_titres.values()) + valeur_tresorerie

    df_portefeuille_total = pd.DataFrame(list(valeur_portefeuille_total.items()), columns=['Date', 'Valeur'])
    df_portefeuille_total['Date'] = pd.to_datetime(df_portefeuille_total['Date'])
    df_portefeuille_total = df_portefeuille_total.sort_values('Date')

    df_indice_total = pd.DataFrame(list(valeur_indice_total.items()), columns=['Date', 'Valeur'])
    df_indice_total['Date'] = pd.to_datetime(df_indice_total['Date'])
    df_indice_total = df_indice_total.sort_values('Date')

    if len(df_portefeuille_total) == 0:
        return {"erreur": "Aucune donnée disponible pour la simulation"}

    derniere_date = df_indice_total['Date'].max()

    rentabilite_portefeuille_absolue = (df_portefeuille_total['Valeur'].iloc[-1] - montant) / montant
    rentabilite_indice_absolue = (df_indice_total['Valeur'].iloc[-1] - montant) / montant

    nombre_jours_boursiers = len(df_portefeuille_total)
    duree_annees = (fin_investissement - pd.to_datetime(start_date)).days / 365.25

    if duree_annees <= 0:
        rentabilite_portefeuille_annuelle = 0
        rentabilite_indice_annuelle = 0
    else:
        rentabilite_portefeuille_annuelle = (df_portefeuille_total['Valeur'].iloc[-1] / montant) ** (1 / duree_annees) - 1
        rentabilite_indice_annuelle = (df_indice_total['Valeur'].iloc[-1] / montant) ** (1 / duree_annees) - 1

    def max_drawdown(series):
        roll_max = series.cummax()
        drawdown = (series - roll_max) / roll_max
        return drawdown.min()

    max_drawdown_portefeuille = max_drawdown(df_portefeuille_total['Valeur'])
    max_drawdown_indice = max_drawdown(df_indice_total['Valeur'])

    valeur_minimum_portefeuille = df_portefeuille_total['Valeur'].min()
    valeur_maximum_portefeuille = df_portefeuille_total['Valeur'].max()
    valeur_minimum_indice = df_indice_total['Valeur'].min()
    valeur_maximum_indice = df_indice_total['Valeur'].max()

    plt.figure(figsize=(10, 6))
    plt.plot(df_portefeuille_total['Date'], df_portefeuille_total['Valeur'], color='b', label='Valeur du portefeuille', marker='o', markersize=3)
    plt.plot(df_indice_total['Date'], df_indice_total['Valeur'], color='r', label="Valeur de l'indice", marker='s', markersize=3)
    plt.title("\u00c9volution de la valeur du portefeuille compar\u00e9 \u00e0 l'indice du march\u00e9 (jour par jour)")
    plt.xlabel("Date")
    plt.ylabel("Valeur (€)")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    dossier = "../image/graphique"
    os.makedirs(dossier, exist_ok=True)
    nom_fichier = f"simulationdyn_rendement_{user_id}_{nom_simulation}.png"
    chemin_fichier = os.path.join(dossier, nom_fichier)
    plt.savefig(chemin_fichier, dpi=300)
    plt.close()

    return {
        "rentabilite_portefeuille_absolue": rentabilite_portefeuille_absolue / 100,
        "rentabilite_indice_absolue": rentabilite_indice_absolue / 100,
        "rentabilite_portefeuille_annuelle": rentabilite_portefeuille_annuelle / 100,
        "rentabilite_indice_annuelle": rentabilite_indice_annuelle / 100,
        "max_drawdown_portefeuille": max_drawdown_portefeuille / 100,
        "max_drawdown_indice": max_drawdown_indice / 100,
        "nombre_jours_boursiers": nombre_jours_boursiers,
        "valeur_minimum_portefeuille": valeur_minimum_portefeuille,
        "valeur_maximum_portefeuille": valeur_maximum_portefeuille,
        "valeur_minimum_indice": valeur_minimum_indice,
        "valeur_maximum_indice": valeur_maximum_indice,
        "premiere_date": premiere_date.strftime('%Y-%m-%d'),
        "derniere_date": derniere_date.strftime('%Y-%m-%d')
    }

#################################################################################################################################################################################
#################################################################################################################################################################################
######################################### Simulation d'investissement dynamique d'un portefeuille avec un graphique par période  ################################################
#################################################################################################################################################################################
#################################################################################################################################################################################

def simuler_rendement_rapide3(start_date, end_date, duree_estimation, duree_investissement, symboles, poids_risque, indice, montant, taux_sans_risque, taux_benchmark, user_id, nom_simulation, methode,lamb):
    titres = [indice] + symboles
    donnees_titres = charger_donnees3(titres)
    fin_investissement = pd.to_datetime(end_date)
    
    valeur_portefeuille_total = {}
    valeur_indice_total = {}
    
    montant_portefeuille = montant
    valeur_actuelle_indice = montant
    
    current_date = pd.to_datetime(start_date)
    while not verifier_presence_date3(titres, current_date, donnees_titres):
        current_date += pd.Timedelta(days=1)
    premiere_date = current_date
    valeur_portefeuille_total[premiere_date] = montant
    valeur_indice_total[premiere_date] = montant
    fin_investissement_periode = current_date
    while current_date < fin_investissement:
        fin_investissement_periode = fin_investissement_periode + pd.Timedelta(days=duree_investissement)
        end_period = fin_investissement_periode
        if end_period > fin_investissement:
            end_period = fin_investissement
        while not verifier_presence_date3(titres, end_period, donnees_titres):
            end_period -= pd.Timedelta(days=1)
            if end_period < current_date:
                break
        if end_period < current_date:
            break
        
        fin_estimation = current_date - pd.Timedelta(days=1)
        debut_estimation = fin_estimation - pd.Timedelta(days=int(duree_estimation))
        while not verifier_presence_date3(titres, fin_estimation, donnees_titres):
            fin_estimation -= pd.Timedelta(days=1)
            if fin_estimation < debut_estimation:
                break
        while not verifier_presence_date3(titres, debut_estimation, donnees_titres):
            debut_estimation += pd.Timedelta(days=1)
            if debut_estimation > fin_estimation:
                break
        if debut_estimation > fin_estimation:
            continue
        if methode=='sharpe' :
            poids_symboles = maximiser_ratio_sharpe3(symboles, debut_estimation, fin_estimation, taux_benchmark)
        elif methode=='moment_ordre_superieur':
            poids_symboles, score = optimiser_utilite_CARA3(symboles, debut_estimation, fin_estimation, lamb)
        elif methode == 'sortino' :
            poids_symboles = maximiser_ratio_sortino3(symboles, debut_estimation, fin_estimation, taux_benchmark)

        montant_risque = montant_portefeuille * poids_risque
        valeurs_titres = {symbole: montant_risque * poids_symboles[i] for i, symbole in enumerate(symboles)}
        valeur_tresorerie = montant_portefeuille * (1 - poids_risque)
        
        for symbole in symboles:
            rendement = calculer_rentabilite_1_titre3(symbole, current_date, end_period, donnees_titres)
            valeurs_titres[symbole] *= (1 + rendement)
        
        nb_jours = (end_period - current_date).days
        valeur_tresorerie *= (1 + taux_sans_risque) ** (nb_jours / 365)
        
        rendement_indice = calculer_rentabilite_1_titre3(indice, current_date, end_period, donnees_titres)
        valeur_actuelle_indice *= (1 + rendement_indice)
        
        valeur_portefeuille_total[end_period] = sum(valeurs_titres.values()) + valeur_tresorerie
        valeur_indice_total[end_period] = valeur_actuelle_indice
        
        montant_portefeuille = valeur_portefeuille_total[end_period]
        if current_date == end_period:
            current_date += pd.Timedelta(days=1)
        else :
            current_date = end_period
    
    df_portefeuille_total = pd.DataFrame(list(valeur_portefeuille_total.items()), columns=['Date', 'Valeur'])
    df_portefeuille_total['Date'] = pd.to_datetime(df_portefeuille_total['Date'])
    df_portefeuille_total = df_portefeuille_total.sort_values('Date')
    
    df_indice_total = pd.DataFrame(list(valeur_indice_total.items()), columns=['Date', 'Valeur'])
    df_indice_total['Date'] = pd.to_datetime(df_indice_total['Date'])
    df_indice_total = df_indice_total.sort_values('Date')
    
    if len(df_portefeuille_total) == 0:
        return {"erreur": "Aucune donnée disponible pour la simulation"}
    
    derniere_date = df_indice_total['Date'].max()
    
    rentabilite_portefeuille_absolue = (df_portefeuille_total['Valeur'].iloc[-1] - montant) / montant
    rentabilite_indice_absolue = (df_indice_total['Valeur'].iloc[-1] - montant) / montant
    
    nombre_periodes = len(df_portefeuille_total)
    duree_annees = (derniere_date - pd.to_datetime(start_date)).days / 365.25
    
    if duree_annees <= 0:
        rentabilite_portefeuille_annuelle = 0
        rentabilite_indice_annuelle = 0
    else:
        rentabilite_portefeuille_annuelle = (df_portefeuille_total['Valeur'].iloc[-1] / montant) ** (1 / duree_annees) - 1
        rentabilite_indice_annuelle = (df_indice_total['Valeur'].iloc[-1] / montant) ** (1 / duree_annees) - 1
    
    def max_drawdown(series):
        roll_max = series.cummax()
        drawdown = (series - roll_max) / roll_max
        return drawdown.min()
    
    max_drawdown_portefeuille = max_drawdown(df_portefeuille_total['Valeur'])
    max_drawdown_indice = max_drawdown(df_indice_total['Valeur'])
    
    valeur_minimum_portefeuille = df_portefeuille_total['Valeur'].min()
    valeur_maximum_portefeuille = df_portefeuille_total['Valeur'].max()
    valeur_minimum_indice = df_indice_total['Valeur'].min()
    valeur_maximum_indice = df_indice_total['Valeur'].max()
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_portefeuille_total['Date'], df_portefeuille_total['Valeur'], color='b', label='Valeur du portefeuille', marker='o')
    plt.plot(df_indice_total['Date'], df_indice_total['Valeur'], color='r', label="Valeur de l'indice", marker='s')
    plt.title("Évolution de la valeur du portefeuille comparé à l'indice du marché (par période)")
    plt.xlabel("Date")
    plt.ylabel("Valeur (€)")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    dossier = "../image/graphique"
    os.makedirs(dossier, exist_ok=True)
    nom_fichier = f"simulationdyn_rendement_{user_id}_{nom_simulation}.png"
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
        "nombre_periodes_simulees": nombre_periodes,
        "valeur_minimum_portefeuille": valeur_minimum_portefeuille,
        "valeur_maximum_portefeuille": valeur_maximum_portefeuille,
        "valeur_minimum_indice": valeur_minimum_indice,
        "valeur_maximum_indice": valeur_maximum_indice,
        "nombre_jours_boursiers": len(df_portefeuille_total),
        "premiere_date": premiere_date.strftime('%Y-%m-%d'), 
        "derniere_date": derniere_date.strftime('%Y-%m-%d')
    }

#################################################################################################################################################################################
#################################################################################################################################################################################
######################################################### Traitement des arguments ##############################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################
