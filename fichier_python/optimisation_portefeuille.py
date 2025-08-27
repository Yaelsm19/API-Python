from datetime import datetime
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import sys
import json

#################################################################################################################################################################################
#################################################################################################################################################################################
################################################################# Calcul de la rentabilité moyenne ###############################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################

def calculer_rentabilite(symbole, start_date, end_date):
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

def calculer_matrice_rentabilite(symboles, start_date, end_date):
    rentabilites = []
    for s in symboles:
        r = calculer_rentabilite(s, start_date, end_date)
        if r is None:
            print(f"Attention : rentabilité manquante pour {s}, valeur 0 par défaut")
            r = 0.0
        rentabilites.append(r)
    return np.array(rentabilites)

#################################################################################################################################################################################
#################################################################################################################################################################################
############################################################ Calcul de la variance, covariance, écart-type ######################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################

def calculer_covariance(symbole1, symbole2, start_date, end_date):
    df1 = pd.read_csv(
        f"../fichier_python/historique_action/{symbole1}_cloture.csv",
        encoding='utf-8',
        parse_dates=['Date'],
        index_col='Date'
    )
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

def calculer_ecart_type(symbole, start_date, end_date):
    return calculer_covariance(symbole, symbole, start_date, end_date) ** 0.5

def calculer_matrice_covariance(symboles, start_date, end_date):
    n = len(symboles)
    matrice_covariance = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            cov = calculer_covariance(symboles[i], symboles[j], start_date, end_date)
            matrice_covariance[i][j] = cov
            matrice_covariance[j][i] = cov
    return matrice_covariance

def calculer_risque_portefeuille(w, symboles, start_date, end_date):
    w_transpose = np.transpose(w)
    matrice_covariance = calculer_matrice_covariance(symboles, start_date, end_date)
    variance_portefeuille = np.dot(np.dot(w_transpose, matrice_covariance), w)
    return variance_portefeuille**0.5

#################################################################################################################################################################################
#################################################################################################################################################################################
############################################################ Calcul de la semi-variance, semi-covariance, semi-écart-type #######################################################
#################################################################################################################################################################################
#################################################################################################################################################################################

def calculer_co_semi_variance(symbole1, symbole2, start_date, end_date, taux_benchmark):
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


def calculer_matrice_semi_variance(symboles, start_date, end_date, taux_benchmark):
    n = len(symboles)
    matrice_covariance = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            cov = calculer_co_semi_variance(symboles[i], symboles[j], start_date, end_date, taux_benchmark)
            matrice_covariance[i][j] = cov
            matrice_covariance[j][i] = cov
    return matrice_covariance

def calculer_semi_risque_portefeuille(w, symboles, start_date, end_date, taux_benchmark):
    w_transpose = np.transpose(w)
    matrice_covariance = calculer_matrice_semi_variance(symboles, start_date, end_date, taux_benchmark)
    variance_portefeuille = np.dot(np.dot(w_transpose, matrice_covariance), w)
    return variance_portefeuille**0.5

#################################################################################################################################################################################
#################################################################################################################################################################################
################################################################################# Calcul de la matrice du skewness ##############################################################
#################################################################################################################################################################################
#################################################################################################################################################################################

def calculer_skewness_matrice(symboles, start_date, end_date):
    donnees = []
    for symbole in symboles:
        chemin = f"../fichier_python/historique_action/{symbole}_cloture.csv"
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

def calculer_kurtosis_matrice(symboles, start_date, end_date):
    donnees = []
    for symbole in symboles:
        chemin = f"../fichier_python/historique_action/{symbole}_cloture.csv"
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
###################################################### Calcul et stockage de la rentabilité et de l'écart-type pour chaque action ###############################################
#################################################################################################################################################################################
#################################################################################################################################################################################

def process_une_action(symbole, start_date, end_date):
    rentabilite = calculer_rentabilite(symbole, start_date, end_date)
    ecart_type = calculer_ecart_type(symbole, start_date, end_date)
    return (rentabilite, ecart_type)

def process_tout_action(symboles, start_date, end_date):
    data = []
    for symbole in symboles:
        rentabilite, ecart_type = process_une_action(symbole, start_date, end_date)
        data.append({'Symbole': symbole, 'Rentabilite': f"{rentabilite:.6f}", 'Ecart_Type': f"{ecart_type:.4f}"})
    df_process = pd.DataFrame(data)
    return df_process

#################################################################################################################################################################################
#################################################################################################################################################################################
################################################################## Méthode ratio de sharpe ######################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################

def maximiser_ratio_sharpe(symboles, start_date, end_date, taux_benchmark):
    rendements = calculer_matrice_rentabilite(symboles, start_date, end_date).flatten()
    covariance = calculer_matrice_covariance(symboles, start_date, end_date)
    n = len(rendements)
    taux_benchmark_journalier = (1+taux_benchmark)**(1/365)-1

    def ratio_sharpe(w, taux_benchmark_journalier):
        rendement_portefeuille = np.dot(w, rendements)
        risque_portefeuille = np.sqrt(np.dot(w.T, np.dot(covariance, w)))
        return (rendement_portefeuille - taux_benchmark_journalier) / risque_portefeuille if risque_portefeuille != 0 else 0


    contrainte_somme = {
        'type': 'eq',
        'fun': lambda w: float(np.sum(w) - 1)
    }

    bornes = [(0, 1) for _ in range(n)]
    w0 = np.ones(n) / n

    resultat = minimize(
        lambda w: -ratio_sharpe(w, taux_benchmark_journalier),
        w0,
        method='SLSQP',
        bounds=bornes,
        constraints=contrainte_somme
    )

    if resultat.success:
        poids = resultat.x
        rendement_final = np.dot(poids, rendements)
        risque_final = calculer_risque_portefeuille(poids, symboles, start_date, end_date)
        data = []
        for i in range(len(symboles)):
            if poids[i] > 0.00001:
                data.append({'Symbole': symboles[i], 'Poids': f"{poids[i]*100:.3f}"})
        df_optimisation = pd.DataFrame(data)
        df_portefeuille = pd.DataFrame([{"Symbole": f"Portefeuille", "Rendement": f"{rendement_final:.6f}", "Risque": f"{risque_final:.4f}"}])
        return df_optimisation, df_portefeuille
    else:
        raise ValueError("Échec de l'optimisation :", resultat.message)
        
#################################################################################################################################################################################
#################################################################################################################################################################################
################################################################# Méthode ratio de Sortino ######################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################

def maximiser_ratio_sortino(symboles, start_date, end_date, taux_benchmark):
    taux_benchmark_journalier = (1+taux_benchmark)**(1/365)-1
    rendement_matrice = calculer_matrice_rentabilite(symboles, start_date, end_date)
    covariance_matrice = calculer_matrice_semi_variance(symboles, start_date, end_date, taux_benchmark)
    n = len(rendement_matrice)

    def ratio_sortino(w):
        rendement = np.dot(w, rendement_matrice)
        risque = np.sqrt(np.dot(w.T, np.dot(covariance_matrice, w)))
        return (rendement - taux_benchmark_journalier) / risque

    def gradient_ratio_sortino(w):
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
        lambda w: -ratio_sortino(w),
        w0,
        method='SLSQP',
        bounds=bornes,
        constraints=contrainte_somme,
        jac=gradient_ratio_sortino
    )

    if resultat.success:
        poids = resultat.x
        rendement_final = np.dot(poids, rendement_matrice)
        risque_final = calculer_semi_risque_portefeuille(poids, symboles, start_date, end_date, taux_benchmark)
        data = []
        for i in range(len(symboles)):
            if poids[i] > 0.00001:
                data.append({'Symbole': symboles[i], 'Poids': f"{poids[i]*100:.3f}"})
        df_optimisation = pd.DataFrame(data)
        df_portefeuille = pd.DataFrame([{"Symbole": f"Portefeuille", "Rendement": f"{rendement_final:.6f}", "Risque": f"{risque_final:.4f}"}])
        return df_optimisation, df_portefeuille
    else:
        raise ValueError("Échec de l'optimisation :", resultat.message)
        
#################################################################################################################################################################################
#################################################################################################################################################################################
######################################################### Méthode utilite exponentielle #########################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################

def gradient_utilite(w, lambd, matrice_rentabilite, matrice_covariance, matrice_skewness, matrice_kurtosis):
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
def utilite_exponentielle(w, lambd, matrice_rentabilite, matrice_covariance, matrice_skewness, matrice_kurtosis):
    E_w = np.dot(w.T, matrice_rentabilite)
    M_w = np.dot(np.dot(w.T, matrice_covariance), w)
    S_w = np.einsum('i,j,k,ijk->', w, w, w, matrice_skewness)
    K_w = np.einsum('i,j,k,l,ijkl->', w, w, w, w, matrice_kurtosis)

    ajustement = 1 + ((lambd**2) / 2) * M_w - ((lambd**3) / 6) * S_w + ((lambd**4) / 24) * K_w
    utilite = -np.exp(-lambd * E_w) * ajustement
    return -utilite


def optimiser_utilite_CARA(symboles, start_date, end_date, lambd):
    matrice_rentabilite = calculer_matrice_rentabilite(symboles, start_date, end_date).flatten()
    matrice_covariance = calculer_matrice_covariance(symboles, start_date, end_date)
    matrice_skewness = calculer_skewness_matrice(symboles, start_date, end_date)
    matrice_kurtosis = calculer_kurtosis_matrice(symboles, start_date, end_date)
    n = len(symboles)
    w0 = np.ones(n) / n
    contraintes = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1, 'jac': lambda w: np.ones_like(w)}]
    bornes = [(0, 1) for _ in range(n)]

    def objectif(w, lambd, matrice_rentabilite, matrice_covariance, matrice_skewness, matrice_kurtosis):
        return utilite_exponentielle(w, lambd, matrice_rentabilite, matrice_covariance, matrice_skewness, matrice_kurtosis)

    resultat = minimize(objectif, w0, method='SLSQP',
                        jac=gradient_utilite,
                        args=(lambd, matrice_rentabilite, matrice_covariance, matrice_skewness, matrice_kurtosis),
                        bounds=bornes, constraints=contraintes,
                        options={'disp': False, 'maxiter': 1000})

    if resultat.success:
        poids = resultat.x
        rendement_final = np.dot(poids, matrice_rentabilite)
        risque_final = calculer_risque_portefeuille(poids, symboles, start_date, end_date)
        data = []
        for i in range(n):
            if poids[i] > 0.00001:
                data.append({'Symbole': symboles[i], 'Poids': f"{poids[i]*100:.3f}"})
        df_optimisation = pd.DataFrame(data)
        df_portefeuille = pd.DataFrame([{"Symbole": f"Portefeuille", "Rendement": f"{rendement_final:.6f}", "Risque": f"{risque_final:.4f}"}])
        return df_optimisation, df_portefeuille
        
    raise ValueError("Optimisation échouée : " + resultat.message)

#################################################################################################################################################################################
#################################################################################################################################################################################
####################################################################### traitement des arguments ################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################

def traitement_arguments() :
    if len(sys.argv) !=7 and len(sys.argv)!=6:
        print("Nombre d'arguments incorrect.")
        sys.exit(1)
    titres = sys.argv[1].split(',')
    date_debut = datetime.strptime(sys.argv[2], "%Y-%m-%d")
    date_fin = datetime.strptime(sys.argv[3], "%Y-%m-%d")
    methode = sys.argv[4]
    if methode not in ['sharpe', 'moment_ordre_superieur', 'sortino']:
        print("Méthode non reconnue. Utilisez 'sharpe' ou 'moment_ordre_superieur'.")
        sys.exit(1)
    taux_benchmark = float(sys.argv[5])
    if methode=="sharpe" :
            df_optimisation, df_portefeuille = maximiser_ratio_sharpe(titres, date_debut, date_fin, taux_benchmark)
    elif methode=="moment_ordre_superieur":
            lamb = float(sys.argv[6])
            df_optimisation, df_portefeuille = optimiser_utilite_CARA(titres, date_debut, date_fin, lamb)
    elif methode=="sortino" :
        df_optimisation, df_portefeuille = maximiser_ratio_sortino(titres, date_debut, date_fin, taux_benchmark)
    df_process = process_tout_action(titres, date_debut, date_fin)
    reponse = {
    "optimisation": df_optimisation.to_dict(orient='records'),
    "process": df_process.to_dict(orient='records'),
    "portefeuille": df_portefeuille.to_dict(orient='records')
    }
    print(json.dumps(reponse))

traitement_arguments()