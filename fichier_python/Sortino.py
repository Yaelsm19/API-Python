import numpy as np
import pandas as pd
from scipy.optimize import minimize


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



def calculer_co_semi_variance(symbole1, symbole2, start_date, end_date, B):
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

    d1 = np.minimum(r1 - B, 0)
    d2 = np.minimum(r2 - B, 0)

    co_semi_var = (d1 * d2).sum() / (T - 1)

    if pd.isna(co_semi_var):
        co_semi_var = 0.0

    return float(co_semi_var)


def calculer_matrice_semi_variance(symboles, start_date, end_date, B):
    n = len(symboles)
    matrice_covariance = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            cov = calculer_co_semi_variance(symboles[i], symboles[j], start_date, end_date, B)
            matrice_covariance[i][j] = cov
            matrice_covariance[j][i] = cov
    return matrice_covariance




def calculer_semi_risque_portefeuille(w, symboles, start_date, end_date, B):
    w_transpose = np.transpose(w)
    matrice_covariance = calculer_matrice_semi_variance(symboles, start_date, end_date, B)
    variance_portefeuille = np.dot(np.dot(w_transpose, matrice_covariance), w)
    return variance_portefeuille**0.5

def maximiser_ratio_sortino(symboles, start_date, end_date, B):
    rf = (1+B)**(1/365)-1
    rendement_matrice = calculer_matrice_rentabilite(symboles, start_date, end_date).flatten()
    covariance_matrice = calculer_matrice_semi_variance(symboles, start_date, end_date, B)
    n = len(rendement_matrice)

    def ratio_sortino(w):
        rendement = np.dot(w, rendement_matrice)
        risque = np.sqrt(np.dot(w.T, np.dot(covariance_matrice, w)))
        return (rendement - rf) / risque

    def gradient_ratio_sortino(w):
        w = np.array(w)
        risque_carre = np.dot(w.T, np.dot(covariance_matrice, w))
        risque = np.sqrt(risque_carre)

        num1 = rendement_matrice * risque
        num2 = np.dot(w.T, rendement_matrice - rf)
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
        return poids
    else:
        raise ValueError("Échec de l'optimisation :", resultat.message)

