# API Python pour l'Optimisation de Portefeuille

Cette API permet de récupérer des données financières, d'optimiser des portefeuilles d'actions et de simuler leurs performances à l'aide de méthodes d'optimisation avancées, telles que le ratio de Sharpe, le ratio de Sortino et la fonction d'utilité exponentielle ajustée (CARA). Cette partie n'est pas prévue pour être utilisée telle quelle. Il est recommandé d'utiliser l'interface dédiée sur Moonalyse : [Moonalyse](https://moonalyse.com/fichier_php/accueil.php).

## Base URL
```
https://api-python-znmhag.fly.dev/
```

## Fonctionnalités principales
- **Optimisation de portefeuilles** : Détermine les poids optimaux des actifs selon différentes stratégies.
- **Simulation de performances** : Offre des simulations dynamiques et simples avec génération de graphiques.
- **Récupération de données financières** : Utilise **yfinance** pour accéder aux données de marché en temps réel.

## Endpoints

### 1. Test de fonctionnement


- **Méthode** : `GET`
- **URL** : `/`
- **Description** : Cet endpoint permet de vérifier que l'API est opérationnelle.
- **Réponse** :
  ```json
  "Flask fonctionne !"
  ```

### 2. Optimisation de portefeuille

- **Méthode** : `POST`
- **URL** : `/optimiser`
- **Description** : Cet endpoint optimise un portefeuille en déterminant les poids optimaux des titres selon la méthode choisie (Sharpe, Sortino ou CARA).
- **Paramètres** : titres, date_debut, date_fin, methode, taux_benchmark, lambda (optionnel, uniquement si methode = "moment_ordre_superieur")

### 3. Récupération des données
**Note** : Ces endpoints sont en phase de développement et réservés à un usage interne.

- **Méthode** : `GET`
- **URL** : `/recuperer_tout`
- **Paramètres** : date_debut
- **Description** : Récupère les données de toutes les actions à partir de `date_debut`.

- **Méthode** : `GET`
- **URL** : `/recuperer_un`
- **Paramètres** :date_debut, nom_titre, symbole_titre
- **Description** : Récupère les données d'une seule action.

- **Méthode** : `GET`
- **URL** : `/completer_tout` 
- **Paramètres** : aucun
- **Description** : Met à jour tous les fichiers CSV jusqu'à la date actuelle.
### 4. Simulation dynamique

- **Méthode** : `GET`
- **URL** : `/simulation_dynamique`
- **Description** : Cet endpoint simule un investissement avec une réoptimisation périodique des poids. Il génére également un graphique.
- **Paramètres** : date_debut, date_fin, duree_estimation, duree_investissement, titres, niveau_risque, indice, user_id, montant, taux_sans_risque_annuel, taux_benchmark, nom_simulation, graphique_option, methode, lambda (optionnel, uniquement si methode = "moment_ordre_superieur")

### 5. Simulation simple
- **Méthode** : `GET`
- **URL** : `/simulation_simple`
- **Description** : Cet endpoint simule un investissement basé sur une optimisation initiale, avec la possibilité de générer un graphique.
- **Paramètres** : date_debut, date_fin, w, titres, poids_str, indice, user_id, montant, taux_sans_risque, nom_simulation

## Notes techniques
- **Framework** : Développée avec **Flask**, un framework Python léger et flexible.
- **Données financières** : Les données sont récupérées via **yfinance**, une bibliothèque Python pour les données de Yahoo Finance.
- **Stockage** : Les historiques d'actions sont stockés dans des fichiers CSV dans le dossier `historique_action`.
- **Méthodes d'optimisation** : Supporte le ratio de Sharpe, le ratio de Sortino et l'utilité CARA (Coefficient d'Aversion au Risque Absolu).
