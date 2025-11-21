# AI Clustering & Insight Dashboard

Une application interactive **Streamlit** pour l'exploration de données, le clustering, et l'analyse automatique via LLM (Groq).

## 📋 Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Fonctionnalités](#fonctionnalités)

---

## 🎯 Vue d'ensemble

**AI Clustering & Insight Dashboard** combine plusieurs techniques de data science pour :

1. **Charger** et prétraiter des données (CSV, Excel, ou échantillons)
2. **Réduire** la dimensionalité (PCA, t-SNE, UMAP)
3. **Regrouper** les données avec plusieurs algorithmes (KMeans, DBSCAN, GMM, OPTICS, KMedoids)
4. **Évaluer** la qualité des clusters (Silhouette, Davies-Bouldin, Calinski-Harabasz)
5. **Profiler** chaque cluster (statistiques, distributions, résumés)
6. **Générer** des analyses textuelles via LLM Groq

### Points forts

✅ **Architecture modulaire** : séparation claire entre logique métier et UI  
✅ **Groq LLM intégré** : résumés automatiques en français  
✅ **Visualisations interactives** : Plotly 2D/3D, radar charts, heatmaps  
✅ **Configuration flexible** : hyperparamètres ajustables en temps réel  
✅ **Gestion robuste des erreurs** : fallback gracieux si bibliothèques optionnelles manquent  

---

## 🏗️ Architecture

```
projet_2_data_mining/
├── app.py                      # App Streamlit (UI uniquement)
├── requirements.txt            # Dépendances Python
├── README.md                   # Ce fichier
│
├── src/                        # Modules de logique métier
│   ├── __init__.py
│   ├── preprocessing.py        # Chargement, imputation, encoding, scaling
│   ├── dimensionality.py       # PCA, t-SNE, UMAP
│   ├── clustering.py           # Algorithmes de clustering + scoring
│   ├── profiling.py            # Profils de clusters et résumés textuels
│   └── groq_integration.py     # Intégration API Groq
│
└── env/                        # Environnement Python (venv)
```

### Séparation des responsabilités

| Module | Responsabilité |
|--------|-----------------|
| **preprocessing.py** | Chargement données, imputation, one-hot encoding, scaling StandardScaler |
| **dimensionality.py** | PCA, t-SNE, UMAP (optionnel) |
| **clustering.py** | KMeans, DBSCAN, GMM, OPTICS, KMedoids (optionnel) ; calcul silhouette, Davies-Bouldin, Calinski-Harabasz |
| **profiling.py** | Génération profils clusters, résumés textuels template |
| **groq_integration.py** | Résolution clé API Groq, appel LLM Groq |
| **app.py** | Interface utilisateur Streamlit, mise en page, interactions |

---

## 💻 Installation

### Prérequis

- **Python 3.8+**
- **pip** ou **conda**

### Étapes

1. **Cloner ou télécharger** le projet :
   ```bash
   cd c:\Users\lanouar\Desktop\projet_2_data_mining
   ```

2. **Créer un environnement virtuel** (recommandé) :
   ```powershell
   python -m venv env
   env\Scripts\Activate.ps1
   ```

3. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

4. **Installer Groq** (pour LLM) :
   ```bash
   pip install groq
   ```

### Dépendances principales

- **streamlit** : framework UI
- **scikit-learn** : clustering, PCA, preprocessing
- **pandas, numpy** : manipulation données
- **plotly** : visualisations interactives
- **umap-learn** : dimensionality reduction (optionnel)
- **hdbscan** : clustering robuste (optionnel)
- **groq** : LLM integration

Voir `requirements.txt` pour la liste complète.

---

## 🚀 Utilisation

### Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre dans votre navigateur à `http://localhost:8501`.

### Workflow type

1. **Charger des données**
   - Choisir un dataset échantillon (iris, wine) OU uploader un CSV/Excel
   - Aperçu automatique des 5 premières lignes

2. **Configurer la réduction de dimensionalité**
   - Sélectionner technique : PCA, t-SNE, UMAP
   - Choisir nombre de dimensions : 2 ou 3

3. **Configurer le clustering**
   - Choisir algorithme : KMeans, DBSCAN, GMM, OPTICS, KMedoids
   - Ajuster hyperparamètres (n_clusters, eps, min_samples) avec les sliders

4. **Visualiser résultats**
   - Projection 2D/3D des clusters
   - Métriques de qualité (Silhouette, Davies-Bouldin, Calinski-Harabasz)
   - Profils détaillés par cluster (expandable)

5. **(Optionnel) Générer analyse LLM**
   - Cocher "Activer résumé via LLM (Groq)"
   - Fournir clé Groq API dans la barre latérale
   - L'app génère un résumé textuel automatique
   - Télécharger le rapport en `.txt`

6. **Explorer visualisations avancées**
   - Radar chart : comparer clusters sur variables numériques
   - Heatmap : moyennes des features par cluster

7. **Exporter résultats**
   - Télécharger CSV avec clusters assignés

---

## ✨ Fonctionnalités

### 1. Prétraitement automatique
- Détection automatique features numériques/catégorielles
- Imputation valeurs manquantes (médiane pour numériques)
- One-hot encoding catégories (≤20 modalités)
- Frequency encoding catégories haute cardinalité
- Scaling StandardScaler

### 2. Réduction dimensionnalité
| Méthode | Avantages | Inconvénients |
|---------|-----------|---------------|
| **PCA** | Rapide, linéaire, variance expliquée | Pas de non-linéarité |
| **t-SNE** | Capture non-linéarité, clusters visibles | Lent, distances absolues sans sens |
| **UMAP** | Non-linéaire, rapide, scalable | Optionnel, moins d'interprétabilité |

### 3. Clustering1
| Algorithme | Type | Avantages | Inconvénients |
|-----------|------|-----------|---------------|
| **KMeans** | Centroïde | Rapide, scalable, k fixé | Clusters sphériques |
| **DBSCAN** | Densité | Détecte bruit, clusters arbitraires | Paramètres eps/min_samples sensibles |
| **GMM** | Probabiliste | Probabilités, clusters elliptiques | Lent, assomptions |
| **OPTICS** | Densité | Robuste, moins params | Complexe, lent |
| **KMedoids** | Centroïde (médoïdes) | Robuste aux outliers | Lent, optionnel |

### 4. Métriques de qualité
- **Silhouette** : [-1, 1] (proche 1 = bon)
- **Davies-Bouldin** : [0, ∞) (proche 0 = bon)
- **Calinski-Harabasz** : [0, ∞) (élevé = bon)

### 5. Profils clusters
Pour chaque cluster :
- Taille (nombre d'observations)
- Top 5 features numériques (moyennes)
- Distributions catégories principales (top 3 par catégorie)
- Résumé textuel auto-généré

### 6. Intégration Groq LLM
- Appelle modèle `llama-3.3-70b-versatile`
- Génère résumé synthétique français
- Téléchargement rapport `.txt`
- Gestion clé API sécurisée (sidebar password input)

### 7. Visualisations
- **Projection 2D/3D** : scatter plot coloré par cluster
- **Indices métriques** : tableau comparatif
- **Radar chart** : comparer profils clusters
- **Heatmap** : moyennes features/clusters
- **Export CSV** : résultats complets

---
