"""
AI Clustering & Insight Dashboard
Main Streamlit app combining data science algorithms with interactive UI.

Modular architecture:
- preprocessing.py: data loading, imputation, encoding
- dimensionality.py: PCA, t-SNE, UMAP
- clustering.py: KMeans, DBSCAN, GMM, OPTICS, KMedoids + scoring
- profiling.py: cluster profiling and text summaries
- groq_integration.py: LLM summaries via Groq

Usage:
    pip install streamlit scikit-learn pandas numpy plotly umap-learn hdbscan scikit-learn-extra groq
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Import business logic modules
from src.preprocessing import load_sample_data, preprocess
from src.dimensionality import reduce_dim, UMAP_AVAILABLE
from src.clustering import run_clustering, cluster_scores, KMEDOIDS_AVAILABLE
from src.profiling import profile_clusters
from src.groq_integration import resolve_groq_api_key, generate_groq_summary, GROQ_AVAILABLE

st.set_page_config(layout="wide", page_title="AI Clustering & Insight Dashboard")
st.title("🤖 AI Clustering & Insight Dashboard")
# ========================= UI: SIDEBAR =========================

with st.sidebar:
    st.header("Data")
    data_source = st.radio("Charger des données:", ("Sample: iris", "Sample: wine", "Upload CSV"))
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv', 'xlsx'])
    else:
        uploaded_file = None

    st.markdown("---")
    st.header("Reduction & Clustering")
    dr_method = st.selectbox("Technique de réduction", options=["PCA", "t-SNE"] + (["UMAP"] if UMAP_AVAILABLE else []))
    dr_components = st.selectbox("Dimensions de projection", options=[2, 3], index=0)
    clustering_alg = st.selectbox("Algorithme de clustering", options=["KMeans", "DBSCAN", "GMM", "OPTICS"] + (["KMedoids"] if KMEDOIDS_AVAILABLE else []))

    # hyperparameters
    st.subheader("Hyperparamètres")
    if clustering_alg in ['KMeans', 'GMM', 'KMedoids']:
        n_clusters = st.slider("n_clusters (k)", min_value=2, max_value=20, value=3)
    else:
        n_clusters = None
    eps = st.slider("eps (DBSCAN)", min_value=0.01, max_value=5.0, value=0.5, step=0.01)
    min_samples = st.slider("min_samples (DBSCAN/OPTICS)", min_value=1, max_value=50, value=5)

    st.markdown("\n")
    st.button("Run Analysis", key='run')

    # GPT / LLM configuration (Groq is the default provider)
    st.markdown("---")
    st.header("LLM Summaries (Groq)")
    enable_gpt = st.checkbox("Activer résumé via LLM (Groq) (clé requise)", value=False)
    if enable_gpt:
        groq_key_input = st.text_input("Groq API Key (gsk-...)", type="password")
    else:
        groq_key_input = None

# ========================= BUSINESS LOGIC: DATA LOADING ========================= 
if data_source == "Sample: iris":
    df = load_sample_data('iris')
elif data_source == "Sample: wine":
    df = load_sample_data('wine')
elif data_source == "Upload CSV" and uploaded_file is not None:
    try:
        if str(uploaded_file.name).endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Impossible de lire le fichier: {e}")
        st.stop()
else:
    st.info("Chargez des données échantillons ou importez un CSV pour commencer.")
    st.stop()

st.write("### Aperçu des données")
st.dataframe(df.head())

# ========================= BUSINESS LOGIC: PREPROCESSING ========================= 
X_scaled, numeric_features, categorical_features = preprocess(df)

st.write(f"Features détectées: {len(X_scaled.columns)} (numériques: {len(numeric_features)}, catégorielles encodées: {len(categorical_features)})")

# ========================= BUSINESS LOGIC: DIMENSIONALITY REDUCTION ========================= 
proj = reduce_dim(X_scaled, method=dr_method, n_components=dr_components)
proj_df = pd.DataFrame(proj, columns=[f"dim_{i+1}" for i in range(proj.shape[1])])

# ========================= BUSINESS LOGIC: CLUSTERING ========================= 
cluster_kwargs = {}
if n_clusters is not None:
    cluster_kwargs['n_clusters'] = n_clusters
cluster_kwargs['eps'] = eps
cluster_kwargs['min_samples'] = min_samples
labels = run_clustering(X_scaled.values, alg=clustering_alg, **cluster_kwargs)

# ========================= BUSINESS LOGIC: SCORING & PROFILING ========================= 
scores = cluster_scores(X_scaled.values, labels)
profiles = profile_clusters(df, X_scaled, labels, numeric_features)

# ========================= UI: VISUALIZATION LAYOUT =========================
col1, col2 = st.columns((2, 1))

with col1:
    st.subheader("Projection & Clusters")
    if dr_components == 2:
        fig = px.scatter(proj_df, x='dim_1', y='dim_2', color=labels.astype(str), hover_data=[df.index], title='Projection 2D des clusters')
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.scatter_3d(proj_df, x='dim_1', y='dim_2', z='dim_3', color=labels.astype(str), hover_data=[df.index], title='Projection 3D des clusters')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Comparaison des indices de clustering")
    metrics_df = pd.DataFrame({
        'metric': ['silhouette', 'davies_bouldin', 'calinski_harabasz'],
        'value': [scores.get('silhouette'), scores.get('davies_bouldin'), scores.get('calinski_harabasz')]
    })
    st.table(metrics_df)

with col2:
    st.subheader("Profil des clusters")
    st.write(f"Nombre de clusters détectés: {scores['n_clusters']}")
    st.write("Taille par label (inclut -1 pour bruit si présent):")
    st.write(scores['sizes'])

    # show profiles
    for c, prof in profiles.items():
        with st.expander(f"Cluster {c} — {prof['size']} items"):
            st.write(prof['summary'])
            st.write("Top numeric means:")
            st.write(prof['top_numeric_means'])
            st.write("Top categorical distributions:")
            st.write(prof['top_categorical'])


# ========================= UI: HEATMAP ========================= 
st.subheader("Heatmap: moyennes par cluster")
heat_df = df.copy()
heat_df['_cluster'] = labels
cluster_means = heat_df.groupby('_cluster')[numeric_features].mean()
if not cluster_means.empty:
    fig = px.imshow(cluster_means.T, labels=dict(x='cluster', y='feature', color='mean'), x=cluster_means.index.astype(str))
    st.plotly_chart(fig, use_container_width=True)

# ========================= UI: EXPORT =========================
st.subheader("Export")
export_df = df.copy()
export_df['_cluster'] = labels
csv = export_df.to_csv(index=False).encode('utf-8')
st.download_button(label="Télécharger résultats (CSV)", data=csv, file_name='clustered_results.csv', mime='text/csv')

# ========================= BUSINESS LOGIC: GROQ LLM SUMMARY ========================= 
if enable_gpt:
    if not GROQ_AVAILABLE:
        st.error('Le package `groq` n\'est pas installé. Installez-le (`pip install groq`) pour utiliser Groq.')
    else:
        resolved_key = resolve_groq_api_key(sidebar_input=groq_key_input)
        if not resolved_key:
            st.error('Clé Groq introuvable. Définissez-la dans la barre latérale ou via la variable d\'environnement `GROQ_API_KEY`.')
        else:
            try:
                llm_output = generate_groq_summary(profiles, resolved_key)
                st.markdown("---")
                st.subheader("Résumé LLM (Groq)")
                st.write(llm_output)

                st.download_button(
                    label="Télécharger le rapport LLM",
                    data=llm_output,
                    file_name="rapport_llm.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"Erreur lors de l'appel au LLM : {e}")

st.success("Analyse terminée — modifiez les hyperparamètres et relancez pour comparer.")

